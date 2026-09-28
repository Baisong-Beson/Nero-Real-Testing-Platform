"""Offline log/filter replay and fixed recorded-image model probes. No ROS/CAN IO."""
import asyncio
import json
from pathlib import Path
import threading
import cv2
import numpy as np
from act_eval_workbench import common as m,controller as c
from act_eval_workbench.backend import ModelServer
from nero_pi05_bridge.policy_client import PolicyClient

def main():
    trial=m.RUNS/'20260923_193307_937712173_formal'
    directory=m.new_session('readonly_oscillation_audit')
    print('AUDIT_DIRECTORY',directory,flush=True)
    events=[json.loads(s) for s in (trial/'execution/events.jsonl').read_text().splitlines()]
    all_commands=[e for e in events if e['event']=='command_dispatch']
    commands=[e for e in all_commands if e['chunk']>=0]
    chunks={e['chunk']:e for e in events if e['event']=='policy_chunk'}
    t0=all_commands[0]['wall_time_s'];times=np.array([e['wall_time_s']-t0 for e in commands])
    actual=np.array([e['feedback_rad'] for e in commands]);sent=np.array([e['joints_rad'] for e in commands])
    actions=np.array([chunks[e['chunk']]['actions'][e['step']] for e in commands])
    raw=np.stack([actions[:,:7],actions[:,8:15]],axis=1);late=times>=20
    report=dict(trial=str(trial),physical_motion_executed=False,commands=len(all_commands),chunks=len(chunks),
        duration_s=all_commands[-1]['wall_time_s']-t0,
        max_control_interval_s=max(np.diff([e['wall_time_s'] for e in all_commands])),
        inference_ms_percentiles=np.percentile([e['capture']['inference_ms'] for e in chunks.values()],[50,95,100]).tolist(),
        raw_max_excursion_deg=float(np.rad2deg(np.abs(raw-np.array(all_commands[0]['feedback_rad']))).max()),
        late_raw_target_max_error_deg=float(np.rad2deg(np.abs(raw[late]-actual[late])).max()),
        late_lead_clip_fraction=float(np.mean(np.abs(raw[late]-actual[late])>c.BOUNDS['tracking_target_lead_rad'])),
        late_max_tracking_error_deg=float(np.rad2deg(np.abs(sent[late]-actual[late])).max()),
        windows=[],replay={},model_probes={})
    for start,end in ((0,10),(10,20),(20,30),(30,40),(40,50),(50,60)):
        mask=(times>=start)&(times<end)
        report['windows'].append(dict(seconds=[start,end],
            actual_range_deg=np.rad2deg(np.ptp(actual[mask],axis=0)).tolist(),
            raw_range_deg=np.rad2deg(np.ptp(raw[mask],axis=0)).tolist(),
            sent_range_deg=np.rad2deg(np.ptp(sent[mask],axis=0)).tolist(),
            raw_target_max_error_deg=float(np.rad2deg(np.abs(raw[mask]-actual[mask])).max()),
            lead_clip_fraction=float(np.mean(np.abs(raw[mask]-actual[mask])>.02))))
    # Hold the recorded observations and predictions fixed. This attributes the
    # filter's immediate effect; it is not a simulated new physical closed loop.
    original=dict(c.BOUNDS);outputs={}
    for name,changes in [('baseline',{}),('without_lead_clip',dict(tracking_target_lead_rad=10.)),
                         ('double_speed_cap',dict(joint_velocity_rad_s=.30)),
                         ('double_acceleration_cap',dict(joint_acceleration_rad_s2=.60))]:
        try:
            c.BOUNDS.update(original);c.BOUNDS.update(changes)
            first=all_commands[0];slew=c.Slew(first['joints_rad'],first['gripper_width_m'])
            previous=next(e for e in reversed(all_commands) if e['chunk']==-1)['wall_time_s'];pred=[]
            for e,a in zip(commands,actions):
                aq=np.stack([a[:7],a[8:15]]);aw=.1*(1-np.clip(a[[7,15]],0,1))
                slew,q,w,_=c.advance_after_delay(slew,aq,aw,np.array(e['feedback_rad']),e['feedback_gripper_width_m'],e['wall_time_s']-previous)
                previous=e['wall_time_s'];pred.append(q)
            outputs[name]=np.array(pred)
        finally:c.BOUNDS.update(original)
        report['replay'][name]=dict(late_max_difference_from_baseline_deg=float(np.rad2deg(np.abs(outputs[name][late]-outputs['baseline'][late])).max()),
            late_max_difference_from_recorded_deg=float(np.rad2deg(np.abs(outputs[name][late]-sent[late])).max()))
    phase=[]
    for step in range(4):
        mask=late & (np.array([e['step'] for e in commands])==step)
        phase.append(dict(step=step,mean_raw_minus_actual_deg=np.rad2deg((raw[mask]-actual[mask]).mean(0)).tolist()))
    report['late_phase_offsets']=phase
    for arm in (0,1):
        values=raw[late,arm,6];d=np.diff(values);sign=np.sign(d[np.abs(d)>np.deg2rad(.03)])
        report.setdefault('late_joint7_direction_reversals',[]).append(int(np.sum(sign[1:]!=sign[:-1])))
    report['replay_scope']='Recorded feedback/predictions held fixed; no claim about changed closed-loop task success.'
    m.write(directory/'analysis.json',report)
    print('FILTER_REPLAY',json.dumps(report['replay']),flush=True)
    print('PHASE_OFFSETS_J7',[(x['step'],[x['mean_raw_minus_actual_deg'][a][6] for a in (0,1)]) for x in phase],flush=True)

    chosen=min(chunks.values(),key=lambda e:abs(e['wall_time_s']-t0-35))
    frames=[json.loads(s) for s in (trial/'execution/video_frames.jsonl').read_text().splitlines()]
    stamp=chosen['capture']['image_source_unix_s'];frame=min(frames,key=lambda e:abs(e['source_unix_s']-stamp))
    cap=cv2.VideoCapture(str(trial/'execution/main_camera.avi'));cap.set(cv2.CAP_PROP_POS_FRAMES,frame['frame']);ok,bgr=cap.read();cap.release()
    if not ok:raise RuntimeError('Cannot decode recorded evidence frame')
    rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB);state=np.array(chosen['capture']['state'],np.float32)
    cv2.imwrite(str(directory/'probe_recorded_frame.png'),bgr)
    report['probe_source']=dict(chunk=chosen['chunk'],time_s=chosen['wall_time_s']-t0,frame=frame['frame'],
        timestamp_difference_s=frame['source_unix_s']-stamp,encoding_note='Recorded MJPG is lossy, not original inference JPEG.')
    saved=dict(rgb=rgb,state=state);server=ModelServer(directory,lambda s:print(s,flush=True))
    async def probe(port):
        client=PolicyClient('127.0.0.1',port,inference_timeout_sec=10)
        try:
            await client.connect();answers=[]
            for _ in range(10):
                reply=await client.infer({'observation/exterior_image_1_left':rgb,'observation/state':state})
                answers.append(np.array(reply['actions']))
            return np.array(answers)
        finally:await client.close()
    try:
        for model in ('mpi-base','p0','g2'):
            port=server.ensure(m.Settings(model=model),threading.Event());a=asyncio.run(probe(port));saved[model]=a
            q=np.stack([a[0,:,:7],a[0,:,8:15]],axis=1);current=np.stack([state[:7],state[8:15]])
            report['model_probes'][model]=dict(repetitions=10,identity=m.catalog('plate',model),
                max_repeat_abs_difference=float(np.abs(a-a[0]).max()),
                first4_max_target_difference_deg=float(np.rad2deg(np.abs(q[:4]-current)).max()),
                last_frame_max_target_difference_deg=float(np.rad2deg(np.abs(q[-1]-current)).max()),
                first4_gripper_closedness_range=np.stack([a[0,:4][:,[7,15]].min(0),a[0,:4][:,[7,15]].max(0)]).tolist())
            m.write(directory/'analysis.json',report);print('MODEL_PROBE',model,json.dumps({k:v for k,v in report['model_probes'][model].items() if k!='identity'}),flush=True)
    finally:server.stop()
    np.savez_compressed(directory/'fixed_observation_predictions.npz',**saved)
    print('READONLY_AUDIT_COMPLETE',directory,flush=True)

if __name__=='__main__':main()
