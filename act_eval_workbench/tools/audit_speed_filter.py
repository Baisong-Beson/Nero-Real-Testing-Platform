"""Read-only recorded-speed audit and deterministic old/new filter comparison.

No ROS nodes, publishers, motor services or inference are created.
"""
import json
import time
from pathlib import Path
import numpy as np
from act_eval_workbench import controller as c
from act_eval_workbench.common import DEFAULT_DEG, RUNS, new_session, write


class PreviousSlew(c.Slew):
    def next(self, target_q, target_w, elapsed, actual_q=None):
        if actual_q is not None:
            target_q = np.clip(target_q, actual_q-.02, actual_q+.02)
        target = np.r_[target_q.reshape(-1), target_w]
        err = target-self.x
        desired = np.sign(err)*np.minimum(np.minimum(3*np.abs(err), np.sqrt(self.amax*np.abs(err))), self.vmax)
        self.v = np.clip(desired, np.maximum(-self.vmax, self.v-self.amax*elapsed),
                         np.minimum(self.vmax, self.v+self.amax*elapsed))
        self.x = self.x+self.v*elapsed
        return self.x[:14].reshape(2,7).copy(), self.x[14:].copy()


def compare(cls, feedback_gain):
    actual = np.deg2rad(DEFAULT_DEG); target = actual.copy(); target[:,0] += .3
    slew = cls(actual, [.1,.1]); peak = lead = 0.; arrived = gripper_arrived = None
    for i in range(600):
        q,w = slew.next(target, [0.,.1], c.DT, actual)
        lead = max(lead,float(np.abs(q-actual).max()))
        peak = max(peak,float(np.abs(slew.v[:14]).max()))
        actual += feedback_gain*(q-actual)
        if arrived is None and np.max(np.abs(target-actual)) < .001: arrived = (i+1)*c.DT
        if gripper_arrived is None and w[0]<.0001: gripper_arrived = (i+1)*c.DT
    return dict(joint_arrival_s=arrived,gripper_arrival_s=gripper_arrived,
                peak_joint_command_rad_s=peak,max_command_lead_rad=lead)


def recorded(path):
    events = [json.loads(line) for line in (path/'execution/events.jsonl').read_text().splitlines()]
    commands = [r for r in events if r['event']=='command_dispatch']
    velocity=[]; model_large=0; eligible=0; chunks={}
    for row in events:
        if row['event']=='policy_chunk': chunks[row['chunk']]=np.asarray(row['actions'])
    for previous,row in zip(commands,commands[1:]):
        elapsed=row['wall_time_s']-previous['wall_time_s']
        if row['chunk']<0 or not 0<elapsed<.35: continue
        q=np.asarray(row['joints_rad']); old=np.asarray(previous['joints_rad'])
        velocity.append(np.abs(q-old)/elapsed)
        raw=chunks[row['chunk']][row['step']]
        target=np.stack([raw[:7],raw[8:15]])
        model_large+=int(np.sum(np.abs(target-np.asarray(row['feedback_rad']))>.02))
        eligible+=14
    values=np.asarray(velocity)
    result=json.loads((path/'result.json').read_text())
    return dict(session=path.name,commands=len(commands),
                command_speed_rad_s_percentiles=np.percentile(values,[50,95,99,100]).tolist(),
                fraction_joint_targets_outside_soft_lead=model_large/eligible,
                termination=result.get('error'),recorded_protocol=result.get('protocol_sha256'))


def main():
    directory=new_session('speed_filter_readonly_audit')
    comparisons={str(gain):dict(old=compare(PreviousSlew,gain),new=compare(c.Slew,gain)) for gain in [1.,.5,.35]}
    actual=np.deg2rad(DEFAULT_DEG); target=actual.copy();target[:,0]+=.3;slew=c.Slew(actual,[.1,.1])
    began=time.perf_counter()
    for _ in range(1000):actual,_=slew.next(target,[0.,.1],c.DT,actual)
    report=dict(physical_motion_executed=False,kind='recorded audit and synthetic tracking comparison',
        protocol=c.PROTOCOL,simulation_description='0.3 rad joint move; 0.1 m gripper close; feedback += gain*(command-feedback); dt=0.05 s',
        simulation_warning='Synthetic plant, not a prediction of real task duration or success.',
        comparisons=comparisons,mean_filter_compute_ms=(time.perf_counter()-began),
        recorded=[recorded(p) for p in sorted(RUNS.glob('*_formal'))[-3:]])
    write(directory/'audit.json',report)
    print(directory);print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
