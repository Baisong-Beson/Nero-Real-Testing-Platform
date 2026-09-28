"""Main-view ACT shadow capture. No publishers or actuation service clients."""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import threading
import time

import cv2
import numpy as np

from nero_pi05_bridge.observation import decode_camera_message, parse_joint_state, topic_is_compressed
from nero_pi05_bridge.policy_client import PolicyClient


def act_arm_state(names, positions):
    joints, closedness = parse_joint_state(names, positions, gripper_min_width_m=0., gripper_max_width_m=.1)
    return np.concatenate((joints, closedness))


def validate_metadata(metadata, task, model, policy_sha):
    expected = dict(schema='real_robot_round1.inference_policy.v1', task=task, model=model,
                    commandable=False, policy_sha256=policy_sha, action_shape=[16,16], latent='zero',
                    preprocessing='real_robot_full_fov_v1')
    for key, value in expected.items():
        if metadata.get(key) != value or (key == 'commandable' and metadata.get(key) is not False):
            raise ValueError(f'ACT server metadata mismatch: {key}')


def observation_quality(entries, now, max_age, max_skew):
    missing = [key for key in ('image','right','left') if key not in entries]
    if missing:
        return False, dict(reason='missing',streams=missing)
    ages = {k: now-v['received'] for k,v in entries.items()}
    source_ages = {k: v['source_age_at_receipt']+ages[k] for k,v in entries.items()}
    stamps = [v['stamp_ns']/1e9 for v in entries.values()]
    skew = max(stamps)-min(stamps)
    meta = dict(ages_sec=ages,source_ages_sec=source_ages,stamp_skew_sec=skew,
                stamps_ns={k:v['stamp_ns'] for k,v in entries.items()})
    if any(not np.isfinite(a) or a < -.05 or a > max_age for a in [*ages.values(),*source_ages.values()]):
        return False, dict(reason='stale_or_invalid_timestamp',**meta)
    if skew > max_skew:
        return False, dict(reason='skew',**meta)
    return True, meta


async def save_capture(image_path, image_rgb, log, **fields):
    """Await image and log writes without starving the ROS/asyncio receive loop."""
    def write():
        started = time.monotonic()
        if not cv2.imwrite(str(image_path), cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)):
            raise RuntimeError('image save failed')
        log('inference', ok=True, image_path=str(image_path.resolve()), **fields)
        return (time.monotonic()-started)*1000
    # Await each record so storage cannot build an unbounded image backlog.
    # Exceptions still fail the capture; stale inputs are never accepted.
    return await asyncio.to_thread(write)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task',choices=('banana','holder','plate'),required=True)
    parser.add_argument('--model',choices=('mpi-base','p0','g2'),required=True)
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8016)
    parser.add_argument('--duration',type=float,default=30.)
    parser.add_argument('--inference-rate',type=float,default=5.)
    parser.add_argument('--max-age',type=float,default=.5)
    parser.add_argument('--max-skew',type=float,default=.25)
    parser.add_argument('--startup-timeout',type=float,default=20.)
    parser.add_argument('--image-topic',default='/zed_m/zed_node/rgb/color/rect/image/compressed')
    parser.add_argument('--right-topic',default='/right_arm/feedback/joint_states')
    parser.add_argument('--left-topic',default='/left_arm/feedback/joint_states')
    parser.add_argument('--export',type=Path,default=Path(__file__).resolve().parents[5]/'models/act_export')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if any(not np.isfinite(v) or v<=0 for v in (args.duration,args.inference_rate,args.max_age,args.max_skew,args.startup_timeout)):
        parser.error('duration/rate/age/skew/timeout must be finite and positive')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    policy_sha=hashlib.sha256((args.export/args.task/args.model/'policy.pt').read_bytes()).hexdigest()
    manifest_path=args.export/'manifest.json'
    seal=json.loads((args.export/'EXPORT_COMPLETE').read_text())
    if hashlib.sha256(manifest_path.read_bytes()).hexdigest()!=seal['manifest_sha256']:
        raise ValueError('export seal mismatch')
    if json.loads(manifest_path.read_text())['files'][f'{args.task}/{args.model}/policy.pt']['sha256']!=policy_sha:
        raise ValueError('policy hash mismatch')
    image_dir=args.output.with_suffix('.images')
    stream=args.output.open('x')
    image_dir.mkdir(exist_ok=False)
    log_lock=threading.Lock()
    def log(event,**fields):
        record=json.dumps(dict(event=event,wall_time_s=time.time(),monotonic_s=time.monotonic(),commandable=False,**fields))+'\n'
        with log_lock:
            stream.write(record);stream.flush()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage,Image,JointState
    rclpy.init()
    class ShadowNode(Node):
        def __init__(self):
            super().__init__('nero_act_shadow');self.latest={};self.last_stamps={}
            for key,typ,topic in [('image',CompressedImage if topic_is_compressed(args.image_topic) else Image,args.image_topic),('right',JointState,args.right_topic),('left',JointState,args.left_topic)]:
                self.create_subscription(typ,topic,lambda msg,k=key:self.receive(k,msg),qos_profile_sensor_data)
        def receive(self,key,msg):
            try:
                stamp=msg.header.stamp.sec*10**9+msg.header.stamp.nanosec
                if stamp<=0 or stamp<self.last_stamps.get(key,0):
                    raise ValueError('missing or regressed message timestamp')
                received=time.monotonic();source_age=(self.get_clock().now().nanoseconds-stamp)/1e9
                value=decode_camera_message(msg) if key=='image' else act_arm_state(msg.name,msg.position)
                self.latest[key]=dict(value=value,received=received,stamp_ns=stamp,source_age_at_receipt=source_age)
                self.last_stamps[key]=stamp
            except Exception as exc:
                self.latest.pop(key,None);log('input_error',key=key,error=str(exc))
        def publisher_counts(self):
            return {a:self.count_publishers(f'/{a}_arm/control/joint_states') for a in ('right','left')}
        def snapshot(self):
            entries=dict(self.latest)
            ok,meta=observation_quality(entries,time.monotonic(),args.max_age,args.max_skew)
            if not ok:return None,meta
            state=np.concatenate((entries['right']['value'],entries['left']['value'])).astype(np.float32)
            meta.update(state=state.tolist(),joint_position=state[:7].tolist(),left_joint_position=state[8:15].tolist(),
                        gripper_closedness=[float(state[7])],left_gripper_closedness=[float(state[15])])
            return {'observation/exterior_image_1_left':entries['image']['value'],'observation/state':state},meta
    node=ShadowNode();client=PolicyClient(args.host,args.port,inference_timeout_sec=10.)
    summary=dict(commandable=False,task=args.task,model=args.model,policy_sha256=policy_sha,ok=False,successful_inferences=0)
    async def run():
        async def spin():
            while rclpy.ok():
                rclpy.spin_once(node,timeout_sec=0.);await asyncio.sleep(.001)
        spinning=asyncio.create_task(spin())
        try:
            await client.connect();validate_metadata(client.metadata,args.task,args.model,policy_sha)
            log('start',task=args.task,model=args.model,policy_metadata=client.metadata,image_topic=args.image_topic)
            deadline=time.monotonic()+args.startup_timeout
            while True:
                counts=node.publisher_counts()
                if any(counts.values()):raise RuntimeError(f'control publishers present: {counts}')
                obs,meta=node.snapshot()
                if obs is not None:break
                if time.monotonic()>=deadline:raise RuntimeError(f'input startup timeout: {meta}')
                await asyncio.sleep(.05)
            started=time.monotonic();latencies=[];jumps=[];recording_times=[];sample_gaps=[];valid_duration=0.;last_good=None
            while time.monotonic()-started<args.duration:
                tick=time.monotonic()
                if spinning.done():await spinning
                counts=node.publisher_counts()
                if any(counts.values()):raise RuntimeError(f'control publishers present: {counts}')
                obs,meta=node.snapshot()
                if obs is None:raise RuntimeError(f'observation invalid: {meta}')
                sent=time.monotonic();response=await client.infer(obs);returned=time.monotonic()
                actions=np.asarray(response['actions'],dtype=np.float32)
                if actions.shape!=(16,16) or not np.isfinite(actions).all():raise ValueError('nonfinite or invalid action shape')
                return_ages={k:v+returned-sent for k,v in meta['source_ages_sec'].items()}
                if max(return_ages.values())>args.max_age:raise RuntimeError(f'observation expired during inference: {return_ages}')
                if any(node.publisher_counts().values()):raise RuntimeError('control publisher appeared during inference')
                index=summary['successful_inferences'];image_path=image_dir/f'{index:06d}.png'
                delta=float(np.abs((actions[0]-obs['observation/state'])[[*range(7),*range(8,15)]]).max())
                latency=(returned-sent)*1000
                recording_ms=await save_capture(image_path,obs['observation/exterior_image_1_left'],log,
                    observation=meta,return_source_ages_sec=return_ages,inference_returned_monotonic_s=returned,
                    actions={'chunk':actions.tolist(),'first_action':actions[0].tolist()},
                    roundtrip_ms=latency,server_timing=response.get('server_timing',{}),max_abs_joint_delta=delta,control_publishers=counts)
                summary['successful_inferences']+=1;latencies.append(latency);jumps.append(delta);recording_times.append(recording_ms)
                if last_good is not None:
                    sample_gaps.append(returned-last_good);valid_duration+=returned-last_good
                last_good=returned
                await asyncio.sleep(max(0,1/args.inference_rate-(time.monotonic()-tick)))
            summary.update(ok=bool(latencies),capture_elapsed_s=time.monotonic()-started,
                           successful_sample_span_s=valid_duration,
                           roundtrip_ms=dict(zip(('p50','p95','max'),np.percentile(latencies,[50,95,100]).tolist())),
                           recording_ms=dict(zip(('p50','p95','max'),np.percentile(recording_times,[50,95,100]).tolist())),
                           max_successful_sample_gap_s=max(sample_gaps,default=0.),
                           max_abs_first_joint_delta_rad=max(jumps))
        finally:
            spinning.cancel()
            try:await spinning
            except asyncio.CancelledError:pass
            await client.close()
    try:
        asyncio.run(run())
    except BaseException as exc:
        summary['error']=str(exc) or type(exc).__name__;log('failure',error=summary['error']);raise
    finally:
        log('finish',summary=summary);stream.close()
        args.output.with_suffix('.summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        node.destroy_node()
        if rclpy.ok():rclpy.shutdown()
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
