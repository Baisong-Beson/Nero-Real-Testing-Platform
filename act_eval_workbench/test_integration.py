"""Isolated ROS integration: no SDK/CAN. Must run on domain 174 / localhost."""
import asyncio
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time
import numpy as np
from .common import *
from . import controller as c
from . import worker as w

def isolated():
    if os.environ.get('ROS_DOMAIN_ID')!='174' or os.environ.get('ROS_LOCALHOST_ONLY')!='1':
        raise RuntimeError('Requires isolated ROS_DOMAIN_ID=174 and ROS_LOCALHOST_ONLY=1')

class Policy:
    def __init__(self,io,kind):self.io=io;self.kind=kind;self.metadata=dict(schema='real_robot_round1.inference_policy.v1',task=c.TASK,model=c.MODEL,commandable=False,
        policy_sha256=c.POLICY_SHA,encoder_state_sha256=c.ENCODER_SHA,action_shape=[16,16],latent='zero',preprocessing='real_robot_full_fov_v1')
    async def connect(self):pass
    async def close(self):pass
    async def infer(self,obs):
        if self.kind=='inference_failure' and self.io.command_count>=24:await asyncio.sleep(.3)
        await asyncio.sleep(.012)
        q=c.pos.READY.copy();q[:,0]+=.045
        actions=np.tile(np.r_[q[0],.3,q[1],.3],(16,1))
        if self.kind=='future_soft_limit':actions[4:,9]=UPPER[1]+.05
        if self.kind=='active_soft_limit' and self.io.command_count>=24:actions[1,9]=UPPER[1]+.01
        return dict(actions=actions)

def formal_case(directory,kind):
    isolated();logs=[];processes=[];io=rec=None
    try:
        for command,name in [('--drivers','drivers'),('--camera','camera')]:
            log=(directory/f'{name}.log').open('w');logs.append(log)
            env=dict(os.environ,PLATE_FAKE_STUCK='1' if kind=='stuck_driver' else '0')
            processes.append(subprocess.Popen([sys.executable,str(LEGACY/'test_plate_formal_trial.py'),command],env=env,stdout=log,stderr=subprocess.STDOUT))
        settings=Settings(duration_s=4);c.configure(settings,catalog('plate','mpi-base'),8026)
        heart=directory/'heartbeat';heart.touch();op=w.Operator(directory,heart)
        io=c.FormalIO(directory,op);rec=c.VideoRecorder(directory,(240,320,3));io.recorder=rec
        heart_running=threading.Event();heart_running.set()
        def heartbeat():
            while heart_running.is_set():heart.touch();time.sleep(.1)
        thread=threading.Thread(target=heartbeat,daemon=True);thread.start()
        q,widths=io.preflight();io.camera();io.monitor_camera=True
        if kind in ('scheduler_delay','scheduler_interrupt'):
            original_publish=io.publish_full
            def delayed_publish(*args,**kwargs):
                original_publish(*args,**kwargs)
                if io.command_count==25:
                    # Block the controller process once, as with the observed
                    # scheduling stall; DDS sensors keep publishing independently.
                    time.sleep(.27 if kind=='scheduler_delay' else .38)
                    io.pump()
            io.publish_full=delayed_publish
        stopped=False
        def permit():
            nonlocal stopped
            if io.command_count>=24 and not stopped:
                stopped=True
                if kind=='operator_stop':(directory/'STOP').touch()
                if kind=='heartbeat_loss':heart_running.clear()
                if kind=='camera_failure':processes[1].terminate();processes[1].wait(timeout=3)
            return op.permit()
        policy=Policy(io,kind);error=None
        try:
            answer=asyncio.run(c.run_attempt(io,dict(start_joints_rad=q.tolist(),start_widths_m=widths,trial_id='ISOLATED_SIMULATION'),permit,policy))
            assert kind in ('complete','scheduler_delay','future_soft_limit') and answer['runtime_completed']
            if kind=='scheduler_delay':assert answer['scheduler_delay_rebases']>=1
        except (RuntimeError,asyncio.TimeoutError,c.PolicyJointLimitError) as exc:
            error=str(exc) or type(exc).__name__
            expected={'operator_stop':['operator stop'],'heartbeat_loss':['operator stop'],'inference_failure':['TimeoutError'],'camera_failure':['camera'],'stuck_driver':['tracking stalled'],
                'scheduler_interrupt':['control interval','policy action observation expired'],
                'active_soft_limit':['joint soft limit exceeded']}
            assert kind not in ('complete','scheduler_delay') and any(term in error for term in expected[kind]),(kind,error)
            if kind=='scheduler_interrupt':assert io.command_count==25
        events=[json.loads(x) for x in (directory/'events.jsonl').read_text().splitlines()]
        cleanup=[e for e in events if e['event']=='cleanup'][-1];assert cleanup['errors']==[]
        if kind=='future_soft_limit':
            warnings=[e for e in events if e['event']=='policy_future_limit_warning'];assert warnings
            assert all(v['step']>4 for e in warnings for v in e['violations'])
            for e in events:
                if e['event']=='command_dispatch':limits(e['joints_rad'])
        if kind=='active_soft_limit':
            rejected=[e for e in events if e['event']=='policy_joint_limit_rejected'];assert rejected
            assert rejected[-1]['violations'][0]['step']==2
            evidence=read(directory/'rejected_policy_input.json')
            assert evidence['image_sha256']==sha(directory/'rejected_policy_input.jpg')
            assert evidence['actions']==rejected[-1]['actions']
        calls=[(e['service'],e['value']) for e in events if e['event']=='service_response']
        assert calls[-2:]==[('control_enable',False),('emergency_stop',None)]
        io.recorder=None;video=rec.close();rec=None;assert not video['error']
        assert io.command_count>0
        return dict(case=kind,passed=True,commands=io.command_count,error=error,physical_motion_executed=False,video=video,
            scheduler_delay_rebases=cleanup.get('scheduler_delay_rebases',0),max_control_interval_s=cleanup.get('max_control_interval_s'))
    finally:
        if 'heart_running' in locals():heart_running.clear();thread.join(timeout=1)
        if io:io.close()
        if rec:rec.close()
        for proc in processes:
            if proc.poll() is None:proc.terminate()
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        for log in logs:log.close()

def position_case(directory,task,stage):
    isolated();processes=[];io=rec=None;keep=threading.Event();keep.set();heart=directory/'heartbeat';heart.touch()
    def beat():
        while keep.is_set():heart.touch();time.sleep(.1)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    try:
        for flag,name in [('--drivers','drivers'),('--camera','camera')]:
            with (directory/f'{name}.log').open('w') as log:
                args=[sys.executable,'-m','act_eval_workbench.fake_reset_driver'] if flag=='--drivers' else [sys.executable,str(LEGACY/'test_plate_formal_trial.py'),flag]
                processes.append(subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT))
        op=w.Operator(directory,heart);io=w.ResetIO(directory,op);q,width=io.preflight();io.camera()
        goal=np.array(ready_pose(task)['joints_rad']) if stage=='ready' else np.deg2rad(DEFAULT_DEG)
        target=ready_gripper_target(task) if stage=='ready' else [.1,.1]
        plan=w.reset_plan(q,goal,width,via_default=stage=='ready',final_widths=target,contact_grasp=stage=='ready' and task=='banana');w.reset_geometry(plan);io.reset_path=plan
        rec=c.VideoRecorder(directory,(240,320,3));io.recorder=rec;io.monitor_camera=True
        result=w.reset.run_motion(io,plan,op.permit,lambda *_:None);assert result['ok']
        np.testing.assert_allclose(result['final_widths_m'],target,atol=.002)
        io.recorder=None;video=rec.close();rec=None;assert not video['error']
        return dict(case=f'{task}_{stage}',passed=True,physical_motion_executed=False,commands=io.command_count,error_rad=result['max_error_rad'])
    finally:
        keep.clear();thread.join(timeout=1)
        if io:io.close()
        if rec:rec.close()
        for proc in processes:
            if proc.poll() is None:proc.terminate()
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()

def main():
    isolated();directory=new_session('isolated_ros_validation');results=[]
    print('ISOLATED_DIRECTORY',directory,flush=True)
    for kind in ('complete','future_soft_limit','active_soft_limit','scheduler_delay','scheduler_interrupt','operator_stop','heartbeat_loss','inference_failure','camera_failure','stuck_driver'):
        path=directory/kind;path.mkdir()
        results.append(formal_case(path,kind));write(directory/'summary.json',dict(results=results,physical_motion_executed=False));print(json.dumps(results[-1]),flush=True)
    for task,stage in [('plate','default'),('plate','ready'),('banana','ready'),('holder','ready')]:
        path=directory/(task+'_'+stage);path.mkdir()
        results.append(position_case(path,task,stage));write(directory/'summary.json',dict(results=results,physical_motion_executed=False));print(json.dumps(results[-1]),flush=True)
    print('ISOLATED_ALL_PASSED',directory,flush=True)
if __name__=='__main__':main()
