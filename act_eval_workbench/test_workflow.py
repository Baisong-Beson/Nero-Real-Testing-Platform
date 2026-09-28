"""Production worker process/approval/recording workflow on isolated fake ROS."""
import argparse
import asyncio
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
import numpy as np
from .common import *
from .test_integration import isolated

async def server(port):
    isolated()
    import websockets.asyncio.server as ws
    from nero_pi05_bridge import msgpack_numpy as msgpack
    task=os.environ.get('FAKE_TASK','plate');model=os.environ.get('FAKE_MODEL','mpi-base')
    identity=catalog(task,model);pose=np.array(ready_pose(task)['joints_rad'])
    meta=dict(schema='real_robot_round1.inference_policy.v1',task=task,model=model,commandable=False,
        policy_sha256=identity['policy_sha256'],encoder_state_sha256=identity['encoder_state_sha256'],action_shape=[16,16],latent='zero',preprocessing='real_robot_full_fov_v1')
    async def handle(client):
        await client.send(msgpack.packb(meta))
        async for payload in client:
            observation=msgpack.unpackb(payload);assert observation['observation/state'].shape==(16,)
            q=pose.copy();q[:,0]+=.015
            actions=np.tile(np.r_[q[0],1. if task=='banana' else 0.,q[1],0.],(16,1))
            if os.environ.get('FAKE_POLICY_FUTURE_OOB')=='1':actions[4:,9]=UPPER[1]+.01
            await client.send(msgpack.packb(dict(actions=actions)))
    async with ws.serve(handle,'127.0.0.1',port,compression=None,max_size=None):await asyncio.Future()

def main():
    isolated();directory=new_session('worker_workflow_simulation');processes=[];keep=threading.Event();keep.set()
    heart=directory/'heartbeat';heart.touch()
    def beat():
        while keep.is_set():heart.touch();time.sleep(.1)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    results=[]
    def worker(mode,d,kind=None,expected=0):
        args=[sys.executable,'-m','act_eval_workbench.worker',mode,'--directory',str(d),'--heartbeat',str(heart)]
        if kind:args+=['--kind',kind]
        r=subprocess.run(args,capture_output=True,text=True,timeout=70)
        (d/f'{mode}_{time.time_ns()}.log').write_text(r.stdout+r.stderr)
        assert r.returncode==expected,(mode,kind,r.stdout,r.stderr,read(d/'job_error.json') if (d/'job_error.json').exists() else read(d/'result.json') if (d/'result.json').exists() else None)
    def session(kind):
        d=directory/kind;d.mkdir();write(d/'config.json',dict(settings=Settings(duration_s=3).json(),port=port));return d
    try:
        for args,name in [([sys.executable,'-m','act_eval_workbench.fake_reset_driver'],'driver'),
                          ([sys.executable,str(LEGACY/'test_plate_formal_trial.py'),'--camera'],'camera'),
                          ([sys.executable,'-m','act_eval_workbench.test_workflow','--server',str(port)],'server')]:
            with (directory/f'{name}.log').open('w') as f:processes.append(subprocess.Popen(args,env=dict(os.environ,FAKE_START_WIDTH='.1'),stdout=f,stderr=subprocess.STDOUT))
        d=session('formal');worker('prepare',d,'formal');plan=read(d/'plan.json');assert plan['simulation_only']
        if os.environ.get('FAKE_POLICY_FUTURE_OOB')=='1':
            events=[json.loads(line) for line in (d/'preparation/events.jsonl').read_text().splitlines()]
            assert any(e['event']=='policy_future_limit_warning' for e in events)
            assert plan['protocol']['raw_joint_limit_scope']=='executed_prefix_only_unused_tail_warning'
        assert not (d/'execution').exists();approve(plan,d,'SIMULATION TEST')
        worker('execute',d);result=read(d/'result.json');assert result['started'] and result['runtime_completed'] and result['simulation_only']
        assert result['commands_sent']>20;assert not result['video']['error'];results.append(dict(case='formal_full_worker',passed=True,commands=result['commands_sent']))
        worker('execute',d,expected=2);results.append(dict(case='single_use_plan_replay_rejected',passed=True))
        d=session('default');worker('prepare',d,'default');plan=read(d/'plan.json');approve(plan,d,'SIMULATION TEST')
        worker('execute',d);result=read(d/'result.json');assert result['runtime_completed'] and result['feedback_driven'] and result['commands_sent']>0
        events=[json.loads(line) for line in (d/'execution/events.jsonl').read_text().splitlines()]
        assert next(e for e in events if e['event']=='execution_ready')['fixed_countdown_s']==0.
        results.append(dict(case='default_full_worker',passed=True,commands=result['commands_sent']))
        d=session('tampered');worker('prepare',d,'disable');plan=read(d/'plan.json');approve(plan,d,'SIMULATION TEST')
        approval=read(d/'approval.json');approval['plan_sha256']='wrong';write(d/'approval.json',approval)
        worker('execute',d,expected=2);assert not (d/'execution').exists();results.append(dict(case='tampered_approval_no_execution',passed=True))
        d=session('disable');worker('prepare',d,'disable');approve(read(d/'plan.json'),d,'SIMULATION TEST');worker('execute',d)
        result=read(d/'result.json');assert not result['errors'] and result['motor_disable_requested'];results.append(dict(case='disable_full_worker',passed=True))
        d=session('stop');worker('stop',d);assert not read(d/'stop_result.json')['errors'];results.append(dict(case='emergency_stop_full_worker',passed=True))
        write(directory/'summary.json',dict(passed=True,physical_motion_executed=False,results=results))
        print('WORKFLOW_PASSED',directory,json.dumps(results),flush=True)
    finally:
        keep.clear();thread.join(timeout=2)
        for p in processes:
            if p.poll() is None:p.terminate()
            try:p.wait(timeout=3)
            except subprocess.TimeoutExpired:p.kill();p.wait()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--server',type=int);args=p.parse_args()
    if args.server:asyncio.run(server(args.server))
    else:main()
