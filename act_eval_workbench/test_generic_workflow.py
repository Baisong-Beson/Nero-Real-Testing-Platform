"""Run variable-horizon slow VLA through production worker on isolated fake ROS."""
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
from . import common as c,library as lib

class Fixture:
    def __init__(self,config):self.config=config
    def infer(self,obs):
        assert obs['prompt']==self.config['prompt']
        time.sleep(.25)
        return {'actions':np.tile(np.array(self.config['target'],np.float32),(3,1))}
def fixture(config):return Fixture(config)

def main():
    from .test_integration import isolated
    from .backend import ModelServer
    isolated()
    assert os.environ.get('ACT_EVAL_FAKE_DRIVER')=='1'
    directory=c.new_session('generic_worker_validation');suffix=str(time.time_ns());processes=[];keep=threading.Event();keep.set()
    heartbeat=directory/'heartbeat'
    def beat():
        while keep.is_set():heartbeat.touch();time.sleep(.05)
    thread=threading.Thread(target=beat,daemon=True);thread.start();cancel=threading.Event();server=ModelServer(directory,print)
    task='generic_task_'+suffix;model='generic_model_'+suffix
    pose=c.ready_pose('plate');q=np.asarray(pose['joints_rad']);target=np.r_[q[0],0.,q[1],0.];target[0]+=.003;target[8]+=.003
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    td=dict(schema='nero.task.v1',id=task,label='通用流程任务'+suffix,instruction='Move both arms a little.',success_definition='仅测试模拟驱动',ready=dict(method='explicit',joints_deg=pose['joints_deg'],gripper_width_m=[.04,.06]))
    md=dict(schema='nero.model.v1',id=model,label='慢速 VLA 流程'+suffix,adapter='websocket',tasks=[task],endpoint=f'ws://127.0.0.1:{port}',service_model_id='isolated_fake',checkpoint_sha256='c'*64,action_contract=lib.CONTRACT,
        runtime=dict(action_horizon=3,steps_per_replan=2,dt_s=.05,inference_timeout_s=1.,cameras=['main']))
    c.write(directory/'task.json',td);lib.import_package(directory/'task.json','task');c.write(directory/'model.json',md);lib.import_package(directory/'model.json','model')
    c.write(directory/'fixture.json',dict(prompt=td['instruction'],target=target.tolist()))
    try:
        commands=[([sys.executable,'-m','act_eval_workbench.fake_reset_driver'],'driver'),
            ([sys.executable,str(c.LEGACY/'test_plate_formal_trial.py'),'--camera'],'camera'),
            ([sys.executable,'-m','nero_eval_workbench.adapter_server','--factory','nero_eval_workbench.test_generic_workflow:fixture','--model',str(directory/'model.json'),'--config',str(directory/'fixture.json'),'--port',str(port)],'policy')]
        for command,name in commands:
            with (directory/(name+'.log')).open('w') as f:processes.append(subprocess.Popen(command,env=dict(os.environ,FAKE_START_WIDTH='.1'),stdout=f,stderr=subprocess.STDOUT))
        time.sleep(1)
        settings=c.Settings(task=task,model=model,duration_s=3.5);gateway=server.ensure(settings,cancel)
        trial=directory/'formal';trial.mkdir();c.write(trial/'config.json',dict(settings=settings.json(),port=gateway))
        def worker(mode,kind=None):
            cmd=[sys.executable,'-m','nero_eval_workbench.worker',mode,'--directory',str(trial),'--heartbeat',str(heartbeat)]
            if kind:cmd+=['--kind',kind]
            r=subprocess.run(cmd,capture_output=True,text=True,timeout=45);(directory/(trial.name+'_'+mode+'.log')).write_text(r.stdout+r.stderr)
            assert r.returncode==0,(r.stdout,r.stderr,c.read(trial/'job_error.json') if (trial/'job_error.json').exists() else c.read(trial/'result.json'))
        worker('prepare','formal');plan=c.read(trial/'plan.json');assert plan['model_identity']['runtime']['action_horizon']==3
        c.approve(plan,trial,'ISOLATED TEST');worker('execute');result=c.read(trial/'result.json')
        assert result['runtime_completed'] and result['commands_sent']>22,result
        import json
        events=[json.loads(line) for line in (trial/'execution/events.jsonl').read_text().splitlines()]
        chunks=[e for e in events if e['event']=='policy_chunk'];assert chunks and len(chunks[0]['actions'])==3
        assert chunks[0]['capture']['inference_ms']>=250
        assert result['scheduler_delay_rebases']==0,result
        trial=directory/'ready';trial.mkdir();c.write(trial/'config.json',dict(settings=settings.json(),port=gateway))
        worker('prepare','ready');plan=c.read(trial/'plan.json');assert plan['trajectory']['gripper_goal_m']==[.04,.06]
        c.approve(plan,trial,'ISOLATED RESET TEST');worker('execute');reset_result=c.read(trial/'result.json')
        assert reset_result['runtime_completed'],reset_result
        c.write(directory/'validation.json',dict(passed=True,physical_motion_executed=False,commands=result['commands_sent'],chunks=len(chunks),horizon=3,inference_delay_s=.25,task_instruction_recorded=True,custom_ready_grippers_verified=True))
        print('GENERIC_WORKFLOW_PASSED',directory)
    finally:
        keep.clear();thread.join(2);server.stop()
        for p in processes:
            p.terminate()
            try:p.wait(5)
            except subprocess.TimeoutExpired:p.kill();p.wait()

if __name__=='__main__':main()
