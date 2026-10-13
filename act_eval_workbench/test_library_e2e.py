"""Isolated real-process gateway and trained policy import validation, no ROS."""
import argparse
import asyncio
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import time
import numpy as np
from . import common as c,library as lib

class Fixture:
    def __init__(self,config):self.config=config
    def infer(self,obs):
        assert obs['prompt']=='Move the cup to the tray.'
        assert obs['observation/left_wrist_image'][0,0,0]==11
        assert obs['observation/right_wrist_image'][0,0,0]==22
        c.write(self.config['capture'],dict(prompt=obs['prompt'],task=obs['task_id'],keys=list(obs),state=obs['observation/state'].tolist()))
        return {'actions':np.tile(obs['observation/state'],(3,1)).astype(np.float32)}

def fixture(config):return Fixture(config)

def main(real_policy):
    if os.environ.get('ACT_EVAL_FAKE_DRIVER')!='1' or os.environ.get('ROS_DOMAIN_ID')!='174':raise RuntimeError('Use isolated staging environment')
    from .backend import ModelServer,offline_replay
    directory=c.new_session('model_import_validation');cancel=threading.Event();server=ModelServer(directory,print);process=None;out={}
    suffix=str(time.time_ns());task_id='validation_task_'+suffix;model_id='validation_vla_'+suffix
    source=directory/'packages';source.mkdir()
    task=dict(schema='nero.task.v1',id='validation_task',label='导入验证任务',instruction='Move the cup to the tray.',success_definition='测试接口，不评判真实任务',ready=dict(method='explicit',joints_deg=c.DEFAULT_DEG.tolist(),gripper_width_m=[.1,.1]))
    task.update(id=task_id,label=task['label']+suffix)
    c.write(source/'task.json',task);lib.import_package(source/'task.json','task')
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    state=np.r_[np.deg2rad(c.DEFAULT_DEG[0]),0,np.deg2rad(c.DEFAULT_DEG[1]),0].astype(np.float32)
    np.savez(source/'probe.npz',state=state[None],rgb=np.zeros((1,32,32,3),np.uint8),rgb_left_wrist=np.full((1,32,32,3),11,np.uint8),rgb_right_wrist=np.full((1,32,32,3),22,np.uint8))
    model=dict(schema='nero.model.v1',id='validation_vla',label='VLA 适配器验证',adapter='websocket',tasks=['validation_task'],endpoint=f'ws://127.0.0.1:{port}',service_model_id='test_only',checkpoint_sha256='a'*64,action_contract=lib.CONTRACT,probe_file='probe.npz',
        runtime=dict(action_horizon=3,steps_per_replan=2,dt_s=.1,inference_timeout_s=2.,cameras=['main','left_wrist','right_wrist']))
    model.update(id=model_id,label=model['label']+suffix,tasks=[task_id])
    c.write(source/'model.json',model);lib.import_package(source/'model.json','model');c.write(source/'service_config.json',{'capture':str(directory/'service_capture.json')})
    command=[sys.executable,'-m','nero_eval_workbench.adapter_server','--factory','nero_eval_workbench.test_library_e2e:fixture','--model',str(source/'model.json'),'--config',str(source/'service_config.json'),'--port',str(port)]
    try:
        with (directory/'fixture.log').open('w') as log:process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            if process.poll() is not None:raise RuntimeError((directory/'fixture.log').read_text())
            try:
                with socket.create_connection(('127.0.0.1',port),timeout=.2):break
            except OSError:time.sleep(.2)
        s=c.Settings(task=task_id,model=model_id)
        p=server.ensure(s,cancel);d=directory/'vla_offline';d.mkdir();result=asyncio.run(offline_replay(d,s,p,cancel,print))
        assert result['golden_passed'] is None and len(result['trajectory'])==3 and result['physical_motion_executed'] is False
        assert c.read(directory/'service_capture.json')['prompt']==task['instruction']
        out['vla_gateway']=dict(passed=True,horizon=3,cameras=3,language_verified=True,golden_passed=None,physical_motion_executed=False)
        server.stop()
        bad=dict(model,id='validation_wrong_'+suffix,label='错误身份验证'+suffix,checkpoint_sha256='b'*64)
        c.write(source/'wrong.json',bad);lib.import_package(source/'wrong.json','model')
        try:server.ensure(c.Settings(task=s.task,model=bad['id']),cancel)
        except Exception as exc:out['wrong_identity_rejected']=str(exc)
        else:raise AssertionError('wrong model accepted')
        if real_policy:
            trained_id='validation_trained_'+suffix
            lib.import_sealed(real_policy,trained_id,'训练权重导入验证'+suffix,[task_id])
            s=c.Settings(task=task_id,model=trained_id);p=server.ensure(s,cancel)
            d=directory/'trained_offline';d.mkdir();result=asyncio.run(offline_replay(d,s,p,cancel,print))
            assert result['golden_passed'] is True and len(result['probes'])==3
            out['trained_policy']=dict(passed=True,probes=3,golden_passed=True,physical_motion_executed=False)
    finally:
        server.stop()
        if process:
            process.terminate()
            try:process.wait(10)
            except subprocess.TimeoutExpired:process.kill();process.wait()
    c.write(directory/'validation.json',out);print('VALIDATION',directory,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--real-policy',type=Path);args=p.parse_args();main(args.real_policy)
