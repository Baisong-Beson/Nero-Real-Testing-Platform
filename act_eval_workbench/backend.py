from __future__ import annotations
import asyncio
import contextlib
import os
from pathlib import Path
import queue
import socket
import subprocess
import sys
import threading
import time
import numpy as np
from .common import *
from nero_pi05_bridge.policy_client import PolicyClient
from .policy_protocol import validate_metadata,actions as validate_actions
from . import library

class ModelServer:
    """Owns only the Popen it launches, never takes over/kills an occupied port."""
    def __init__(self,directory,notify):
        self.directory=directory;self.notify=notify;self.process=None;self.selected=None;self.port=None
    def stop(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:self.process.wait(timeout=8)
            except subprocess.TimeoutExpired:self.process.kill();self.process.wait(timeout=3)
        self.process=None;self.selected=None
    def ensure(self,settings,cancel):
        selected=(settings.task,settings.model)
        if self.selected==selected and self.process and self.process.poll() is None:return self.port
        self.stop();identity=catalog(*selected,verify_policy=True)
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));self.port=sock.getsockname()[1]
        log=self.directory/f'server_{settings.task}_{settings.model}_{time.time_ns()}.log'
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(PACKAGE.parent),str(ROOT/'examples/nero_pi05_bridge'),os.environ.get('PYTHONPATH','')]),PYTHONNOUSERSITE='1')
        command=[os.environ.get('NERO_SERVER_PYTHON', sys.executable),'-m','nero_pi05_bridge.act_policy_server',
                 '--task',settings.task,'--model',settings.model,'--export',str(EXPORT),'--device','cuda:0','--host','127.0.0.1','--port',str(self.port)]
        if identity.get('adapter'):
            command=[command[0],'-m','nero_eval_workbench.model_gateway','--task',settings.task,'--model',settings.model,'--port',str(self.port)]
        load_timeout=180
        if identity.get('adapter')=='openpi_nero':
            command[0]=str(PLATFORM/'runtime/python/openpi/bin/python')
            if not Path(command[0]).is_file():raise RuntimeError('缺少本地模型 Python 环境 runtime/python/openpi')
            env.update(PYTHONPATH=os.pathsep.join([str(PACKAGE.parent),str(ROOT/'src'),str(ROOT/'packages/openpi-client/src'),str(ROOT/'examples/nero_pi05_bridge')]),
                XLA_PYTHON_CLIENT_MEM_FRACTION='.75',XLA_PYTHON_CLIENT_PREALLOCATE='false',JAX_PLATFORMS='cuda',PYTHONUNBUFFERED='1')
            load_timeout=600
        with log.open('w') as stream:self.process=subprocess.Popen(command,cwd=PLATFORM,env=env,stdout=stream,stderr=subprocess.STDOUT)
        self.notify(f'加载 {settings.task}/{settings.model}，独立端口 {self.port}；日志 {log.name}')
        deadline=time.monotonic()+load_timeout
        async def check():
            client=PolicyClient('127.0.0.1',self.port,connect_timeout_sec=.5,inference_timeout_sec=max(10,identity.get('runtime',{}).get('inference_timeout_s',10)))
            try:
                await client.connect()
                validate_metadata(client.metadata,*selected,identity['policy_sha256'])
                if client.metadata.get('encoder_state_sha256')!=identity['encoder_state_sha256']:raise ValueError('encoder身份不符')
                probe_path,_=library.probe_paths(*selected)
                if probe_path:
                    cfg=identity.get('runtime',dict(action_horizon=16,cameras=['main']))
                    library.validate_probe(probe_path,cfg['cameras'])
                    with np.load(probe_path,allow_pickle=False) as probe:
                        extra={library.CAMERAS[k]:probe['rgb_'+k][0] for k in cfg['cameras'] if k!='main'}
                        for _ in range(2):
                            reply=await client.infer(library.observation(identity,probe['rgb'][0],probe['state'][0],extra))
                            validate_actions(reply['actions'],cfg['action_horizon'])
            finally:await client.close()
        while time.monotonic()<deadline:
            if cancel.is_set():self.stop();raise RuntimeError('模型加载已取消')
            if self.process.poll() is not None:raise RuntimeError('模型服务启动失败：\n'+log.read_text(errors='replace')[-3500:])
            try:asyncio.run(check());self.selected=selected;return self.port
            except (OSError,asyncio.TimeoutError):time.sleep(.3)
            except Exception:self.stop();raise
        self.stop();raise TimeoutError(f'模型服务加载超过 {load_timeout} 秒，查看日志')

async def offline_replay(directory,settings,port,cancel,notify):
    """Actual model on sealed RGB/state probes + kinematic response, no ROS init."""
    from . import controller as c
    identity=catalog(settings.task,settings.model,verify_policy=True);c.configure(settings,identity,port)
    probe_path,golden_path=library.probe_paths(settings.task,settings.model)
    if probe_path is None:raise ValueError('此模型/任务未导入离线样本 probe.npz。请导入样本后离线测试，或使用实时相机只读推理。')
    cfg=identity.get('runtime',dict(action_horizon=16,cameras=['main'],inference_timeout_s=10))
    library.validate_probe(probe_path,cfg['cameras'])
    with np.load(probe_path,allow_pickle=False) as z:probe={k:z[k] for k in z.files}
    with np.load(golden_path,allow_pickle=False) if golden_path else contextlib.nullcontext() as z:golden=z['expected_raw_action_chunks'] if z else None
    client=PolicyClient('127.0.0.1',port,inference_timeout_sec=max(10,cfg['inference_timeout_s']))
    rows=[];all_trajectory=[]
    try:
        await client.connect();validate_metadata(client.metadata,settings.task,settings.model,identity['policy_sha256'])
        if client.metadata.get('encoder_state_sha256')!=identity['encoder_state_sha256']:raise ValueError('encoder mismatch')
        for i,state in enumerate(probe['state']):
            if cancel.is_set():raise RuntimeError('离线模拟已停止')
            extra={library.CAMERAS[k]:probe['rgb_'+k][i] for k in cfg['cameras'] if k!='main'}
            start=time.monotonic();reply=await client.infer(library.observation(identity,probe['rgb'][i],state,extra))
            inference_ms=(time.monotonic()-start)*1000
            actual=validate_actions(reply['actions'],cfg['action_horizon']);expected=golden[i] if golden is not None else None
            ok=bool(np.allclose(actual,expected,rtol=2e-4,atol=2e-4)) if expected is not None else None
            q=np.stack([state[:7],state[8:15]]);w=.1*(1-np.clip(state[[7,15]],0,1));trajectory=[];violations=[]
            try:
                aq,aw=c.unpack(actual);slew=c.DriverTargets(q,w);geometry=c.Geometry(q,w)
                for step in range(len(actual)):
                    try:geometry.check(aq[step],aw[step])
                    except (ValueError,RuntimeError) as exc:violations.append(str(exc))
                    q,w=slew.next(aq[step],aw[step],.05,q)
                    trajectory.append(dict(step=step,joints_deg=np.rad2deg(q).tolist(),width_m=w.tolist()))
            except (ValueError,RuntimeError) as exc:violations.append(str(exc))
            row=dict(probe=i,golden_passed=ok,max_abs_error=float(np.abs(actual-expected).max()) if expected is not None else None,
                     inference_ms=inference_ms,violations=violations,trajectory=trajectory,
                     raw_actions=actual.tolist())
            rows.append(row);all_trajectory.extend(trajectory);notify(f'离线样本 {i+1}/{len(probe["state"])}：golden={ok}，轨迹告警 {len(violations)}')
            import cv2
            cv2.imwrite(str(directory/f'probe_{i}.png'),cv2.cvtColor(probe['rgb'][i],cv2.COLOR_RGB2BGR))
        result=dict(kind='offline_simulation',settings=settings.json(),model_identity=identity,
                    golden_passed=all(x['golden_passed'] for x in rows) if golden is not None else None,probes=rows,trajectory=all_trajectory,
                    physical_motion_executed=False,simulation_scope='RGB/state/language replay and ideal joint tracking. No contact physics or task-success adjudication; golden null means no reference supplied.')
        write(directory/'result.json',result);return result
    finally:await client.close()

class Backend:
    def __init__(self,start_telemetry=True):
        self.events=queue.Queue();self.directory=new_session('desktop');self.heartbeat=self.directory/'heartbeat';self.tick()
        self.cancel=threading.Event();self.busy=False;self.thread=None;self.process=None;self.active=None;self.closing=False
        self.server=ModelServer(self.directory,self.log);self.telemetry=None;self.stop_jobs=[]
        self.hardware_busy=False;self.hardware_thread=None;self.hardware_cancel=threading.Event()
        from .camera_reconnect import ReconnectWatch
        self.camera_watch=ReconnectWatch();self.camera_watch_enabled=start_telemetry
        if start_telemetry:
            with (self.directory/'telemetry.log').open('w') as f:self.telemetry=subprocess.Popen([sys.executable,'-m','nero_eval_workbench.telemetry','--directory',str(self.directory)],stdout=f,stderr=subprocess.STDOUT,cwd=PLATFORM)
            self.recover_connection()
    def tick(self):self.heartbeat.touch()
    def log(self,text):self.events.put(dict(type='log',text=str(text)))
    def telemetry_state(self):
        try:return read(self.directory/'telemetry.json')
        except (OSError,ValueError):return {}
    def recover_connection(self,camera_only=False):
        if self.busy or self.hardware_busy or self.stop_jobs or self.closing:
            raise RuntimeError('请在当前任务结束后重新检查连接')
        self.hardware_busy=True;self.hardware_cancel.clear()
        self.events.put(dict(type='hardware',message='正在检查主相机连接…' if camera_only else '正在检查 CAN、双臂和主相机连接…'))
        def work():
            try:
                from .hardware_startup import recover,isolated,camera_usb_status
                result=dict(ok=True,message='') if camera_only else recover(self.log,self.hardware_cancel)
                if result.get('ok') and not result.get('skipped'):
                    if not camera_only:
                        from .driver_reconnect import recover as recover_drivers
                        result['drivers']=recover_drivers(PLATFORM,self.telemetry_state,self.hardware_cancel,self.log)
                        result['message']+='；'+result['drivers']['message']
                if not isolated():
                    from .camera_reconnect import recover as recover_camera,usb_snapshot
                    result['main_camera']=recover_camera(PLATFORM,self.telemetry_state,self.hardware_cancel,self.log)
                    result['message']+=('；' if result['message'] else '')+result['main_camera']['message']
                    self.camera_watch.completed(usb_snapshot())
                    result['camera_usb']=camera_usb_status()
                elif camera_only:result.update(skipped=True,message='隔离测试：跳过主相机恢复')
                result['can_ok']=result['ok'] if not camera_only else None
                result['ok']=result['ok'] and result.get('drivers',{}).get('ok',True) and result.get('main_camera',{}).get('ok',True)
                write(self.directory/'hardware_startup.json',result)
                self.log(result['message'])
                self.events.put(dict(type='hardware',message=result['message'],result=result))
            except Exception as exc:
                self.log(str(exc));self.events.put(dict(type='hardware',message='连接检查失败：'+str(exc)))
            finally:self.hardware_busy=False
        self.hardware_thread=threading.Thread(target=work,daemon=True);self.hardware_thread.start()
    def check_camera_reconnect(self):
        if not self.camera_watch_enabled or self.busy or self.hardware_busy or self.stop_jobs or self.closing:return
        now=time.monotonic()
        if now-self.camera_watch.last_check<2:return
        self.camera_watch.last_check=now
        from .hardware_startup import isolated
        if isolated():return
        from .camera_reconnect import usb_snapshot
        if self.camera_watch.due(self.telemetry_state(),usb_snapshot(),now):
            self.log('检测到主相机接入或画面中断，自动恢复连接')
            self.recover_connection(camera_only=True)
    def _worker(self,mode,directory,kind=None):
        command=[sys.executable,'-m','nero_eval_workbench.worker',mode,'--directory',str(directory),'--heartbeat',str(self.heartbeat)]
        if kind:command+=['--kind',kind]
        with (directory/f'{mode}.log').open('w') as stream:
            self.process=subprocess.Popen(command,cwd=PLATFORM,stdout=stream,stderr=subprocess.STDOUT)
        code=self.process.wait();self.process=None
        if code:
            if (directory/'job_error.json').exists():raise RuntimeError(read(directory/'job_error.json')['error'])
            if (directory/'result.json').exists():return read(directory/'result.json')
            raise RuntimeError((directory/f'{mode}.log').read_text(errors='replace')[-3000:])
        return read(directory/('plan.json' if mode=='prepare' else 'result.json'))
    def start(self,kind,settings,directory=None):
        if self.hardware_busy:raise RuntimeError('正在恢复设备连接，请完成后再启动任务')
        if self.busy:raise RuntimeError('已有任务运行，请先停止或等待完成')
        settings.checked();self.busy=True;self.cancel.clear()
        session=Path(directory) if directory else new_session(kind);self.active=session
        if not directory:write(session/'config.json',dict(settings=settings.json(),port=self.server.port or 8026))
        def work():
            try:
                if kind in ('load','offline','shadow','formal'):
                    port=self.server.ensure(settings,self.cancel);write(session/'config.json',dict(settings=settings.json(),port=port))
                    self.events.put(dict(type='model',task=settings.task,model=settings.model,port=port))
                if self.cancel.is_set():raise RuntimeError('任务已取消')
                if kind=='load':result=dict(model_loaded=True)
                elif kind=='pose':result=ready_pose(settings.task)
                elif kind=='offline':result=asyncio.run(offline_replay(session,settings,self.server.port,self.cancel,self.log))
                elif kind=='execute':result=self._worker('execute',session)
                elif kind=='shadow':result=self._worker('shadow',session)
                else:result=self._worker('prepare',session,kind)
                self.events.put(dict(type='done',kind=kind,directory=str(session),result=result))
            except Exception as exc:self.events.put(dict(type='error',kind=kind,directory=str(session),error=str(exc)))
            finally:self.busy=False
        self.thread=threading.Thread(target=work,daemon=True);self.thread.start();return session
    def stop(self,physical=False):
        self.cancel.set()
        if self.active:(self.active/'STOP').touch()
        self.log('已请求停止'+('，同时关闭双臂控制并请求保持。' if physical else '。'))
        if physical:
            directory=new_session('emergency_stop')
            with (directory/'stop.log').open('w') as stream:
                proc=subprocess.Popen([sys.executable,'-m','nero_eval_workbench.worker','stop','--directory',str(directory)],cwd=PLATFORM,stdout=stream,stderr=subprocess.STDOUT)
            self.stop_jobs.append((proc,directory))
    def close(self):
        self.closing=True;self.cancel.set();self.hardware_cancel.set()
        if self.active:(self.active/'STOP').touch()
        (self.directory/'STOP_TELEMETRY').touch()
        # Keep heartbeat alive while the GUI waits for cooperative cleanup.
        if self.busy or self.hardware_busy:return False
        self.server.stop()
        if self.telemetry:
            try:self.telemetry.wait(timeout=3)
            except subprocess.TimeoutExpired:self.telemetry.terminate()
        return True
