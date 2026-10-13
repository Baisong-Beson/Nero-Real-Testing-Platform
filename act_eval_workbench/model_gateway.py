"""Owned inference gateway. No ROS imports or robot control capability."""
import argparse
import asyncio
import json
from pathlib import Path
import numpy as np
from .common import catalog
from .library import observation
from .policy_protocol import SCHEMA,actions,validate_service,validate_observation

async def serve(task,model,port):
    from nero_pi05_bridge.policy_client import PolicyClient
    from nero_pi05_bridge import msgpack_numpy
    import websockets
    identity=catalog(task,model,True);cfg=identity['runtime'];definition=identity['model_definition']
    policy=None
    if identity['adapter']=='sealed_policy':
        import torch
        from nero_pi05_bridge.act_policy_server import ACTPolicy
        policy=ACTPolicy(Path(identity['policy_path']),torch.device('cuda:0'))
        if policy.metadata['policy_sha256']!=identity['policy_sha256'] or policy.metadata['encoder_state_sha256']!=identity['encoder_state_sha256']:raise ValueError('加载权重身份不匹配')
    elif identity['adapter']=='openpi_nero':
        from .openpi_nero_adapter import NativePolicy
        policy=NativePolicy(identity)
        # Complete JAX compilation before advertising ready; it is not a normal
        # inference and must not repeatedly hit the client's runtime deadline.
        from .library import probe_paths,CAMERAS
        probe_path,_=probe_paths(task,model)
        if probe_path is None:raise ValueError('本地模型首次编译需要包内三相机 probe_file')
        with np.load(probe_path,allow_pickle=False) as z:
            extra={CAMERAS[k]:z['rgb_'+k][0] for k in cfg['cameras'] if k!='main'}
            print('正在编译并预热本地模型…',flush=True)
            policy.infer(observation(identity,z['rgb'][0],z['state'][0],extra))
        print('本地模型编译和预热完成',flush=True)
    meta=dict(identity,schema=SCHEMA,commandable=False,action_shape=[cfg['action_horizon'],16])
    lock=asyncio.Lock()
    async def handler(ws,*_):
        client=None
        try:
            if policy is None:
                client=PolicyClient(definition['endpoint'],0,connect_timeout_sec=5,inference_timeout_sec=cfg['inference_timeout_s'])
                await client.connect();validate_service(client.metadata,definition)
            await ws.send(msgpack_numpy.packb(meta))
            async for message in ws:
                obs=msgpack_numpy.unpackb(message)
                obs['prompt']=identity['instruction'];obs['task_id']=task
                validate_observation(obs,cfg['cameras'])
                async with lock:
                    result=await client.infer(obs) if client else await asyncio.to_thread(policy.infer,obs)
                result=dict(result);result['actions']=actions(result['actions'],cfg['action_horizon'])
                await ws.send(msgpack_numpy.packb(result))
        except Exception as exc:
            try:await ws.send(f'{type(exc).__name__}: {exc}')
            except Exception:pass
        finally:
            if client:await client.close()
    async with websockets.serve(handler,'127.0.0.1',port,max_size=64*1024*1024,compression=None):await asyncio.Future()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--model',required=True);p.add_argument('--port',type=int,required=True)
    a=p.parse_args();asyncio.run(serve(a.task,a.model,a.port))
