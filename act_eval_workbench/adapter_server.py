"""VLA SDK: serve a user's Python policy factory with the NERO wire protocol.

Run in the model's own environment. User factory receives the JSON config,
returns an object with infer(observation) -> {actions: float32[H,16]}.
Only a user-launched CLI loads factory code; importing a manifest never does.
"""
import argparse
import asyncio
import importlib
import json
from pathlib import Path
from .policy_protocol import metadata,actions,validate_observation
from .library import runtime_config

async def serve(factory,definition,config,host,port):
    import websockets
    from nero_pi05_bridge import msgpack_numpy
    module,name=factory.split(':',1);policy=getattr(importlib.import_module(module),name)(config)
    cfg=runtime_config(definition)
    meta=metadata(definition['service_model_id'],definition['checkpoint_sha256'],cfg)
    lock=asyncio.Lock()
    async def handler(ws,*_):
        await ws.send(msgpack_numpy.packb(meta))
        async for message in ws:
            try:
                obs=msgpack_numpy.unpackb(message);validate_observation(obs,cfg['cameras'])
                async with lock:result=await asyncio.to_thread(policy.infer,obs)
                result=dict(result);result['actions']=actions(result['actions'],cfg['action_horizon'])
                await ws.send(msgpack_numpy.packb(result))
            except Exception as exc:await ws.send(f'{type(exc).__name__}: {exc}')
    async with websockets.serve(handler,host,port,max_size=64*1024*1024,compression=None):await asyncio.Future()

if __name__=='__main__':
    p=argparse.ArgumentParser(description='NERO VLA inference adapter server')
    p.add_argument('--factory',required=True,help='Python module:function');p.add_argument('--model',type=Path,required=True)
    p.add_argument('--config',type=Path);p.add_argument('--host',default='127.0.0.1');p.add_argument('--port',type=int,default=9000)
    a=p.parse_args();definition=json.loads(a.model.read_text());config=json.loads(a.config.read_text()) if a.config else {}
    asyncio.run(serve(a.factory,definition,config,a.host,a.port))
