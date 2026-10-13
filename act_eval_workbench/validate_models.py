"""Full GPU RGB→encoder→ACT test for all nine sealed policies; no ROS control."""
import asyncio
import threading
import traceback
from .common import *
from .backend import ModelServer,offline_replay

def main():
    directory=new_session('nine_model_validation');print('VALIDATION_DIRECTORY',directory,flush=True)
    cancel=threading.Event();server=ModelServer(directory,lambda x:print(x,flush=True));results=[]
    try:
        for model in MODELS:
            for task in TASKS:
                settings=Settings(task=task,model=model)
                sub=directory/(task+'_'+model);sub.mkdir()
                try:
                    port=server.ensure(settings,cancel)
                    report=asyncio.run(offline_replay(sub,settings,port,cancel,lambda x:print(x,flush=True)))
                    result=dict(task=task,model=model,passed=report['golden_passed'],
                        max_abs_error=max(row['max_abs_error'] for row in report['probes']),
                        inference_ms=[row['inference_ms'] for row in report['probes']],
                        trajectory_warnings=[row['violations'] for row in report['probes']])
                except Exception as exc:
                    result=dict(task=task,model=model,passed=False,error=str(exc),traceback=traceback.format_exc())
                results.append(result);write(directory/'summary.json',dict(physical_motion_executed=False,results=results))
                print(json.dumps(result,ensure_ascii=False),flush=True)
    finally:server.stop()
    if not all(r['passed'] for r in results):raise SystemExit(2)
if __name__=='__main__':main()
