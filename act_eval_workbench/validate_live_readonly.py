"""Exercise production backend/model manager/worker against live sensors; no motion."""
import threading
import time
from .common import *
from .backend import Backend

def main():
    backend=Backend(start_telemetry=False);keep=threading.Event();keep.set();rows=[]
    def beat():
        while keep.is_set():backend.tick();time.sleep(.1)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    print('READONLY_DIRECTORY',backend.directory,flush=True)
    try:
        for model in MODELS:
            settings=Settings(task='plate',model=model,duration_s=3)
            session=backend.start('shadow',settings)
            while backend.busy:time.sleep(.1)
            messages=[]
            while not backend.events.empty():messages.append(backend.events.get())
            errors=[x for x in messages if x['type']=='error']
            result=read(session/'result.json') if (session/'result.json').exists() else None
            row=dict(model=model,directory=str(session),errors=errors,result=result)
            rows.append(row);write(backend.directory/'live_readonly_summary.json',dict(physical_motion_executed=False,results=rows))
            print(json.dumps(dict(model=model,errors=errors,completed=result.get('runtime_completed') if result else False,
                inferences=result.get('inferences') if result else None,error=result.get('error') if result else None)),flush=True)
    finally:
        backend.close();keep.clear();thread.join(timeout=2)
    if not all(x['result'] and x['result'].get('runtime_completed') for x in rows):raise SystemExit(2)
if __name__=='__main__':main()
