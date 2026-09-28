"""Explicit small hardware check of the production policy motion filter.

Synthetic joint4 target +0.3 deg, then return; never a model/task success trial.
"""
import argparse
import time
import traceback
import numpy as np
from act_eval_workbench import controller as c
from act_eval_workbench.common import new_session, write
from act_eval_workbench.worker import motion_lock, stop_services


class Gate:
    closed=False; stopped=False
    def __init__(self,directory): self.directory=directory;self.began=time.monotonic()
    def pump(self): pass
    def permit(self): return time.monotonic()-self.began<30 and not (self.directory/'STOP').exists()


class BoundedIO(c.FormalIO):
    def snapshot(self,stationary=False):
        q,w=super().snapshot(stationary)
        if hasattr(self,'origin'):
            self.observed=max(self.observed,float(np.max(np.abs(q-self.origin))))
            if self.observed>np.deg2rad(.5): raise RuntimeError('measured movement exceeds 0.5 degrees')
        return q,w
    def publish_full(self,q,*args,**kwargs):
        if np.max(np.abs(q-self.origin))>np.deg2rad(.31): raise RuntimeError('command exceeds 0.31 degrees')
        return super().publish_full(q,*args,**kwargs)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute-small-motion',action='store_true');args=parser.parse_args()
    if not args.execute_small_motion: parser.error('requires --execute-small-motion')
    directory=new_session('slew_bounded_real');gate=Gate(directory)
    report=dict(kind='motion_filter_micro_validation',protocol=c.PROTOCOL,physical_motion_executed=False,legs=[])
    io=rec=None;owned=False
    with motion_lock():
        try:
            io=BoundedIO(directory,gate);origin,widths=io.preflight()
            if max(abs(np.array(widths)-.1))>.002: raise RuntimeError('requires already-open grippers')
            io.origin=origin.copy();io.observed=0.
            rec=c.VideoRecorder(directory,c.decode_compressed_image(io.camera()['data']).shape)
            io.recorder=rec;io.monitor_camera=True;geometry=c.Geometry(origin,widths)
            def guard():
                io.pump()
                if not gate.permit() or io.interrupted: raise RuntimeError('bounded verification stopped')
                io.check_publishers(owned=owned)
                return io.snapshot()
            io.prepare_publishers();owned=True
            for service,value in [('control_enable',False),('enable_agx_arm',True),('control_enable',True)]:
                guard();io.call_pair(service,value,gate.permit)
            for label in ('out','return'):
                q,w=guard();slew=c.Slew(q,w);goal=origin.copy()
                if label=='out':goal[:,3]+=np.deg2rad(.3)
                geometry.check(goal,[.1,.1]);last=io.now();began=last;count=io.command_count
                while True:
                    actual,widths=guard()
                    if io.now()-began>8: raise RuntimeError('bounded leg did not arrive')
                    if io.now()-last<c.DT: io.idle();continue
                    tick=io.now()
                    slew,q,w,rebased=c.advance_after_delay(slew,goal,[.1,.1],actual,widths,tick-last)
                    geometry.check(q,w,commit=True)
                    io.publish_full(q,w,actual,widths,io.command_count,0,0);last=tick
                    if np.max(np.abs(actual-goal))<=.001 and np.max(np.abs(slew.v))<.001:
                        try: final,_=io.snapshot(stationary=True)
                        except RuntimeError as exc:
                            if 'not stationary' in str(exc) or 'need 0.5 s' in str(exc): continue
                            raise
                        if np.max(np.abs(final-goal))>.001: continue
                        break
                report['legs'].append(dict(label=label,commands=io.command_count-count,
                    max_error_deg=float(np.rad2deg(np.max(np.abs(final-goal)))),elapsed_s=io.now()-began))
            report.update(completed=True,max_observed_deg=float(np.rad2deg(io.observed)))
        except Exception as exc: report.update(completed=False,error=str(exc),traceback=traceback.format_exc())
        finally:
            if io:
                report['physical_motion_executed']=bool(io.command_count)
                report['commands']=io.command_count
                io.recorder=None;io.monitor_camera=False
                try: io.close()
                except Exception as exc: report['close_error']=str(exc)
            try:
                if rec: report['video']=rec.close()
            finally:
                report['final_disable']=stop_services(True)
                write(directory/'validation.json',report)
                print(directory);print(report)
    return 0 if report.get('completed') and not report['final_disable']['errors'] else 2


if __name__=='__main__': raise SystemExit(main())
