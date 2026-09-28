"""Explicit, bounded real validation: joint4 +0.3 degrees, return, disable all.

Only runs with --execute-small-motion. Never invokes a full default/ready reset.
"""
import argparse
import json
import time
import traceback
import numpy as np
from act_eval_workbench.worker import ResetIO,reset_plan,reset_geometry,stop_services,motion_lock,c,serialize
from act_eval_workbench import reset_motion as r
from act_eval_workbench.common import new_session,write

class Gate:
    closed=False;stopped=False
    def __init__(self):self.began=time.monotonic()
    def pump(self):pass
    def permit(self):return time.monotonic()-self.began<30.

class BoundedIO(ResetIO):
    def snapshot(self,stationary=False):
        q,w=super().snapshot(stationary)
        if hasattr(self,'origin'):
            delta=float(np.max(np.abs(q-self.origin)));self.observed=max(self.observed,delta)
            if delta>np.deg2rad(.5):raise RuntimeError('bounded verification exceeded 0.5 deg')
        return q,w
    def publish(self,q,actual,index):
        # A return path starts at measured feedback, which can differ slightly
        # from the previous 0.3-degree target. Permit holding that exact start,
        # never a larger excursion; all measured motion remains bounded at 0.5.
        if np.max(np.abs(q-self.origin))>self.command_extent+1e-9:raise RuntimeError('verification target exceeded bounded path')
        return super().publish(q,actual,index)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute-small-motion',action='store_true');args=parser.parse_args()
    if not args.execute_small_motion:parser.error('requires --execute-small-motion to enable physical motion')
    directory=new_session('reset_tracking_bounded_real')
    report=dict(kind='reset_tracking_micro_validation',scope='joint4 +0.3 deg and return using independent connections',physical_motion_executed=False,legs=[])
    io=rec=None;origin=None
    with motion_lock():
        try:
            for label in ['out','return']:
                sub=directory/label;sub.mkdir();gate=Gate();io=BoundedIO(sub,gate);q,widths=io.preflight()
                if origin is None:origin=q.copy()
                io.origin=origin;io.observed=0.
                io.command_extent=max(np.deg2rad(.301),float(np.max(np.abs(q-origin))))
                if io.command_extent>np.deg2rad(.5):raise RuntimeError('verification start outside 0.5-degree envelope')
                if max(abs(np.array(widths)-.1))>.002:raise RuntimeError('verification requires already-open grippers')
                image=c.diag.decode_compressed_image(io.camera()['data']);rec=c.VideoRecorder(sub,image.shape);io.recorder=rec;io.monitor_camera=True
                goal=origin.copy()
                if label=='out':goal[:,3]+=np.deg2rad(.3)
                path=reset_plan(q,goal,widths);reset_geometry(path);io.reset_path=path;write(sub/'plan.json',serialize(path))
                answer=r.run_motion(io,path,gate.permit,lambda *_:None)
                report['physical_motion_executed']=report['physical_motion_executed'] or bool(io.command_count)
                leg=dict(label=label,result=answer,max_observed_deg=float(np.rad2deg(io.observed)),commands=io.command_count)
                io.recorder=None;io.monitor_camera=False;io.close();io=None
                leg['video']=rec.close();rec=None;report['legs'].append(leg)
                assert answer['max_error_rad']<=r.FINAL_TOL
            report['completed']=True
        except Exception as exc:report.update(completed=False,error=str(exc),traceback=traceback.format_exc())
        finally:
            if io:
                report.update(physical_motion_executed=report['physical_motion_executed'] or bool(io.command_count),unfinished_leg_commands=io.command_count)
                io.recorder=None;io.monitor_camera=False
                try:io.close()
                except Exception as exc:report['close_error']=str(exc)
            if rec:report['unfinished_leg_video']=rec.close()
            report['final_disable']=stop_services(True)
            write(directory/'validation.json',report)
            print('BOUNDED_REAL_REPORT',directory);print(json.dumps(report,ensure_ascii=False,indent=2))
    return 0 if report.get('completed') and not report['final_disable']['errors'] else 2

if __name__=='__main__':raise SystemExit(main())
