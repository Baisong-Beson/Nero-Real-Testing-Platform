"""Replay saved reset paths against a delayed plant, without ROS/SDK/CAN.

This is a fault model, not a reconstruction of unrecorded hardware feedback.
"""
import json
import numpy as np
from act_eval_workbench import reset_motion as r, worker as w
from act_eval_workbench.common import RUNS,new_session,read,write

class Plant:
    def __init__(self,path):
        self.path=path;self.t=0.;self.actual=np.array(path['start']);self.target=self.actual.copy()
        self.widths=np.array(path['gripper_widths_m'][0]);self.interrupted=False
        self.feedback=w.c.pos.Feedback(require_open=False);self.sent=[];self.peak_lead=0.;self.peak_speed=0.
        for stamp in np.arange(-.8,0,.005):self.record(float(stamp))
    def record(self,stamp):
        for i,arm in enumerate(w.c.pos.ARMS):
            self.feedback.add(arm,list(w.c.pos.NAMES)+['gripper'],self.actual[i].tolist()+self.widths[i:i+1].tolist(),1000+stamp,stamp)
    def now(self):return self.t
    def pump(self):pass
    def idle(self):
        self.t+=.005
        if not (self.t<1. or 4.<self.t<4.1):self.actual+=np.clip(self.target-self.actual,-.002,.002)
        self.record(self.t)
    def snapshot(self,stationary=False):
        for rows in self.feedback.rows.values():
            last=rows[-1];older=[x for x in rows if .1<=last['stamp']-x['stamp']<=.2]
            if older:self.peak_speed=max(self.peak_speed,float(np.max(np.abs(last['q']-older[-1]['q']))/(last['stamp']-older[-1]['stamp'])))
        return self.feedback.snapshot(self.t,1000+self.t,stationary)
    def check_publishers(self,owned=False):pass
    def prepare_publishers(self):pass
    def call_pair(self,*args):pass
    def event(self,*args,**kwargs):pass
    def publish(self,q,actual,index):
        self.target=q.copy();self.sent.append(q.copy());self.widths=np.array(self.path['gripper_widths_m'][index])
        self.peak_lead=max(self.peak_lead,float(np.max(np.abs(q-actual))))
    def result(self,error=None):
        qs=np.array([self.path['start']]+self.sent)
        return dict(error=error,commands=len(self.sent),simulated_seconds=self.t,
            peak_lead_rad=self.peak_lead,peak_measured_speed_rad_s=self.peak_speed,
            max_command_velocity_rad_s=float(np.abs(np.diff(qs,axis=0)/r.DT).max(initial=0)),
            max_command_acceleration_rad_s2=float(np.abs(np.diff(qs,n=2,axis=0)/r.DT**2).max(initial=0)))

def main():
    out=new_session('reset_tracking_replay');results=[]
    for name in ['20260923_232129_617501586_ready','20260923_232150_274321690_default']:
        body=read(RUNS/name/'plan.json');old=body['trajectory']
        old={k:np.array(v) if k in ('start','goal','joints','gripper_widths_m') else v for k,v in old.items()}
        io=Plant(old);error=None
        try:
            last=io.now()
            for index,q in enumerate(old['joints']):
                while io.now()-last<r.DT:io.snapshot();io.idle()
                actual=io.snapshot()[0]
                if np.max(np.abs(q-actual))>.03:raise RuntimeError('legacy tracking lead >0.03 rad')
                io.publish(q,actual,index);last=io.now()
        except RuntimeError as exc:error=str(exc)
        baseline=io.result(error)
        path=w.reset_plan(old['start'],old['goal'],body['start_widths_m'],via_default=body['kind']=='ready')
        w.reset_geometry(path);io=Plant(path)
        answer=r.run_motion(io,path,lambda:True,lambda *_:None)
        assert answer['ok']
        revised=io.result();assert revised['peak_measured_speed_rad_s']<=.2
        results.append(dict(original_session=name,old_time_based=baseline,new_feedback_driven=revised))
    write(out/'summary.json',dict(physical_motion_executed=False,
        scope='Saved paths, simulated one-second startup hold and 100 ms midpath hold; not measured hardware dynamics.',results=results))
    print('REPLAY_REPORT',out);print(json.dumps(results,indent=2))

if __name__=='__main__':main()
