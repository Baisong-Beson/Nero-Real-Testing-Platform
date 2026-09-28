"""Firmware target reset: no application ramp, verified arrival and stop faults."""
import unittest
from unittest.mock import Mock
import numpy as np
from . import reset_motion as r, worker as w
from .common import DEFAULT_DEG,ready_pose,ready_gripper_target,check_start_grippers

class IO:
    def __init__(self,plan,stuck=False,grip_stuck=False,delay=0.):
        self.plan=plan;self.t=0.;self.actual=plan['start'].copy();self.target=self.actual.copy()
        self.width=plan['gripper_widths_m'][0].copy();self.target_w=self.width.copy()
        self.interrupted=False;self.stuck=stuck;self.grip_stuck=grip_stuck;self.delay=delay
        self.commands=[];self.calls=[];self.events=[]
    def now(self):return self.t
    def pump(self):pass
    def idle(self):
        self.t+=.005
        if not self.stuck:self.actual+=np.clip(self.target-self.actual,-.8*.005,.8*.005)
        if not self.grip_stuck:self.width+=np.clip(self.target_w-self.width,-.04*.005,.04*.005)
        if self.delay and self.t>.4:self.t+=self.delay;self.delay=0.
    def snapshot(self,stationary=False):
        if stationary and np.max(np.abs(self.target-self.actual))>1e-8:raise RuntimeError('arm not stationary')
        return self.actual.copy(),self.width.tolist()
    def check_publishers(self,owned=False):pass
    def prepare_publishers(self):pass
    def call_pair(self,service,value,permit):self.calls.append((service,value))
    def event(self,name,**values):self.events.append((name,values))
    def publish(self,q,actual,index):
        self.commands.append((q.copy(),self.width.copy(),index));self.target=q.copy()
        self.target_w=self.plan['gripper_widths_m'][index].copy()
    def gripper_contact(self,since,widths):return None

class Tests(unittest.TestCase):
    def plan(self,widths=(.1,.1),via=False):
        start=np.deg2rad(DEFAULT_DEG);start[:,3]-=.4;goal=np.deg2rad(DEFAULT_DEG);goal[:,0]+=.2
        return w.reset_plan(start,goal,widths,via_default=via)
    def run_plan(self,plan,**kwargs):
        io=IO(plan,**kwargs);result=r.run_motion(io,plan,lambda:True,lambda *_:None)
        self.assertTrue(result['ok']);np.testing.assert_allclose(io.actual,plan['goal'],atol=r.FINAL_TOL)
        self.assertNotIn(('enable_agx_arm',False),io.calls);return io,result
    def test_native_endpoints_not_hundreds_of_slow_commands(self):
        io,result=self.run_plan(self.plan())
        self.assertEqual(result['commands_sent'],2);self.assertLess(io.t,1.)
        self.assertGreater(np.max(np.abs(io.commands[-1][0]-io.commands[0][0])),.3)
    def test_both_grippers_open_before_any_joint_motion(self):
        io,result=self.run_plan(self.plan((0.,.029),True))
        self.assertEqual(result['commands_sent'],4)
        for q,width,index in io.commands:
            if np.max(np.abs(q-io.plan['start']))>1e-8:self.assertLessEqual(np.max(np.abs(width-.1)),.002)
        np.testing.assert_allclose(io.commands[-2][0],np.deg2rad(DEFAULT_DEG))
    def test_gripper_stall_never_sends_reset_endpoint(self):
        path=self.plan((0.,0.));io=IO(path,grip_stuck=True)
        with self.assertRaisesRegex(RuntimeError,'张开无进展'):r.run_motion(io,path,lambda:True,lambda *_:None)
        for q,_,_ in io.commands:np.testing.assert_array_equal(q,path['start'])
        self.assertEqual(io.calls[-2:],[('control_enable',False),('emergency_stop',None)])
    def test_joint_stall_stops_without_retries_or_disable(self):
        path=self.plan();io=IO(path,stuck=True)
        with self.assertRaisesRegex(RuntimeError,'no progress'):r.run_motion(io,path,lambda:True,lambda *_:None)
        self.assertEqual(len(io.commands),2);self.assertNotIn(('enable_agx_arm',False),io.calls)
    def test_operator_stop_while_driver_moves(self):
        path=self.plan();io=IO(path)
        with self.assertRaisesRegex(RuntimeError,'operator stopped'):r.run_motion(io,path,lambda:io.t<.1,lambda *_:None)
        self.assertEqual(io.calls[-1],('emergency_stop',None))
    def test_monitor_gap_holds_but_brief_delay_is_tolerated(self):
        self.run_plan(self.plan(),delay=.1641)
        path=self.plan();io=IO(path,delay=.36)
        with self.assertRaisesRegex(RuntimeError,'监控中断'):r.run_motion(io,path,lambda:True,lambda *_:None)
    def test_spatial_audit_is_not_a_timing_promise(self):
        path=self.plan((0.,.029),True)
        self.assertIsNone(path['duration_s']);self.assertTrue(path['samples_are_spatial_only'])
        self.assertLessEqual(np.max(np.abs(np.diff(path['joints'],axis=0))),.01000001)
        self.assertLessEqual(np.max(np.abs(np.diff(path['gripper_widths_m'],axis=0))),.00500001)
        w.reset_geometry(path)
    def banana_plan(self):
        return w.reset_plan(np.deg2rad(DEFAULT_DEG),np.array(ready_pose('banana')['joints_rad']),
            [.029,.04],via_default=True,final_widths=ready_gripper_target('banana'),contact_grasp=True)
    def test_banana_closes_right_only_after_joint_arrival(self):
        plan=self.banana_plan();io,result=self.run_plan(plan)
        q,_,index=io.commands[-1]
        self.assertEqual(index,plan['final_gripper_index'])
        np.testing.assert_allclose(q,plan['goal'])
        self.assertLessEqual(result['final_widths_m'][0],.002)
        self.assertAlmostEqual(result['final_widths_m'][1],.1)
        check_start_grippers('banana',result['final_widths_m'])
        names=[name for name,_ in io.events]
        self.assertGreater(names.index('reset_task_grippers_started'),max(i for i,n in enumerate(names) if n=='reset_segment_completed'))
        self.assertIn('reset_task_grippers_verified',names)
        for target,measured,index in io.commands[:-1]:
            if index>plan['opening_last_index']:self.assertLessEqual(np.max(np.abs(measured-.1)),.002)
        audit=w.reset_geometry(plan)
        np.testing.assert_allclose(audit['gripper_goal_m'],ready_gripper_target('banana'))
    def test_banana_closing_stall_or_cancel_does_not_report_completion(self):
        for stop in (False,True):
            with self.subTest(stop=stop):
                plan=self.banana_plan();io=IO(plan);publish=io.publish
                def close(q,actual,index):
                    publish(q,actual,index)
                    if index==plan['final_gripper_index']:
                        io.grip_stuck=True;io.interrupted=stop
                io.publish=close
                with self.assertRaisesRegex(RuntimeError,'operator stopped' if stop else '收拢无进展'):
                    r.run_motion(io,plan,lambda:True,lambda *_:None)
                self.assertEqual(io.calls[-1],('emergency_stop',None))
                self.assertFalse(next(v for name,v in io.events if name=='cleanup')['reached_target'])
    def test_banana_closing_joint_drift_stops(self):
        plan=self.banana_plan();io=IO(plan);idle=io.idle
        def drift():
            idle()
            if io.target_w[0]<.1:io.actual[0,0]+=.02
        io.idle=drift
        with self.assertRaisesRegex(RuntimeError,'关节偏移'):r.run_motion(io,plan,lambda:True,lambda *_:None)
    def test_contact_width_need_not_equal_close_target_or_training_width(self):
        from .gripper_contact import ContactFeedback
        plan=self.banana_plan();io=IO(plan);feedback=ContactFeedback();idle=io.idle
        def contact_idle():
            idle()
            touching=io.target_w[0]==0 and io.width[0]<=.04205
            if touching:io.width[0]=.04205
            feedback.add(1000+io.t,io.t,io.width[0],1. if touching else 0.,True,False)
        io.idle=contact_idle
        io.gripper_contact=lambda since,widths:feedback.contact(io.t,1000+io.t,since,widths[0])
        result=r.run_motion(io,plan,lambda:True,lambda *_:None)
        self.assertTrue(result['ok']);self.assertEqual(result['gripper_completion']['mode'],'force_contact')
        self.assertAlmostEqual(result['final_widths_m'][0],.04205)
        self.assertEqual(result['gripper_goal_m'],[0.,.1]);self.assertFalse(result['object_presence_verified'])
    def test_reset_publisher_requires_goal_for_closing_and_open_for_moving(self):
        plan=self.banana_plan();io=object.__new__(w.ResetIO);io.reset_path=plan
        io.snapshot=lambda:(plan['goal'].copy(),[.04,.1]);io.publish_full=Mock()
        io.publish(plan['goal'],plan['goal'],plan['final_gripper_index'])
        io.publish_full.assert_called_once()
        with self.assertRaisesRegex(RuntimeError,'尚未到达'):
            io.publish(plan['goal'],plan['start'],plan['final_gripper_index'])
        with self.assertRaisesRegex(RuntimeError,'未张开'):
            io.publish(plan['goal'],plan['start'],plan['segment_end_indices'][-1])
    def test_default_and_open_only_never_close_a_gripper(self):
        for opening_only in (False,True):
            plan=w.reset_plan(np.deg2rad(DEFAULT_DEG),np.deg2rad(DEFAULT_DEG),[.028,.04],opening_only=opening_only)
            io,result=self.run_plan(plan)
            self.assertIsNone(plan['final_gripper_index'])
            self.assertLessEqual(np.max(np.abs(np.array(result['final_widths_m'])-.1)),.002)
        with self.assertRaises(ValueError):w.reset_plan(plan['start'],plan['goal'],[.1,.1],opening_only=True,final_widths=[.028,.1])

if __name__=='__main__':unittest.main()
