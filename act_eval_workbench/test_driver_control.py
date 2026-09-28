import unittest
import numpy as np
from types import SimpleNamespace as NS
from . import controller as c
from .common import DEFAULT_DEG,UPPER,Settings

class Tests(unittest.TestCase):
    def test_raw_targets_reach_dispatch_without_diagnostic_filter(self):
        start=np.deg2rad(DEFAULT_DEG);target=start.copy();target[:,0]+=.3
        control=c.DriverTargets(start,[.1,.1]);q,w=control.next(target,[0.,.1],.05,start)
        np.testing.assert_array_equal(q,target);np.testing.assert_array_equal(w,[0.,.1])
    def test_physical_range_and_nonfinite_targets_still_rejected(self):
        q=np.deg2rad(DEFAULT_DEG);driver=c.DriverTargets(q,[.1,.1])
        for target in [np.full((2,7),float('nan')),np.tile(UPPER+.001,(2,1))]:
            with self.assertRaises(ValueError):driver.next(target,[.1,.1],.05)
        for widths in ([float('nan'),.1],[-.01,.1],[.1,.11]):
            with self.assertRaises(ValueError):driver.next(q,widths,.05)
    def feedback(self,speed):
        f=c.pos.Feedback(require_open=False);q=np.deg2rad(DEFAULT_DEG)
        for t in np.arange(0,.801,.005):
            now=q.copy();now[:,0]+=speed*t
            for i,a in enumerate(c.pos.ARMS):f.add(a,list(c.pos.NAMES)+['gripper'],now[i].tolist()+[.1],1000+t,t)
        return f
    def test_ordinary_firmware_speed_no_longer_hits_point_two_abort(self):
        f=self.feedback(.8);q,_=c.feedback_snapshot(f,.8,1000.8)
        self.assertAlmostEqual(q[0,0],.64)
        with self.assertRaisesRegex(RuntimeError,'not stationary'):c.feedback_snapshot(f,.8,1000.8,True)
    def test_stale_skew_and_invalid_feedback_remain_fail_closed(self):
        f=self.feedback(0.)
        with self.assertRaisesRegex(RuntimeError,'stale'):c.feedback_snapshot(f,1.1,1001.1)
        f.rows['left'][-1]['stamp']-=.07
        with self.assertRaisesRegex(RuntimeError,'skew'):c.feedback_snapshot(f,.8,1000.8)
        f.error='invalid names'
        with self.assertRaisesRegex(RuntimeError,'invalid names'):c.feedback_snapshot(f,.8,1000.8)
    def test_gap_discards_old_target_and_requests_replan(self):
        actual=np.deg2rad(DEFAULT_DEG);target=actual.copy();target[:,0]+=.3
        d=c.DriverTargets(actual,[.1,.1])
        _,q,w,rebase=c.advance_after_delay(d,target,[0.,0.],actual,[.1,.1],.2)
        self.assertTrue(rebase);np.testing.assert_array_equal(q,actual);np.testing.assert_array_equal(w,[.1,.1])
    def test_protocol_exposes_no_application_rate_limits(self):
        p=Settings().protocol();self.assertEqual(p['version'],c.PROTOCOL)
        for k in ['joint_speed_rad_s','joint_accel_rad_s2','target_lead_rad','hard_lead_rad','measured_speed_abort_rad_s']:
            self.assertIsNone(p[k])
        self.assertEqual(p['target_filter'],'firmware_move_j_absolute_targets')

    def test_readonly_preflight_accepts_driver_full_range(self):
        q=np.deg2rad(DEFAULT_DEG)
        for speed,allowed in [(35,True),(100,True),(0,False),(101,False)]:
            response=NS(values=[NS(type=4,string_value='nero'),NS(type=2,integer_value=speed),NS(type=1,bool_value=False)])
            client=NS(service_is_ready=lambda:True,call_async=lambda request:None)
            services=[(f'/{arm}_arm/{s}',[t]) for arm in c.pos.ARMS for s,t in [('control_enable','std_srvs/srv/SetBool'),('enable_agx_arm','std_srvs/srv/SetBool'),('emergency_stop','std_srvs/srv/Empty')]]
            io=NS(now=lambda:0.,pump=lambda:None,interrupted=False,gui=None,
                snapshot=lambda stationary=False:(q,[.1,.1]),camera=lambda:None,
                clients={(a,'params'):client for a in c.pos.ARMS},
                node=NS(get_service_names_and_types=lambda:services),check_publishers=lambda:None,
                wait_futures=lambda futures:{a:response for a in c.pos.ARMS},event=lambda *a,**k:None)
            if allowed:
                c.driver_preflight(io);self.assertEqual(io.driver_parameters['left']['speed_percent'],speed)
            else:
                with self.assertRaises(RuntimeError):c.driver_preflight(io)

if __name__=='__main__':unittest.main()
