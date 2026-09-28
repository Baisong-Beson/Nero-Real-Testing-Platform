"""Relative debugging caps are optional; absolute and step guards stay active."""
import unittest
from unittest.mock import patch
import numpy as np
from .common import Settings,ready_pose,digest,UPPER
from . import controller as c

class Tests(unittest.TestCase):
    def setUp(self):
        self.bounds=patch.dict(c.BOUNDS,relative_limits_enabled=False,joint_excursion_rad=np.deg2rad(45),tcp_displacement_m=.20,frame_displacement_m=.25)
        self.bounds.start();self.addCleanup(self.bounds.stop)
        self.q=np.array(ready_pose('plate')['joints_rad']);self.w=[.1,.1]
    def test_default_allows_task_angle_beyond_debug_radius(self):
        q=self.q.copy();q[1,3]-=np.deg2rad(50)
        geometry=c.Geometry(self.q,self.w);geometry.check(q,self.w)
        c.BOUNDS['relative_limits_enabled']=True
        with self.assertRaisesRegex(RuntimeError,'target offset'):geometry.check(q,self.w)
    def test_both_distance_caps_follow_switch(self):
        q=self.q.copy();q[:,0]+=.05
        geometry=c.Geometry(self.q,self.w)
        c.BOUNDS.update(tcp_displacement_m=.001,frame_displacement_m=.001)
        self.assertGreater(geometry.check(q,self.w),.001)
        c.BOUNDS['relative_limits_enabled']=True
        with self.assertRaisesRegex(RuntimeError,'TCP displacement'):geometry.check(q,self.w)
        c.BOUNDS['tcp_displacement_m']=1.
        with self.assertRaisesRegex(RuntimeError,'link-frame displacement .* exceeds 0.1 cm'):geometry.check(q,self.w)
    def test_absolute_joint_limits_and_invalid_numbers_remain(self):
        geometry=c.Geometry(self.q,self.w)
        for value in (UPPER[0]+.001,float('nan')):
            q=self.q.copy();q[0,0]=value
            with self.assertRaises(ValueError):geometry.check(q,self.w)
    def test_absolute_workspace_remains(self):
        geometry=c.Geometry(self.q,self.w);geometry.hi=geometry.origin.min(axis=0)-.001
        with self.assertRaisesRegex(RuntimeError,'workspace AABB'):geometry.check(self.q,self.w)
    def test_step_limit_remains(self):
        geometry=c.Geometry(self.q,self.w);q=self.q.copy();q[:,0]+=.2
        with self.assertRaisesRegex(RuntimeError,'TCP command step'):geometry.check(q,self.w,commit=True)
    def test_switch_is_audited_and_protocol_grouping_matches_effective_limits(self):
        off=Settings();on=Settings(relative_limits_enabled=True)
        self.assertFalse(off.relative_limits_enabled)
        self.assertNotEqual(digest(off.protocol()),digest(on.protocol()))
        self.assertEqual(digest(off.protocol()),digest(Settings(excursion_deg=90,tcp_cm=40).protocol()))
        self.assertNotEqual(digest(on.protocol()),digest(Settings(relative_limits_enabled=True,tcp_cm=30).protocol()))
        self.assertEqual(on.protocol()['link_displacement_cm'],25.)
        self.assertIsNone(off.protocol()['excursion_deg'])
        for bad in (None,0,1,'false'):
            with self.assertRaises(ValueError):Settings(relative_limits_enabled=bad).checked()

if __name__=='__main__':unittest.main()
