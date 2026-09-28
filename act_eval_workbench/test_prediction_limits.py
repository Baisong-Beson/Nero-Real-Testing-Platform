"""Unused forecast steps must not abort the four-step receding horizon."""
import unittest
import numpy as np
from . import controller as c
from .common import DEFAULT_DEG, UPPER, Settings


class Tests(unittest.TestCase):
    def actions(self):
        q=np.deg2rad(DEFAULT_DEG)
        return np.tile(np.r_[q[0],0.,q[1],0.],(16,1))

    def test_unused_future_violation_does_not_reject_execution_prefix(self):
        for index in range(4,16):
            a=self.actions();a[index,9]=UPPER[1]+.03
            q,w=c.unpack(a,limit_steps=4)
            c.pos.limits(q[:4]);self.assertEqual(w.shape,(16,2))
            self.assertEqual(c.joint_limit_violations(q)[0]['step'],index+1)
            with self.assertRaises(c.PolicyJointLimitError):c.unpack(a)

    def test_each_executed_step_remains_guarded_with_specific_diagnosis(self):
        for index in range(4):
            a=self.actions();a[index,9]=UPPER[1]+.001
            with self.assertRaises(c.PolicyJointLimitError) as raised:c.unpack(a,limit_steps=4)
            row=raised.exception.violations[0]
            self.assertEqual((row['step'],row['arm'],row['joint']),(index+1,'left',2))
            self.assertAlmostEqual(row['upper_deg'],98.54874076)
            self.assertIn('左臂 joint2',str(raised.exception))

    def test_future_value_entering_next_execution_window_is_rejected(self):
        a=self.actions();a[4,0]=UPPER[0]+.001;c.unpack(a,limit_steps=4)
        replanned=np.concatenate([a[4:],a[:4]])
        with self.assertRaises(c.PolicyJointLimitError):c.unpack(replanned,limit_steps=4)

    def test_invalid_shapes_and_nonfinite_future_values_still_rejected(self):
        for value in (float('nan'),float('inf')):
            a=self.actions();a[-1,-1]=value
            with self.assertRaises(ValueError):c.unpack(a,limit_steps=4)
        with self.assertRaises(ValueError):c.unpack(self.actions()[:4],limit_steps=4)
        for steps in (0,17,True,3.5):
            with self.assertRaises(ValueError):c.unpack(self.actions(),limit_steps=steps)

    def test_safety_margin_matches_both_control_adapters(self):
        from .common import LOWER, UPPER
        np.testing.assert_array_equal(LOWER,c.pos.LOWER)
        np.testing.assert_array_equal(UPPER,c.pos.UPPER)

    def test_protocol_distinguishes_new_scope(self):
        protocol=Settings().protocol()
        self.assertEqual(protocol['version'],c.PROTOCOL)
        self.assertEqual(protocol['raw_joint_limit_scope'],'executed_prefix_only_unused_tail_warning')


if __name__=='__main__':unittest.main()
