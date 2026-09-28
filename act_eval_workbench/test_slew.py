"""Motion-filter regressions: progress, braking, feedback lag and reversals."""
import unittest
import numpy as np
from . import controller as c
from .common import DEFAULT_DEG, LOWER, UPPER, Settings


class Tests(unittest.TestCase):
    def trajectory(self, distance, widths, feedback_gain=1., count=600):
        actual = np.deg2rad(DEFAULT_DEG)
        target = actual.copy(); target[:, 0] += distance
        slew = c.Slew(actual, [.1, .1]); rows = []
        previous = slew.x.copy(); velocity = slew.v.copy()
        for i in range(count):
            q, w = slew.next(target, widths, c.DT, actual)
            np.testing.assert_allclose((slew.x-previous)/c.DT, slew.v, atol=1e-12)
            self.assertTrue(np.all(np.abs(slew.v) <= slew.vmax+1e-10))
            self.assertTrue(np.all(np.abs(slew.v-velocity) <= slew.amax*c.DT+1e-10))
            self.assertLessEqual(np.max(np.abs(q-actual)), c.BOUNDS['joint_command_lead_rad'])
            actual += feedback_gain*(q-actual)
            rows.append((i*c.DT, slew.x.copy(), slew.v.copy()))
            previous, velocity = slew.x.copy(), slew.v.copy()
        return target, actual, w, rows

    def test_removes_hidden_point_zero_six_speed_ceiling(self):
        target, actual, widths, rows = self.trajectory(.3, [.03, .04])
        self.assertGreater(max(abs(row[2][0]) for row in rows), .09)
        arrival = next(t for t,x,v in rows if abs(x[0]-target[0,0]) < .001)
        self.assertLess(arrival, 4.)
        np.testing.assert_allclose(actual, target, atol=1e-8)
        np.testing.assert_allclose(widths, [.03, .04], atol=1e-8)
        self.assertLessEqual(max(row[1][0] for row in rows), target[0,0]+1e-10)

    def test_lagging_feedback_does_not_accumulate_large_command(self):
        target, actual, _, rows = self.trajectory(.3, [0., .1], feedback_gain=.35)
        np.testing.assert_allclose(actual, target, atol=1e-8)
        self.assertGreater(max(row[2][0] for row in rows), .055)

    def test_stuck_feedback_stops_at_lead_boundary(self):
        _, actual, _, rows = self.trajectory(.3, [.1, .1], feedback_gain=0.)
        self.assertLessEqual(max(row[1][0] for row in rows)-actual[0,0], .02+1e-10)
        self.assertLess(abs(rows[-1][2][0]), 1e-8)

    def test_tiny_moves_and_gripper_endpoints_without_slow_tail(self):
        target, actual, widths, rows = self.trajectory(.001, [0., .1], count=90)
        np.testing.assert_allclose(actual, target, atol=1e-8)
        np.testing.assert_allclose(widths, [0., .1], atol=1e-8)
        self.assertTrue(all(0 <= row[1][14] <= .1 for row in rows))
        arrival = next(t for t,x,v in rows if abs(x[0]-target[0,0]) < 1e-6)
        self.assertLess(arrival, .3)

    def test_reversals_and_variable_intervals_preserve_physical_bounds(self):
        actual = np.deg2rad(DEFAULT_DEG); actual[:,0] = UPPER[0]-.002
        slew = c.Slew(actual, [.001, .099]); rng = np.random.default_rng(21)
        for i in range(1500):
            target = actual.copy()
            target[:,0] = UPPER[0] if (i//13)%2 else UPPER[0]-.04
            widths = [0., .1] if (i//7)%2 else [.1, 0.]
            elapsed = float(rng.uniform(.05, .15)); old_v = slew.v.copy()
            q, w = slew.next(target, widths, elapsed, actual)
            self.assertTrue(np.all(np.abs(slew.v-old_v) <= slew.amax*elapsed+1e-9))
            self.assertTrue(np.all(q >= LOWER) and np.all(q <= UPPER))
            self.assertTrue(np.all(w >= 0.) and np.all(w <= .1))
            actual = q

    def test_invalid_targets_and_intervals_still_rejected(self):
        q = np.deg2rad(DEFAULT_DEG)
        for dt in (.049, .151, float('nan')):
            with self.assertRaises(RuntimeError): c.Slew(q,[.1,.1]).next(q,[.1,.1],dt,q)
        with self.assertRaises(ValueError): c.Slew(q,[.1,.1]).next(q,[float('nan'),.1],.05,q)

    def test_experiment_protocol_identifies_filter_change(self):
        protocol = Settings().protocol()
        self.assertEqual(protocol['version'], c.PROTOCOL)
        self.assertEqual(protocol['target_filter'], 'firmware_move_j_absolute_targets')


if __name__ == '__main__': unittest.main()
