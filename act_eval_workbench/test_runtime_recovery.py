"""Regression for observed 266.6 ms video gap and transient policy loop delay."""
import tempfile
from pathlib import Path
import unittest
import cv2
import numpy as np
from . import controller as c
from .recording import VideoRecorder

class Tests(unittest.TestCase):
    def test_rebase_uses_live_feedback_and_only_one_nominal_step(self):
        for elapsed in (.1641,.20,.30,.35):
            actual=c.pos.READY.copy();old=actual.copy();old[:,0]+=.02
            slew=c.Slew(old,[.08,.08]);slew.v[:]=.1
            target=actual.copy();target[:,0]+=.1
            new,q,w,rebased=c.advance_after_delay(slew,target,[.04,.04],actual,[.1,.1],elapsed)
            self.assertTrue(rebased);self.assertIsNot(new,slew)
            self.assertLessEqual(np.max(np.abs(q-actual)),c.BOUNDS['joint_acceleration_rad_s2']*c.DT**2+1e-12)
            self.assertLessEqual(np.max(np.abs(w-.1)),c.BOUNDS['gripper_acceleration_m_s2']*c.DT**2+1e-12)
    def test_long_or_invalid_interval_stops(self):
        q=c.pos.READY.copy();slew=c.Slew(q,[.1,.1])
        for delay in (.351,.5,.049,float('nan')):
            with self.assertRaisesRegex(RuntimeError,'control interval'):c.advance_after_delay(slew,q,[.1,.1],q,[.1,.1],delay)
    def test_normal_interval_preserves_velocity_history(self):
        q=c.pos.READY.copy();slew=c.Slew(q,[.1,.1]);slew.v[0]=.01
        new,_,_,rebased=c.advance_after_delay(slew,q,[.1,.1],q,[.1,.1],.05)
        self.assertIs(new,slew);self.assertFalse(rebased)
    def record(self,stamps):
        with tempfile.TemporaryDirectory() as path:
            frame=np.zeros((48,64,3),np.uint8);frame[:,:,1]=120
            ok,jpeg=cv2.imencode('.jpg',frame);self.assertTrue(ok)
            recorder=VideoRecorder(Path(path),frame.shape)
            for stamp in stamps:recorder.submit(dict(stamp=stamp,received=stamp,data=jpeg.tobytes()))
            result=recorder.close()
            cap=cv2.VideoCapture(str(Path(path)/'main_camera.avi'));count=0
            while cap.read()[0]:count+=1
            cap.release();self.assertEqual(count,result['frames'])
            return result
    def test_observed_video_gap_records_warning_and_keeps_frames(self):
        result=self.record([1.,1.0666,1.333206092453,1.4])
        self.assertIsNone(result['error']);self.assertEqual(result['frames'],4)
        self.assertEqual(result['gap_warning_count'],1);self.assertEqual(result['quality'],'minor_gaps')
    def test_long_video_gap_and_regression_still_fail(self):
        for stamps in ([1.,1.501],[1.,.99]):
            result=self.record(stamps);self.assertIn('video source gap',result['error'])
            self.assertEqual(result['frames'],1);self.assertEqual(result['quality'],'error')

if __name__=='__main__':unittest.main()
