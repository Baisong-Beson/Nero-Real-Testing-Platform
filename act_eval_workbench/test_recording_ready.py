import unittest
from types import SimpleNamespace as NS
import numpy as np
from .worker import wait_recording_ready

class Tests(unittest.TestCase):
    def fixture(self,finish_at=.08):
        t=[0.];rec=NS(frames=0,error=None);q=np.zeros((2,7));op=NS(permit=lambda:True)
        def idle():
            t[0]+=.01
            if finish_at is not None and t[0]>=finish_at:rec.frames=1
        io=NS(now=lambda:t[0],pump=lambda:None,check_publishers=lambda:None,snapshot=lambda stationary:(q,[.1,.1]),
              idle=idle,interrupted=False)
        return io,rec,op,q,t
    def test_async_first_frame_waits_without_fixed_delay(self):
        io,rec,op,q,t=self.fixture();wait_recording_ready(io,rec,op,1,q,.01)
        self.assertLess(t[0],.1)
        before=t[0];wait_recording_ready(io,rec,op,1,q,.01);self.assertEqual(t[0],before)
    def test_cancel_error_and_stuck_writer_prevent_motion(self):
        for kind in ('cancel','error','stuck'):
            with self.subTest(kind=kind):
                io,rec,op,q,t=self.fixture(None)
                if kind=='cancel':op.permit=lambda:False
                if kind=='error':rec.error='disk unavailable'
                with self.assertRaises(RuntimeError):wait_recording_ready(io,rec,op,1,q,.01)
    def test_moved_start_is_rejected_while_waiting(self):
        io,rec,op,q,t=self.fixture();goal=q.copy();q[0,0]=.1
        with self.assertRaisesRegex(RuntimeError,'起点发生变化'):wait_recording_ready(io,rec,op,1,goal,.01)

if __name__=='__main__':unittest.main()
