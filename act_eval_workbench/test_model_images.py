import unittest
from types import SimpleNamespace as NS
import numpy as np
from .model_images import wrist_images

class Tests(unittest.TestCase):
    def frame(self):
        return NS(header=NS(stamp=NS(sec=100,nanosec=0)),height=1,width=2,step=8,encoding='bgr8',data=bytes([1,2,3,4,5,6,99,99]))
    def test_color_and_row_padding(self):
        out=wrist_images(['main','left_wrist'],{'left_wrist':(self.frame(),10)},10.1,100.1,100.)
        np.testing.assert_array_equal(out['observation/left_wrist_image'],np.array([[[3,2,1],[6,5,4]]],np.uint8))
    def test_missing_stale_and_short_data_rejected(self):
        with self.assertRaises(RuntimeError):wrist_images(['left_wrist'],{},10.1,100.1,100.)
        with self.assertRaises(RuntimeError):wrist_images(['left_wrist'],{'left_wrist':(self.frame(),10)},11,101,101)
        f=self.frame();f.data=b'abc'
        with self.assertRaises(ValueError):wrist_images(['left_wrist'],{'left_wrist':(f,10)},10.1,100.1,100.)

if __name__=='__main__':unittest.main()
