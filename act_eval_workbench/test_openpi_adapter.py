import unittest
import numpy as np
from .openpi_nero_adapter import native_observation,platform_actions

class Tests(unittest.TestCase):
    def test_camera_slots_and_gripper_calibration(self):
        state=np.zeros(16,np.float32);state[[7,15]]=[.4,.6]
        obs={'observation/state':state,'observation/exterior_image_1_left':np.zeros((2,2,3),np.uint8),'observation/left_wrist_image':np.full((2,2,3),11,np.uint8),'observation/right_wrist_image':np.full((2,2,3),22,np.uint8),'prompt':'marker into drawer'}
        native=native_observation(obs,[.0996,.1005])
        self.assertEqual(native['observation/wrist_image_left'][0,0,0],22)
        self.assertEqual(native['observation/wrist_image_right'][0,0,0],11)
        self.assertEqual(native['prompt'],obs['prompt'])
        converted=platform_actions(np.tile(native['observation/state'],(16,1)),[.0996,.1005])
        np.testing.assert_allclose(converted,np.tile(state,(16,1)),atol=1e-7)
        np.testing.assert_array_equal(state[[7,15]],np.array([.4,.6],np.float32))
    def test_preserve_raw_gripper_predictions_for_audit(self):
        a=np.zeros((16,16),np.float32);a[:,7]=1.1;a[:,15]=-.1
        result=platform_actions(a,[.0996,.1005]);self.assertGreater(result[0,7],1);self.assertLess(result[0,15],0)

if __name__=='__main__':unittest.main()
