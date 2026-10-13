import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from .camera_preview import decode_raw, preview_status, write_preview, PREVIEW_FILES


class Tests(unittest.TestCase):
    def test_rgb_padding_and_color_conversion(self):
        msg=SimpleNamespace(width=2,height=2,step=8,encoding='rgb8',
            data=bytes([255,0,0,0,255,0,99,99,0,0,255,255,255,255,99,99]))
        frame=decode_raw(msg)
        self.assertEqual(frame.tolist(),[[[0,0,255],[0,255,0]],[[255,0,0],[255,255,255]]])

    def test_invalid_payload_and_unsupported_format(self):
        msg=SimpleNamespace(width=2,height=2,step=6,encoding='rgb8',data=b'bad')
        with self.assertRaises(ValueError):decode_raw(msg)
        msg.encoding='unrecognized'
        with self.assertRaises(ValueError):decode_raw(msg)

    def test_independent_preview_files_keep_aspect_ratio(self):
        with tempfile.TemporaryDirectory() as name:
            folder=Path(name)
            for view,color in [('main',(0,0,240)),('left',(0,240,0)),('right',(240,0,0))]:
                write_preview(folder,view,np.full((480,640,3),color,np.uint8))
                frame=cv2.imread(str(folder/PREVIEW_FILES[view]))
                self.assertEqual(frame.shape,(432,576,3))
                self.assertTrue(np.allclose(frame[0,0],color,atol=3))

    def test_freshness_and_disconnect_are_per_camera(self):
        now=time.time()
        row=dict(frames_decoded=5,decode_error=False,source_age_s=.1,received_age_s=.1)
        state=dict(updated_unix_s=now,previews={'main':dict(row),'right':dict(row)},
            wrist_cameras={'right':dict(health='ok',label='通畅'),'left':dict(health='error',label='未连接')})
        self.assertTrue(preview_status(state,'right',now)[0])
        self.assertFalse(preview_status(state,'left',now)[0])
        state['previews']['right']['decode_error']=True
        self.assertFalse(preview_status(state,'right',now)[0])
        self.assertTrue(preview_status(state,'main',now)[0])
        state['previews']['right']=dict(row,source_age_s=2)
        self.assertFalse(preview_status(state,'right',now)[0])
        self.assertFalse(preview_status(state,'main',now+2)[0])


if __name__=='__main__':unittest.main()
