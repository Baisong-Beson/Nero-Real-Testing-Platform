import unittest
from .camera_health import FrameHealth,indicator,COLORS

class Tests(unittest.TestCase):
    def fresh(self):
        h=FrameHealth('/test')
        for i in range(16):h.observe(100+i/15,10+i/15,640,480,1920,921600,'rgb8')
        return h
    def test_green_requires_fresh_valid_advancing_frames(self):
        h=self.fresh();r=h.snapshot(101.01,11.01,1)
        self.assertEqual(r['health'],'ok');self.assertAlmostEqual(r['fps'],15)
        self.assertEqual(r['frames'],16)
    def test_publisher_alone_is_not_green_and_missing_is_red(self):
        h=FrameHealth('/test')
        self.assertEqual(h.snapshot(1,1,1)['label'],'等待图像')
        self.assertEqual(h.snapshot(1,1,0)['label'],'未连接')
        self.assertEqual(self.fresh().snapshot(101.01,11.01,2)['label'],'发布源冲突')
    def test_delay_stall_duplicate_timestamp_and_recovery(self):
        h=self.fresh();h.observe(101,11.6,640,480,1920,921600,'rgb8')
        self.assertEqual(h.snapshot(101.6,11.6,1)['label'],'图像延迟')
        self.assertEqual(h.snapshot(102.6,12.6,1)['label'],'图像中断')
        self.assertEqual(h.frames,16)
        for i in range(16):h.observe(103+i/15,13+i/15,640,480,1920,921600,'rgb8')
        self.assertEqual(h.snapshot(104.01,14.01,1)['health'],'ok')
    def test_invalid_packet_or_regressed_stamp_is_not_healthy(self):
        h=self.fresh();h.observe(101.1,11.1,640,480,1920,10,'rgb8')
        self.assertEqual(h.snapshot(101.1,11.1,1)['label'],'图像数据异常')
        h.observe(100.9,11.1,640,480,1920,921600,'rgb8')
        self.assertEqual(h.snapshot(101.1,11.1,1)['label'],'时间戳回退')
    def test_source_timestamp_must_be_fresh_even_when_receiving(self):
        h=self.fresh()
        self.assertEqual(h.snapshot(103,11.01,1)['health'],'error')
        self.assertEqual(h.snapshot(100,11.01,1)['label'],'时钟异常')
    def test_indicator_drops_green_when_telemetry_stops(self):
        r=self.fresh().snapshot(101.01,11.01,1)
        self.assertEqual(indicator(r,.1)[0],COLORS['ok'])
        self.assertEqual(indicator(r,1)[0],COLORS['unknown'])
        self.assertEqual(indicator(None,.1)[0],COLORS['unknown'])

if __name__=='__main__':unittest.main()
