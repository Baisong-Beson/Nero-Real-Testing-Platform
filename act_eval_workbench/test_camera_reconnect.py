import contextlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, Mock
from . import camera_reconnect as c


def state(stamp=None):
    return dict(updated_unix_s=time.time(),control_publishers=dict.fromkeys(map(str,range(14)),0),
                main_camera=dict(frames_decoded=5,source_unix_s=time.time() if stamp is None else stamp,
                                 source_age_s=.02,received_age_s=.01,decode_error=False))


class Clock:
    def __init__(self):self.value=0.
    def now(self):return self.value
    def wait(self,seconds):self.value+=seconds
    def is_set(self):return False


class Tests(unittest.TestCase):
    def test_fresh_requires_decoded_recent_image(self):
        self.assertTrue(c.fresh(state()))
        for key,value in [('frames_decoded',0),('decode_error',True),('source_age_s',3),('received_age_s',3),('source_age_s',float('nan')),('source_age_s',-5)]:
            s=state();s['main_camera'][key]=value;self.assertFalse(c.fresh(s))
        s=state();s['updated_unix_s']-=10;self.assertFalse(c.fresh(s))
        self.assertFalse(c.fresh({'camera_age_s':.01}))

    def test_recovery_requires_three_advancing_timestamps(self):
        clock=Clock();stamps=iter([1,1,2,2,3]);seen=[]
        def feed():v=next(stamps);seen.append(v);return state(v)
        with patch.object(c.time,'monotonic',clock.now):result=c.wait_frames(feed,clock,1)
        self.assertEqual(result['source_unix_s'],3);self.assertEqual(seen,[1,1,2,2,3])

    def test_repeated_old_image_cannot_pass(self):
        clock=Clock()
        with patch.object(c.time,'monotonic',clock.now):self.assertIsNone(c.wait_frames(lambda:state(1),clock,.5))

    def test_timestamp_regression_restarts_count(self):
        clock=Clock();stamps=iter([2,3,1,2,3,4]);seen=[]
        def feed():v=next(stamps);seen.append(v);return state(v)
        with patch.object(c.time,'monotonic',clock.now):result=c.wait_frames(feed,clock,2)
        self.assertEqual(seen,[2,3,1,2,3,4]);self.assertEqual(result['source_unix_s'],4)

    def test_healthy_camera_is_never_restarted(self):
        with patch.object(c,'isolated',return_value=False),patch.object(c,'usb_snapshot',side_effect=AssertionError('should not scan')):
            self.assertTrue(c.recover(Path('/unused'),state,Clock(),print)['ok'])

    def test_isolated_test_never_scans_real_devices(self):
        with patch.object(c,'isolated',return_value=True),patch.object(c,'usb_snapshot',side_effect=AssertionError('hardware')):
            self.assertTrue(c.recover(Path('/unused'),lambda:{},Clock(),print)['skipped'])

    def test_missing_video_does_not_claim_hid_is_video_or_start_driver(self):
        with patch.object(c,'isolated',return_value=False),patch.object(c,'wait_frames',return_value=None),patch.object(c,'usb_snapshot',return_value={'devices':[{'product':'ZED-M Hid Device'}],'video_devices':[]}),patch.object(c.subprocess,'Popen') as spawn:
            result=c.recover(Path('/unused'),lambda:{},Clock(),print)
        self.assertFalse(result['ok']);self.assertTrue(result['waiting_usb']);spawn.assert_not_called()

    def test_zed_mini_video_without_serial_matches_hid(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);usb=root/'usb';video=root/'video';usb.mkdir();video.mkdir()
            for name,pid,serial in [('3-8.3.2.3','f681',c.SERIAL),('4-8.3.2.3','f682','')]:
                p=usb/name;p.mkdir()
                for key,value in dict(idVendor='2b03',idProduct=pid,product='ZED-M',devnum='39',speed='5000').items():(p/key).write_text(value)
                if serial:(p/'serial').write_text(serial)
            (usb/'4-8.3.2.3/video0').mkdir();(video/'video0').symlink_to(usb/'4-8.3.2.3/video0')
            snapshot=c.usb_snapshot(usb,video)
            self.assertEqual(snapshot['video_devices'],['/dev/video0'])
            self.assertEqual(len(snapshot['devices']),2)

    def test_watch_retries_once_per_usb_connection_and_after_healthy_stream(self):
        w=c.ReconnectWatch();usb={'video_devices':['/dev/video0'],'devices':[{'devnum':'5'}]}
        self.assertFalse(w.due({},usb,0));self.assertTrue(w.due({},usb,4));self.assertFalse(w.due({},usb,60))
        usb['devices'][0]['devnum']='6';self.assertTrue(w.due({},usb,62))
        self.assertFalse(w.due(state(),usb,64));self.assertFalse(w.due({},usb,66));self.assertTrue(w.due({},usb,70))

    def test_watch_waits_for_video_endpoint(self):
        w=c.ReconnectWatch();usb={'video_devices':[],'devices':[{'product':'HID'}]}
        self.assertFalse(w.due({},usb,0));self.assertFalse(w.due({},usb,50))

    def test_only_dedicated_camera_launch_config_accepted(self):
        recipe=self.recipe();c.validate_recipe(recipe)
        recipe['command'][3]='robot_full_stack'
        with self.assertRaises(RuntimeError):c.validate_recipe(recipe)
        recipe=self.recipe();recipe['serial']='another-camera'
        with self.assertRaises(RuntimeError):c.validate_recipe(recipe)

    def recipe(self):
        return dict(serial=c.SERIAL,environment={},command=['/usr/bin/python3','/opt/ros/humble/bin/ros2','launch','zed_wrapper','zed_camera.launch.py','camera_model:=zedm','camera_name:=zed_m','node_name:=zed_node','serial_number:='+c.SERIAL])

    def test_missing_driver_starts_and_requires_real_frames(self):
        for good in [True,False]:
            with tempfile.TemporaryDirectory() as directory:
                root=Path(directory);(root/'config').mkdir();(root/'logs').mkdir()
                (root/'config/main_camera_reconnect.json').write_text(json.dumps(self.recipe()))
                s=state();s.pop('main_camera')
                with contextlib.ExitStack() as stack:
                    for name,value in [('isolated',False),('usb_snapshot',{'video_devices':['/dev/video0']}),('camera_processes',[])]:
                        stack.enter_context(patch.object(c,name,return_value=value))
                    stack.enter_context(patch.object(c,'wait_frames',side_effect=[None,state()['main_camera'] if good else None]))
                    stack.enter_context(patch.object(c.fcntl,'flock'))
                    spawn=stack.enter_context(patch.object(c.subprocess,'Popen',return_value=Mock(pid=123)))
                    result=c.recover(root,lambda:s,Clock(),lambda _:None)
                self.assertEqual(result['ok'],good);self.assertTrue(result['restarted']);self.assertEqual(spawn.call_count,1)

    def test_duplicate_drivers_are_not_killed(self):
        records=[dict(kind='launcher',pid=1),dict(kind='launcher',pid=2)]
        with patch.object(c.os,'kill') as kill:
            with self.assertRaises(RuntimeError):c.stop_camera(records,state,Clock(),print)
            kill.assert_not_called()

    def test_process_identity_change_prevents_signal(self):
        with patch.object(c,'same_process',return_value=False),patch.object(c.os,'kill') as kill:
            c.stop_camera([dict(kind='launcher',pid=1)],state,Clock(),print);kill.assert_not_called()


if __name__=='__main__':unittest.main()
