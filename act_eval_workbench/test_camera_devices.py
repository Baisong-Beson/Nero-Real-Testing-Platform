import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch, Mock

from .camera_devices import (CameraManager, automatic_replacements, binding,
                            config_path, load_config, save_config, validate_roles,
                            discover_devices, run_driver)


def device(serial):
    return dict(id='realsense:'+serial,serial=serial,kind='realsense',name='Test camera')


class Tests(unittest.TestCase):
    def test_inventory_reads_serial_without_choosing_a_role(self):
        result=Mock(returncode=0,stdout='Device Name   Serial Number   Firmware Version\nIntel RealSense D405    33333333    5.17.0.10\n',stderr='')
        with patch(__package__+'.camera_devices.subprocess.run',return_value=result),patch.object(Path,'glob',return_value=[]):
            found=discover_devices()
        self.assertEqual(found,[device('33333333')|{'name':'Intel RealSense D405'}])

    def test_driver_reads_saved_serial_and_correct_topic_role(self):
        with tempfile.TemporaryDirectory() as folder:
            save_config(folder,dict(roles={'main':binding(device('33333333'),'main')},
                driver=dict(setup='/test/setup.bash',executable='/test/camera-driver')))
            with patch(__package__+'.camera_devices.os.execv') as execute:
                run_driver(str(config_path(folder)),'main')
            args=execute.call_args.args[1]
            self.assertIn('serial_no:=_33333333',args)
            self.assertIn('__node:=main_view',args)

    def test_single_replacement_updates_identity_and_role_topic(self):
        roles={'right':binding(device('11111111'),'right')}
        result=automatic_replacements(roles,[device('22222222')],{'right'})
        self.assertEqual(result['right']['serial'],'22222222')
        self.assertEqual(result['right']['topic'],'/right_wrist/color/image_raw')

    def test_no_guess_for_multiple_devices_or_missing_roles(self):
        roles={'left':binding(device('11111111'),'left'),'right':binding(device('22222222'),'right')}
        self.assertEqual(automatic_replacements(roles,[device('33333333')],{'left','right'}),{})
        self.assertEqual(automatic_replacements(roles,[device('33333333'),device('44444444')],{'right'}),{})
        self.assertEqual(automatic_replacements(roles,[device('11111111')],{'right'}),{})

    def test_roles_cannot_share_camera(self):
        with self.assertRaises(ValueError):validate_roles({'left':device('11111111'),'right':device('11111111')})

    def test_failed_driver_update_restores_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            original=dict(roles={'right':binding(device('11111111'),'right')},auto_rebind=True,driver={})
            save_config(folder,original)
            manager=CameraManager.__new__(CameraManager)
            manager.platform=Path(folder);manager.idle=lambda:True;manager.lock=threading.Lock()
            manager.notify=lambda text:None;manager.devices=[];manager.pending=set();manager.config=original
            with patch(__package__+'.camera_devices.configure_drivers',side_effect=[RuntimeError('failed'),None]):
                with self.assertRaises(RuntimeError):manager.apply({'right':binding(device('22222222'),'right')},True)
            self.assertEqual(load_config(folder),original)
            self.assertFalse(manager.updating)

    def test_active_task_cannot_change_binding(self):
        with tempfile.TemporaryDirectory() as folder:
            manager=CameraManager.__new__(CameraManager)
            manager.platform=Path(folder);manager.idle=lambda:False;manager.lock=threading.Lock()
            with patch(__package__+'.camera_devices.configure_drivers') as driver:
                with self.assertRaises(RuntimeError):manager.apply({'right':binding(device('22222222'),'right')},True)
                driver.assert_not_called()
            self.assertFalse(config_path(folder).exists())


if __name__=='__main__':unittest.main()
