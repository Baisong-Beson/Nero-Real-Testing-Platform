"""Exercise camera widgets without launching a robot or a camera driver."""
import os
from pathlib import Path
import queue
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import tkinter as tk

from .app import App
from .camera_preview import VIEWS, write_preview
from .camera_devices import binding


@unittest.skipUnless(os.environ.get('DISPLAY'), 'X11 display required')
class Tests(unittest.TestCase):
    def test_selection_disconnection_assignment_and_minimum_layout(self):
        with tempfile.TemporaryDirectory() as temp:
            directory=Path(temp)
            healthy=dict(frames_decoded=4,source_age_s=.02,received_age_s=.01,decode_error=False)
            state=dict(updated_unix_s=time.time(),previews={r:dict(healthy) for r in VIEWS},
                       wrist_cameras={s:dict(health='ok',label='通畅',fps=15.) for s in ('left','right')})
            devices=[dict(id='realsense:'+s,serial=s,kind='realsense',name='Test camera') for s in ('11111111','22222222')]
            calls=[]
            manager=SimpleNamespace(devices=devices,error='',pending=set(),updating=False,refresh=threading.Event(),
                config=dict(roles={'right':binding(devices[0],'right'),'left':binding(devices[1],'left'),'main':None},auto_rebind=True))
            def apply(roles,auto):
                from .camera_devices import validate_roles
                validate_roles(roles);calls.append((roles,auto));manager.config=dict(roles=roles,auto_rebind=auto)
            manager.apply=apply
            backend=SimpleNamespace(directory=directory,events=queue.Queue(),busy=False,active=None,stop_jobs=[],
                hardware_busy=False,camera_manager=manager,tick=lambda:None,telemetry_state=lambda:state)
            root=tk.Tk()
            # Render the complete production layout, with no model/task changes.
            try:
                with patch.object(App,'selection_changed'),patch.object(App,'refresh_records'),patch.object(App,'tick'):
                    app=App(root,backend)
                root.geometry('1180x820');root.update()
                self.assertEqual(tuple(app.camera_view_box['values']),tuple(VIEWS.values()))
                for role,color in [('main',(0,0,240)),('left',(0,240,0)),('right',(240,0,0))]:
                    write_preview(directory,role,np.full((480,640,3),color,np.uint8))
                for role in ('right','left','main'):
                    state['updated_unix_s']=time.time()
                    app.camera_view.set(VIEWS[role]);app.camera_view_box.event_generate('<<ComboboxSelected>>');root.update()
                    self.assertEqual(app.image_stamp[0],role)
                    self.assertIsNotNone(app.photo)
                app.camera_view.set(VIEWS['right']);app.camera_view_changed();root.update()
                state['wrist_cameras']['right'].update(health='error',label='未连接')
                app.refresh_camera_preview(state);root.update()
                self.assertIsNone(app.photo);self.assertIn('未连接',app.camera['text'])
                state['wrist_cameras']['right'].update(health='ok',label='通畅')
                state['updated_unix_s']=time.time()-2
                app.refresh_camera_preview(state);self.assertIsNone(app.photo)
                self.assertIn('监视未连接',app.camera['text'])
                state['updated_unix_s']=time.time();app.refresh_camera_preview(state)
                app.open_camera_settings();root.update()
                dialog=app.camera_settings_dialog
                self.assertEqual(len(dialog.boxes),3)
                dialog.vars['right'].set('未分配')
                dialog.vars['main'].set(dialog.by_id[devices[0]['id']]);dialog.apply()
                deadline=time.monotonic()+3
                while dialog.busy and time.monotonic()<deadline:root.update();time.sleep(.02)
                self.assertEqual(calls[-1][0]['main']['serial'],'11111111')
                self.assertIsNone(calls[-1][0]['right'])
                self.assertTrue(app.stop_button.winfo_viewable())
                self.assertLessEqual(app.stop_button.winfo_rooty()+app.stop_button.winfo_height(),root.winfo_rooty()+root.winfo_height())
                output=os.environ.get('NERO_UI_TEST_OUTPUT')
                if output:
                    from .capture_window import capture
                    path=Path(output);path.mkdir(parents=True,exist_ok=True)
                    capture(dialog.window,path/'camera_settings.png')
                    dialog.window.destroy();root.update();capture(root,path/'camera_preview.png')
            finally:root.destroy()


if __name__=='__main__':unittest.main()
