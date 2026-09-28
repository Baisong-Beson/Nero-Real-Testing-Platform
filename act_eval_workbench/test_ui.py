"""Exercise actual Tk widgets with a backend that cannot touch ROS/hardware."""
import tempfile
import queue
import time
from pathlib import Path
import tkinter as tk
from unittest.mock import patch
import numpy as np
from .common import *
from .app import App
from .capture_window import capture
from . import records as record_store

class FakeBackend:
    def __init__(self,path):
        self.directory=path;self.events=queue.Queue();self.busy=False;self.active=None;self.stop_jobs=[]
        self.calls=[];self.count=0;self.camera_health={};self.telemetry_age=0.;self.heartbeats=0
    def tick(self):self.heartbeats+=1
    def telemetry_state(self):
        q=ready_pose('plate')['joints_deg']
        return dict(updated_unix_s=time.time()-self.telemetry_age,arms={a:dict(age_s=.01,joints_deg=q[i],width_m=.1) for i,a in enumerate(('right','left'))},camera_age_s=.04,control_publishers={},wrist_cameras=self.camera_health)
    def start(self,kind,settings,directory=None):
        self.count+=1;d=Path(directory) if directory else self.directory/f'{self.count}_{kind}';d.mkdir(exist_ok=True)
        self.calls.append(kind);self.active=d
        if kind in ('ready','default','formal','disable'):
            r=dict(kind=kind,description='模拟驱动计划：不会控制真实机器人',settings=settings.json(),prepared_unix_s=time.time(),
                start_joints_rad=np.deg2rad(DEFAULT_DEG).tolist(),success_definition=SUCCESS[settings.task])
            if kind=='ready' and settings.task=='banana':
                r.update(gripper_policy='task_target_after_joint_arrival',trajectory=dict(
                    goal=ready_pose('banana')['joints_rad'],gripper_goal_m=ready_gripper_target('banana')))
            r['plan_sha256']=digest(r);write(d/'plan.json',r)
        elif kind=='execute':
            assert read(d/'approval.json')['source']=='local_desktop_confirm'
            plan=read(d/'plan.json')
            r=dict(kind=plan['kind'],started=True,runtime_completed=True,settings=settings.json(),protocol_sha256=digest(settings.protocol()),adjudication={'success':None})
            write(d/'result.json',r)
        elif kind=='pose':r=ready_pose(settings.task)
        elif kind=='offline':r=dict(trajectory=[dict(joints_deg=(DEFAULT_DEG+i*.1).tolist()) for i in range(16)],physical_motion_executed=False)
        else:r=dict(physical_motion_executed=False)
        self.events.put(dict(type='done',kind=kind,directory=str(d),result=r));return d
    def stop(self,physical=False):self.calls.append('stop_physical' if physical else 'stop')
    def recover_connection(self,camera_only=False):
        self.calls.append('recover_camera' if camera_only else 'recover_connection')
        self.events.put(dict(type='hardware',message='CAN 已激活（模拟）',result=dict(camera_usb=dict(message='USB 检查（模拟）'))))
    def close(self):return True

def main():
    directory=new_session('ui_simulation_test');fake=FakeBackend(directory);root=tk.Tk()
    with patch('act_eval_workbench.app.RUNS',directory),patch('act_eval_workbench.common.RUNS',directory),patch('act_eval_workbench.common.PREFERENCES',directory/'preferences.json'),patch('act_eval_workbench.common.DEFAULT_EXPORTS',directory/'exports'),patch('subprocess.Popen',side_effect=AssertionError('UI test must never spawn a real backend')):
        write(directory/'preferences.json',dict(last_selection=dict(task='banana',model='g2')))
        app=App(root,fake);root.update();assert root.title()=='nero真机评测工作台'
        app.reconnect_button.invoke();app.tick();root.update()
        assert fake.calls[-1]=='recover_connection' and 'CAN 已激活' in app.hardware_status.get()
        app.camera_reconnect_button.invoke();app.tick();root.update()
        assert fake.calls[-1]=='recover_camera'
        fake.hardware_busy=True;app.tick();root.update()
        assert str(app.buttons['formal']['state'])=='disabled' and str(app.reconnect_button['state'])=='disabled'
        fake.hardware_busy=False;app.tick();root.update()
        assert app.settings().task=='banana' and app.settings().model=='g2'
        assert '右爪闭合' in app.buttons['ready']['text'] and '香蕉' in app.reset_hint['text']
        assert not app.settings().relative_limits_enabled
        assert all(str(e['state'])=='disabled' for e in app.range_entries)
        app.range_toggle.invoke();assert app.settings().relative_limits_enabled
        assert all(str(e['state'])=='normal' for e in app.range_entries)
        app.range_toggle.invoke();assert not app.settings().relative_limits_enabled
        app.excursion.set('unused');app.tcp.set('unused');assert not app.settings().relative_limits_enabled
        app.excursion.set('45');app.tcp.set('20')
        # Reproduce a hidden parent: no synchronous grab or visibility wait.
        root.withdraw();app.request('ready');app.tick();root.update()
        assert app.dialog.winfo_exists() and not app.dialog.winfo_viewable()
        before=fake.heartbeats;app.tick();assert fake.heartbeats>before
        original_grab=tk.Toplevel.grab_set;attempts=[]
        def flaky_grab(dialog):
            attempts.append(True)
            if len(attempts)==1:raise tk.TclError('grab failed: window not viewable')
            return original_grab(dialog)
        with patch.object(tk.Toplevel,'grab_set',flaky_grab):
            root.deiconify();deadline=time.monotonic()+2.
            while time.monotonic()<deadline and (len(attempts)<2 or fake.heartbeats<=before):root.update();time.sleep(.01)
        assert len(attempts)>=2 and app.dialog.grab_current()==app.dialog
        app.confirmation_cancel()
        # An unrelated refresh error must stop work and keep the timer alive.
        fake.busy=True;app.last_kind='execute'
        with patch.object(fake,'telemetry_state',side_effect=ValueError('injected refresh failure')):app.tick()
        assert fake.calls[-1]=='stop_physical' and app.tick_handle is not None
        fake.busy=False;before=fake.heartbeats;deadline=time.monotonic()+1.
        while time.monotonic()<deadline and fake.heartbeats<=before:root.update();time.sleep(.01)
        assert fake.heartbeats>before
        fake.camera_health={'left':dict(health='ok',label='通畅',fps=15.),'right':dict(health='error',label='未连接',fps=0.)}
        app.tick();root.update()
        from .camera_health import COLORS
        def light(side):
            canvas,dot,_,_=app.wrist_indicators[side];return canvas.itemcget(dot,'fill')
        assert light('left')==COLORS['ok'] and light('right')==COLORS['error']
        fake.telemetry_age=2.;app.tick();root.update();assert light('left')==COLORS['unknown']
        fake.telemetry_age=0.;fake.camera_health['right']=dict(health='warning',label='图像延迟',fps=15.)
        app.tick();root.update();assert light('right')==COLORS['warning']
        root.geometry('1180x820');root.update()
        assert app.stop_button.winfo_viewable()
        assert app.stop_button.winfo_rooty()+app.stop_button.winfo_height() <= root.winfo_rooty()+root.winfo_height()
        capture(root,directory/'minimum_size.png')
        fake.camera_health['right']=dict(health='ok',label='通畅',fps=15.)
        app.tick();root.update();assert light('right')==COLORS['ok']
        root.geometry('1400x920');root.update()
        for task in TASKS:
            for model in __import__('act_eval_workbench.library',fromlist=['models_for']).models_for(task):
                app.task.set(TASKS[task]);app.model.set(MODELS[model]);app.selection_changed()
                assert app.settings().task==task and app.settings().model==model
        app.task.set(TASKS['banana']);app.selection_changed()
        assert '闭合度1' in app.pose_text.get('1.0','end') and '驱动目标0 mm' in app.pose_text.get('1.0','end')
        app.request('ready');app.tick();root.update()
        area=next(widget for widget in app.dialog.winfo_children() if isinstance(widget,tk.Frame) and any(isinstance(child,tk.Text) for child in widget.winfo_children()))
        text=next(child for child in area.winfo_children() if isinstance(child,tk.Text)).get('1.0','end')
        assert '右闭合度1 / 0.0 mm' in text and '左 100.0 mm' in text
        capture(app.dialog,directory/'banana_ready_confirmation.png');app.confirmation_cancel()
        app.task.set(TASKS['plate']);app.model.set(MODELS['mpi-base']);app.selection_changed()
        for kind in ('load','pose','offline','shadow'):
            app.request(kind);root.update();app.tick();root.update()
        capture(root,directory/'simulation_tab.png')
        for kind in ('ready','default','formal','disable'):
            app.request(kind);app.tick();root.update()
            assert app.dialog.winfo_exists()
            if kind=='formal':capture(app.dialog,directory/'confirmation.png')
            app.confirmation_check.set(True);app.confirmation_execute();app.tick();root.update()
            if kind in ('ready','default'):assert app.last_session is None
            if kind=='disable':assert read(app.last_session/'result.json')['kind']=='formal'
        app.request('ready');app.tick();root.update();before=len(fake.calls);app.confirmation_cancel();assert len(fake.calls)==before
        app.emergency();assert fake.calls[-1]=='stop_physical'
        fake.busy=True;app.request('default');assert app.pending_reset=='default' and fake.calls[-1]=='stop_physical'
        fake.busy=False;app.tick();app.tick();root.update();assert app.dialog.winfo_exists();app.confirmation_cancel()
        app.tick();root.update()  # Let the normal UI refresh unlock record buttons after cancellation.
        app.refresh_records();root.update()
        assert len(app.records.get_children())==1  # Only started formal model execution.
        formal=next(p for p in directory.glob('*/result.json') if read(p)['kind']=='formal')
        app.last_session=formal.parent;app.outcome.set('成功');app.reason.set('SIMULATION ONLY UI save test');app.save_outcome()
        assert read(formal)['adjudication']['success'] is True
        original=formal.read_bytes();record_id=formal.parent.name
        app.notebook.select(2);app.records.selection_set(record_id);root.update()
        errors=[]
        with patch('act_eval_workbench.app.messagebox.showerror',side_effect=lambda *args,**kw:errors.append(args)),\
             patch('act_eval_workbench.app.messagebox.showinfo'),\
             patch('act_eval_workbench.app.choose_export_directory',return_value=str(directory/'exports')):
            app.record_buttons['export_selected'].invoke()
            assert len(read(app.last_export/'records.json')['records'])==1
            assert read(directory/'preferences.json')['last_export_directory']==str(directory/'exports')
            with patch('act_eval_workbench.app.messagebox.askyesno',return_value=False):
                app.record_buttons['clear_selected'].invoke()
            assert record_id in app.records.get_children()
            with patch('act_eval_workbench.app.messagebox.askyesno',return_value=True):
                app.record_buttons['clear_selected'].invoke()
                assert record_id not in app.records.get_children() and formal.read_bytes()==original
                app.record_buttons['restore'].invoke()
                assert record_id in app.records.get_children()
                app.record_buttons['export_all'].invoke()
                assert len(read(app.last_export/'records.json')['records'])==1
                app.record_buttons['clear_all'].invoke();assert not app.records.get_children()
                app.record_buttons['restore'].invoke();assert len(app.records.get_children())==1
            fake.busy=True;app.tick();root.update()
            assert all(str(b['state'])=='disabled' for b in app.record_buttons.values())
            fake.busy=False;app.tick();root.update()
            assert not errors,errors
        root.geometry('1180x820');root.update()
        for button in app.record_buttons.values():
            assert button.winfo_viewable()
            assert button.winfo_rootx()+button.winfo_width()<=root.winfo_rootx()+root.winfo_width()
        capture(root,directory/'records_minimum_size.png')
        root.geometry('1400x920');root.update();capture(root,directory/'records_tab.png')
        from .save_dialog import ExportDirectoryDialog
        picker=ExportDirectoryDialog(root);root.update()
        assert picker.current==directory/'exports'
        with patch('act_eval_workbench.save_dialog.simpledialog.askstring',return_value='新建测试文件夹'):
            picker.new_button.invoke()
        assert picker.current.name=='新建测试文件夹' and picker.current.is_dir()
        picker.window.geometry('650x400');root.update()
        assert picker.save_button.winfo_height()==picker.new_button.winfo_height()
        assert picker.save_button.winfo_y()+picker.save_button.winfo_height()<=picker.save_button.master.winfo_height()
        capture(picker.window,directory/'save_directory_minimum.png')
        picker.window.geometry('850x520');root.update();capture(picker.window,directory/'save_directory_dialog.png')
        root.after(80,picker.save_button.invoke)
        assert picker.show()==str(directory/'exports/新建测试文件夹')
        cancelled=ExportDirectoryDialog(root);root.update();cancelled.cancel();assert cancelled.result is None
        assert read(directory/'preferences.json')['last_export_directory']==str(directory/'exports')
        app.duration.set('nan')
        try:app.settings()
        except ValueError:pass
        else:raise AssertionError('NaN duration accepted')
        app.duration.set('180');app.notebook.select(0);root.update();capture(root,directory/'live_tab.png')
        app.task.set(TASKS['banana']);app.model.set(MODELS['g2']);app.selection_changed()
        app.on_close();app.tick()
        restarted=tk.Tk();again=App(restarted,FakeBackend(directory));restarted.update()
        assert again.settings().task=='banana' and again.settings().model=='g2'
        assert '右爪闭合' in again.buttons['ready']['text']
        capture(restarted,directory/'restored_banana_g2.png')
        again.request('ready');again.tick();restarted.update()
        restored_plan=read(again.backend.active/'plan.json')
        assert restored_plan['settings']['task']=='banana' and restored_plan['trajectory']['gripper_goal_m']==[0.,.1]
        assert read(directory/'preferences.json')['last_export_directory']==str(directory/'exports')
        again.confirmation_cancel();again.on_close();again.tick()
    write(directory/'ui_test_result.json',dict(passed=True,physical_motion_executed=False,calls=fake.calls,
        tested=['9 task/model selections','4 read-only buttons','4 physical-action widgets with fake backend',
                'relative debugging limits off by default and toggle reflected in settings',
                'hidden parent confirmation heartbeat','transient grab failure recovery','refresh exception stop and timer recovery',
                'plan confirmation and cancellation','emergency stop dispatch','reset queued after stop','result adjudication',
                'selected/all CSV JSON export','persistent export location','new folder in save dialog','cancel preserves preference',
                'clear cancellation','selected/all clear and restore preserving original bytes',
                'record buttons locked during operation','only started formal inference shown/exported',
                'records minimum window layout','nero window name','banana/G2 window restart preserves task and right-close reset target',
                'wrist camera green/red/yellow and monitor-loss gray/recovery','NaN rejection','window close']))
    print('UI_TEST_PASSED',directory,flush=True)
if __name__=='__main__':main()
