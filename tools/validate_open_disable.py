"""Explicitly requested maintenance test: hold joints, open grippers, disable all."""
import argparse
import ctypes
import fcntl
import os
from pathlib import Path
import time

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute-open-disable',action='store_true');args=parser.parse_args()
    if not args.execute_open_disable:raise SystemExit('Physical action requires --execute-open-disable')
    assert os.environ.get('ROS_DOMAIN_ID')!='174' and os.environ.get('ACT_EVAL_FAKE_DRIVER')!='1'
    lock=open('/tmp/act_eval_workbench_desktop.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for library in ('libtcl8.6.so','libtk8.6.so'):ctypes.CDLL('/usr/lib/x86_64-linux-gnu/'+library,mode=ctypes.RTLD_GLOBAL)
    os.environ['TCL_LIBRARY']='/usr/share/tcltk/tcl8.6';os.environ['TK_LIBRARY']='/usr/share/tcltk/tk8.6'
    import tkinter as tk
    import numpy as np
    from act_eval_workbench.app import App
    from act_eval_workbench.common import new_session,read,write
    from act_eval_workbench.capture_window import capture
    from act_eval_workbench.worker import stop_services
    directory=new_session('hardware_open_disable');summary=dict(physical_motion_executed=False,scope='hold joints; open both grippers; disable arms and grippers')
    root=tk.Tk();app=App(root)
    def wait(check,timeout):
        deadline=time.monotonic()+timeout
        while not check():
            root.update();time.sleep(.02)
            if time.monotonic()>deadline:raise TimeoutError('GUI state wait expired')
    try:
        def live():
            state=app.backend.telemetry_state()
            return len(state.get('arms',{}))==2 and len(state.get('grippers',{}))==2 and all(-.05<=r['age_s']<=.2 for r in state['arms'].values())
        wait(live,20);state=app.backend.telemetry_state();assert not any(state['control_publishers'].values())
        write(directory/'before.json',state);capture(root,directory/'before.png')
        app.operator.set('Authorized maintenance: open grippers then disable all')
        app.buttons['disable'].invoke();wait(lambda:app.dialog is not None and app.dialog.winfo_exists(),15)
        plan=read(app.last_session/'plan.json');assert plan['kind']=='disable' and plan['gripper_policy']=='open_then_disable'
        capture(app.dialog,directory/'confirmation.png');session=app.last_session
        app.confirmation_check.set(True);app.confirmation_execute();summary['physical_motion_executed']=True
        wait(lambda:not app.backend.busy and (session/'result.json').exists(),90)
        result=read(session/'result.json');summary.update(session=str(session),result=result)
        assert result['runtime_completed'] and result['gripper_disable_verified'] and not result['errors'],result
        events=[__import__('json').loads(line) for line in (session/'execution/events.jsonl').read_text().splitlines()]
        commands=[e for e in events if e['event']=='command_dispatch']
        if result['opening'].get('already_open'):
            assert not commands
            summary.update(physical_motion_executed=False,max_joint_displacement_while_opening_deg=0.,max_command_joint_displacement_deg=0.)
        else:
            assert len(commands)==len(read(session/'execution/opening_plan.json')['joints'])
            q=np.array([e['feedback_rad'] for e in commands]);requested=np.array([e['joints_rad'] for e in commands])
            summary['max_joint_displacement_while_opening_deg']=float(np.rad2deg(np.max(np.abs(q-q[0]))))
            summary['max_command_joint_displacement_deg']=float(np.rad2deg(np.max(np.abs(requested-requested[0]))))
        assert summary['max_command_joint_displacement_deg']==0 and summary['max_joint_displacement_while_opening_deg']<=.5
        summary['passed']=True
    except Exception as exc:
        summary.update(passed=False,error=f'{type(exc).__name__}: {exc}');raise
    finally:
        if app.backend.busy:
            app.backend.stop(physical=True)
            try:wait(lambda:not app.backend.busy,15)
            except TimeoutError:pass
        summary['final_disable']=stop_services(True)
        wait(lambda:all(not r['enabled'] and -.05<=r['age_s']<=.3 for r in app.backend.telemetry_state().get('grippers',{}).values()) and len(app.backend.telemetry_state().get('grippers',{}))==2,10)
        write(directory/'after.json',app.backend.telemetry_state());capture(root,directory/'after.png')
        write(directory/'summary.json',summary);print('HARDWARE_OPEN_DISABLE',directory,summary,flush=True)
        app.on_close();wait(lambda:app.backend.closing,3)
        deadline=time.monotonic()+5
        while time.monotonic()<deadline:
            try:root.update();time.sleep(.02)
            except tk.TclError:break

if __name__=='__main__':main()
