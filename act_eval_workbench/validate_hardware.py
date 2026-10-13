"""Explicit opt-in small hardware acceptance. Always attempts final disable.

This is a maintenance validation, never a task-success trial. User authorized
development tests, limited motion and final disable on 2026-09-23.
"""
import ctypes
import os
import sys
import time
from pathlib import Path
for lib in ('libtcl8.6.so','libtk8.6.so'):
    ctypes.CDLL('/usr/lib/x86_64-linux-gnu/'+lib,mode=ctypes.RTLD_GLOBAL)
os.environ['TCL_LIBRARY']='/usr/share/tcltk/tcl8.6'
os.environ['TK_LIBRARY']='/usr/share/tcltk/tk8.6'
import tkinter as tk
import numpy as np
from .app import App
from .common import *
from .capture_window import capture
from .worker import stop_services

def main():
    if sys.argv[1:]!=['--execute-micro']:raise SystemExit('Explicit --execute-micro required; both joint4 +0.3 deg then disable')
    if os.environ.get('ACT_EVAL_FAKE_DRIVER') or os.environ.get('ROS_DOMAIN_ID')=='174':raise RuntimeError('Expected real ROS domain')
    directory=new_session('hardware_micro_acceptance');root=tk.Tk();app=App(root);app.operator.set('User-authorized desktop acceptance')
    summary=dict(physical_motion_executed=False,scope='Both joint4 +0.3 deg; observed deviation guard 0.5 deg; no gripper commands; final disable',
                 user_authorization='2026-09-23 development tests authorized without individual permission; small motion only; disable after tests')
    def until(condition,seconds=90):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            root.update()
            if condition():return
            time.sleep(.01)
        raise TimeoutError('Desktop acceptance wait exceeded')
    def start(kind):
        app.request(kind);d=app.last_session
        until(lambda:not app.backend.busy)
        until(lambda:bool(app.dialog and app.dialog.winfo_exists()),5)
        plan=read(d/'plan.json')
        if kind=='micro':
            delta=np.rad2deg(np.array(plan['trajectory']['goal'])-plan['start_joints_rad'])
            expected=np.zeros((2,7));expected[:,3]=.3
            np.testing.assert_allclose(delta,expected,atol=1e-9)
            assert plan['gripper_commands']==0 and not plan['simulation_only']
            capture(app.dialog,directory/'micro_confirmation.png')
        app.confirmation_check.set(True);app.confirmation_execute()
        until(lambda:not app.backend.busy)
        # Process completion event so the UI shows recorded results.
        until(lambda:app.last_kind=='execute' and (d/'result.json').exists(),5)
        root.update();capture(root,directory/(kind+'_result.png'))
        return d,read(d/'result.json')
    try:
        until(lambda:len(app.backend.telemetry_state().get('arms',{}))==2,15)
        before=app.backend.telemetry_state();write(directory/'before_telemetry.json',before)
        assert sum(before.get('control_publishers',{}).values())==0
        d,r=start('micro');summary.update(micro_directory=str(d),micro=r,physical_motion_executed=bool(r['started']))
        if not r.get('runtime_completed') or r.get('error'):raise RuntimeError('Micro test stopped: '+str(r.get('error')))
        assert r['micro_max_observed_deg']<=.5
        app.stop_button.invoke();until(lambda:not app.backend.stop_jobs,15)
        d,r=start('disable');summary.update(disable_directory=str(d),disable=r)
        assert r['motor_disable_requested'] and not r['errors']
        summary['passed']=True
    except Exception as exc:
        summary.update(passed=False,error=f'{type(exc).__name__}: {exc}')
    finally:
        # A failed UI test must not leave arms enabled. Stop the active worker
        # cooperatively, then independently repeat close/hold/disable services.
        app.backend.stop(physical=True)
        try:until(lambda:not app.backend.busy and not app.backend.stop_jobs,25)
        except Exception as exc:summary['wait_cleanup_error']=str(exc)
        try:summary['final_disable']=stop_services(True)
        except Exception as exc:summary['final_disable_error']=str(exc);summary['passed']=False
        if summary.get('final_disable',{}).get('errors'):summary['passed']=False
        write(directory/'summary.json',summary)
        app.on_close()
        try:until(lambda:not root.winfo_exists(),15)
        except tk.TclError:pass
    print(directory,json.dumps(summary,ensure_ascii=False),flush=True)
    return 0 if summary.get('passed') else 2
if __name__=='__main__':raise SystemExit(main())
