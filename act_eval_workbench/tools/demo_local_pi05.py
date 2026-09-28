"""Demonstrate real GUI imports, loading and offline replay. No motion action."""
import ctypes
import os
for name in ('libtcl8.6.so','libtk8.6.so'):ctypes.CDLL('/usr/lib/x86_64-linux-gnu/'+name,mode=ctypes.RTLD_GLOBAL)
os.environ['TCL_LIBRARY']='/usr/share/tcltk/tcl8.6';os.environ['TK_LIBRARY']='/usr/share/tcltk/tk8.6'
import fcntl
import json
from pathlib import Path
import time
import tkinter as tk
from unittest.mock import patch
import numpy as np
from nero_eval_workbench.app import App
from nero_eval_workbench import common as c
from nero_eval_workbench.capture_window import capture

def metrics(result,package):
    with np.load(package/'expert_actions.npz',allow_pickle=False) as z:expected=z['actions']
    with np.load(package/'probe.npz',allow_pickle=False) as z:state=z['state']
    actual=np.asarray([p['raw_actions'] for p in result['probes']])
    joints=[*range(7),*range(8,15)];q=actual[:,:,joints].reshape(len(actual),16,2,7)
    lower=c.LOWER[None,None,None,:];upper=c.UPPER[None,None,None,:];outside=(q<lower)|(q>upper)
    err=np.abs(actual[:,:,joints]-expected[:,:,joints])*180/np.pi
    hold=np.abs(state[:,None,joints]-expected[:,:,joints])*180/np.pi
    latency=np.array([r['inference_ms'] for r in result['probes']])
    evidence=c.read(package/'probe_evidence.json')['samples']
    rows=[]
    for i,r in enumerate(evidence):
        rows.append(dict(episode=r['episode'],frame=r['frame'],domain=r['camera_domain'],joint_mae_deg_first4=float(err[i,:4].mean()),joint_mae_deg_h16=float(err[i].mean()),hold_mae_deg_h16=float(hold[i].mean()),inference_ms=float(latency[i]),executed_prefix_joint_limit_violation=bool(outside[i,:4].any()),whole_chunk_joint_limit_violation=bool(outside[i].any()),geometry_warnings=len(result['probes'][i]['violations'])))
    return dict(samples=len(actual),shape=list(actual.shape),finite=bool(np.isfinite(actual).all()),split='training_replay_not_held_out',
        joint_mae_deg_first4=float(err[:,:4].mean()),joint_mae_deg_h16=float(err.mean()),hold_baseline_mae_deg_first4=float(hold[:,:4].mean()),hold_baseline_mae_deg_h16=float(hold.mean()),
        gripper_width_mae_mm=float(np.abs(np.clip(actual[:,:,[7,15]],0,1)-np.clip(expected[:,:,[7,15]],0,1)).mean()*100),
        median_inference_ms=float(np.median(latency)),p95_inference_ms=float(np.percentile(latency,95)),
        samples_with_prefix_joint_violations=int(outside[:,:4].any(axis=(1,2,3)).sum()),samples_with_tail_or_prefix_joint_violations=int(outside.any(axis=(1,2,3)).sum()),
        samples_with_geometry_warnings=sum(bool(p['violations']) for p in result['probes']),rows=rows,
        task_success_rate=None,physical_motion_executed=False)

def main():
    lock=open('/tmp/act_eval_workbench_desktop.lock','a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    package=c.PLATFORM/'import_packages/pi05_nero_dual_v2_14999';out=c.PLATFORM/'diagnostics/pi05_import_demo';out.mkdir(parents=True,exist_ok=True)
    root=tk.Tk();app=App(root);phase=0;deadline=time.monotonic()+1100;load_started=None;load_seconds=None
    task='marker_drawer_pi05_v2';model='pi05_nero_dual_v2_14999'
    def snapshot(name):capture(root,out/(name+'.png'))
    def import_file(kind):
        # Exercise the same real UI handler as a user selecting this file.
        with patch('nero_eval_workbench.library_ui.filedialog.askopenfilename',return_value=str(package/(kind+'.json'))):app.library.import_file(kind)
    def step():
        nonlocal phase,load_started,load_seconds
        try:
            if time.monotonic()>deadline:raise TimeoutError('导入演示未在预期时间内完成，请查看日志')
            if phase==0 and not app.backend.hardware_busy:
                app.notebook.select(3);snapshot('01_import_buttons');app.log('演示：导入任务 '+str(package/'task.json'))
                if task not in c.TASKS:import_file('task')
                else:app.task.set(c.TASKS[task]);app.selection_changed()
                phase=1
            elif phase==1 and not app.library.busy:
                if task not in c.TASKS:raise RuntimeError('任务导入失败')
                snapshot('02_task_imported');app.log('演示：导入模型 '+str(package/'model.json'))
                if model not in c.MODELS:import_file('model')
                else:app.model.set(c.MODELS[model]);app.selection_changed()
                phase=2
            elif phase==2 and not app.library.busy:
                if model not in c.MODELS:raise RuntimeError('模型导入失败')
                app.library.table.selection_set('model:'+model);app.library.details();snapshot('03_model_imported')
                load_started=time.monotonic();app.request('load');phase=3
            elif phase==3 and not app.backend.busy:
                if app.backend.server.selected!=(task,model):raise RuntimeError('模型加载失败，请查看界面与模型服务日志')
                load_seconds=time.monotonic()-load_started;snapshot('04_model_loaded');app.request('offline');phase=4
            elif phase==4 and not app.backend.busy:
                path=app.backend.active/'result.json'
                if not path.exists():raise RuntimeError('离线回放失败，请查看界面日志')
                result=c.read(path);report=metrics(result,package)
                report.update(model_id=model,task_id=task,checkpoint_sha256=c.catalog(task,model)['policy_sha256'],model_load_and_warmup_s=load_seconds,offline_result=str(path),desktop_session=str(app.backend.directory),import_package=str(package),completed=True)
                c.write(out/'result.json',report)
                app.notebook.select(1);snapshot('05_offline_result')
                app.log(f'π0.5 已完成 {report["samples"]} 个训练样本离线推理；中位延迟 {report["median_inference_ms"]:.1f} ms。未启动真机。')
                app.notebook.select(3);app.library.table.selection_set('model:'+model);app.library.details();snapshot('06_ready_to_use')
                print('PI05_GUI_IMPORT_AND_REPLAY_COMPLETED',out,flush=True);return
            c.write(out/'progress.json',dict(phase=phase,updated_unix_s=time.time(),model_state=app.model_state.get(),state=app.state.get(),import_busy=app.library.busy,backend_busy=app.backend.busy))
        except Exception as exc:
            c.write(out/'error.json',dict(error=str(exc),phase=phase));app.log('导入测试未完成：'+str(exc));print('PI05_DEMO_ERROR',str(exc),flush=True);return
        root.after(350,step)
    root.after(1000,step);root.mainloop()

if __name__=='__main__':main()
