"""Production workers: closed/open, enabled/disabled, reset, stop and fault."""
import os
import subprocess
import sys
import threading
import time
import numpy as np
from .common import *
from .test_integration import isolated

def main():
    isolated();root=new_session('gripper_reset_validation');print('RESET_TEST_DIRECTORY',root,flush=True)
    results=[]
    cases=[('default_closed_disabled','default',0.,False,False),
           ('ready_partial_enabled','ready',.029,True,False),
           ('banana_ready_open','ready',.1,False,False),
           ('banana_ready_contact','ready',.1,False,False),
           ('banana_default_closed','default',.028,True,False),
           ('banana_disable_partial','disable',.028,True,False),
           ('disable_partial_enabled','disable',.05,True,False),
           ('disable_already_open','disable',.1,False,False),
           ('ready_stuck','ready',.029,False,True),
           ('disable_stuck','disable',.029,True,True),
           ('disable_cancelled','disable',.029,True,False)]
    for name,kind,width,enabled,stuck in cases:
        directory=root/name;directory.mkdir();heart=directory/'heartbeat';heart.touch();keep=threading.Event();keep.set()
        def beat():
            while keep.is_set():heart.touch();time.sleep(.1)
        thread=threading.Thread(target=beat,daemon=True);thread.start();processes=[]
        env=dict(os.environ,FAKE_START_WIDTH=str(width),FAKE_START_ENABLED=str(int(enabled)),FAKE_GRIP_STUCK=str(int(stuck)))
        if name=='banana_ready_contact':env['FAKE_RIGHT_CONTACT_WIDTH']='.04205'
        try:
            for args,label in [([sys.executable,'-m','act_eval_workbench.fake_reset_driver'],'drivers'),
                               ([sys.executable,str(LEGACY/'test_plate_formal_trial.py'),'--camera'],'camera')]:
                with (directory/(label+'.log')).open('w') as log:processes.append(subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT))
            task='banana' if name.startswith('banana_') else 'plate'
            write(directory/'config.json',dict(settings=Settings(task=task).json(),port=8026))
            base=[sys.executable,'-m','act_eval_workbench.worker']
            args=['--directory',str(directory),'--heartbeat',str(heart)]
            prep=subprocess.run(base+['prepare',*args,'--kind',kind],capture_output=True,text=True,timeout=40)
            assert prep.returncode==0,(name,prep.stdout,prep.stderr)
            plan=read(directory/'plan.json');assert plan['simulation_only'];approve(plan,directory,'ISOLATED SIMULATION')
            if name=='disable_cancelled':
                def cancel():
                    deadline=time.monotonic()+20
                    while time.monotonic()<deadline:
                        f=directory/'execution/events.jsonl'
                        if f.exists() and 'command_dispatch' in f.read_text():
                            time.sleep(.1);(directory/'STOP').touch();return
                        time.sleep(.02)
                    (directory/'STOP').touch()
                stopper=threading.Thread(target=cancel,daemon=True);stopper.start()
            execution=subprocess.run(base+['execute',*args],capture_output=True,text=True,timeout=170)
            (directory/'worker.log').write_text(execution.stdout+execution.stderr)
            result=read(directory/'result.json')
            if stuck or name=='disable_cancelled':
                assert not result['runtime_completed'],result
                if kind=='disable':assert result['gripper_disable_verified'],result
                events=[json.loads(line) for line in (directory/'execution/events.jsonl').read_text().splitlines()]
                commands=[e for e in events if e['event']=='command_dispatch']
                assert commands
                q=np.array([e['joints_rad'] for e in commands]);assert np.max(np.abs(q-q[0]))<1e-12
            else:
                assert execution.returncode==0 and result['runtime_completed'],result
                measured=result['opening']['final_widths_m'] if kind=='disable' else result['final_widths_m']
                target=ready_gripper_target(task) if kind=='ready' else [.1,.1]
                expected=[.04205,.1] if name=='banana_ready_contact' else target
                assert np.max(np.abs(np.array(measured)-expected))<=.002,(name,measured,expected)
                if name=='banana_ready_contact':assert result['gripper_completion']['mode']=='force_contact'
                if kind=='ready':check_start_grippers(task,measured)
                if kind=='disable':
                    assert result['gripper_disable_verified'],result
                    if name=='disable_already_open':
                        assert result['opening']['already_open'] and result['opening']['commands_sent']==0
                else:
                    assert np.max(np.abs(np.deg2rad(result['final_joints_deg'])-np.array(plan['trajectory']['goal'])))<.02
                    events=[json.loads(line) for line in (directory/'execution/events.jsonl').read_text().splitlines()]
                    commands=[e for e in events if e['event']=='command_dispatch'];end=plan['trajectory']['opening_last_index']
                    assert any(e['event']=='reset_grippers_open_verified' for e in events)
                    if task=='banana' and kind=='ready':
                        assert plan['gripper_policy']=='task_target_after_joint_arrival'
                        verified=next(e for e in events if e['event']=='reset_task_grippers_verified')
                        assert not verified['object_presence_verified']
                        assert commands[-1]['index']==plan['trajectory']['final_gripper_index']
                        np.testing.assert_allclose(commands[-1]['feedback_rad'],plan['trajectory']['goal'],atol=.001)
                        np.testing.assert_allclose(commands[-1]['gripper_width_m'],target)
                    for e in commands:
                        if np.max(np.abs(np.array(e['joints_rad'])-plan['start_joints_rad']))>1e-12:
                            assert np.max(np.abs(np.array(e['feedback_gripper_width_m'])-.1))<=.002
            results.append(dict(case=name,passed=True,runtime_completed=result['runtime_completed'],error=result.get('error',result.get('errors'))))
            write(root/'summary.json',dict(results=results,physical_motion_executed=False));print(json.dumps(results[-1]),flush=True)
        finally:
            keep.clear();thread.join(timeout=1)
            for proc in processes:
                if proc.poll() is None:proc.terminate()
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:proc.kill();proc.wait()
            time.sleep(1)
    print('RESET_ALL_PASSED',root,flush=True)

if __name__=='__main__':main()
