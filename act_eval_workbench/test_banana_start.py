"""Real worker + simulated ROS/server: off-reference joints and 42.05 mm grasp."""
import os
import socket
import subprocess
import sys
import threading
import time
import numpy as np
from .common import *
from .test_integration import isolated

def main():
    isolated();root=new_session('banana_start_validation');processes=[];keep=threading.Event();keep.set()
    heart=root/'heartbeat';heart.touch()
    def beat():
        while keep.is_set():heart.touch();time.sleep(.1)
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    env=dict(os.environ,FAKE_TASK='banana',FAKE_MODEL='g2',FAKE_START_WIDTH_RIGHT='.04205',FAKE_START_WIDTH_LEFT='.1',
        FAKE_START_J1_OFFSET='.15',FAKE_START_ENABLED='1',FAKE_RIGHT_CONTACT_WIDTH='.04205')
    try:
        for args,name in [([sys.executable,'-m','act_eval_workbench.fake_reset_driver'],'driver'),
                          ([sys.executable,str(LEGACY/'test_plate_formal_trial.py'),'--camera'],'camera'),
                          ([sys.executable,'-m','act_eval_workbench.test_workflow','--server',str(port)],'server')]:
            with (root/(name+'.log')).open('w') as f:processes.append(subprocess.Popen(args,env=env,stdout=f,stderr=subprocess.STDOUT))
        d=root/'formal';d.mkdir();write(d/'config.json',dict(settings=Settings(task='banana',model='g2',duration_s=3.).json(),port=port))
        def worker(mode):
            args=[sys.executable,'-m','act_eval_workbench.worker',mode,'--directory',str(d),'--heartbeat',str(heart)]
            if mode=='prepare':args+=['--kind','formal']
            done=subprocess.run(args,capture_output=True,text=True,timeout=60)
            (d/(mode+'.log')).write_text(done.stdout+done.stderr)
            assert done.returncode==0,(done.stdout,done.stderr,read(d/'result.json') if (d/'result.json').exists() else None)
        worker('prepare');plan=read(d/'plan.json')
        assert plan['simulation_only'] and not plan['start_pose_check']['within_reference']
        assert not plan['gripper_start_check']['within_training_range'] and plan['start_warnings']
        assert abs(plan['capture']['state'][7]-(1-.04205/.1))<1e-6  # Real feedback, never fake a closed state.
        approve(plan,d,'ISOLATED SIMULATION');worker('execute')
        result=read(d/'result.json');assert result['runtime_completed'] and result['simulation_only']
        events=[json.loads(line) for line in (d/'execution/events.jsonl').read_text().splitlines()]
        commands=[e for e in events if e['event']=='command_dispatch']
        hold=[e for e in commands if e['chunk']==-1];assert hold and all(not e['gripper_command_sent'] for e in hold)
        policy=[e for e in commands if e['chunk']>=0 and e['gripper_command_sent']];assert policy
        assert all(e['gripper_width_m']==[0.,.1] for e in policy)
        assert abs(result['final_widths_m'][0]-.04205)<.002
        write(root/'summary.json',dict(passed=True,physical_motion_executed=False,commands=len(commands),
            off_reference_deg=plan['start_pose_check']['max_deviation_deg'],actual_width_m=.04205,
            inference_state_closedness=plan['capture']['state'][7],policy_command_width_m=policy[0]['gripper_width_m'],
            startup_gripper_command_preserved=True))
        print('BANANA_START_PASSED',root,flush=True)
    finally:
        keep.clear();thread.join(timeout=2)
        for proc in processes:
            if proc.poll() is None:proc.terminate()
            try:proc.wait(timeout=3)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()

if __name__=='__main__':main()
