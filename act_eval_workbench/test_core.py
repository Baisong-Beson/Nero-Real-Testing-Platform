import json
import tempfile
import time
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from . import common as m
from . import controller as c
from . import worker as w
from .library import BUILTIN_TASKS,BUILTIN_MODELS

class CoreTests(unittest.TestCase):
    def test_task_specific_gripper_start(self):
        for task in BUILTIN_TASKS:
            m.check_start_grippers(task,m.ready_pose(task)['gripper_width_mean_m'])
            target=m.ready_gripper_target(task);m.check_start_grippers(task,target)
            self.assertEqual(target[1],.1)
            self.assertAlmostEqual(target[0],0. if task=='banana' else .1)
        for task,widths in [('banana',[.04205,.1]),('banana',[0.,.1]),('banana',[.1,.1]),('plate',[.027,.1]),('holder',[.1,.03])]:
            check=m.check_start_grippers(task,widths)
            self.assertFalse(check['within_training_range']);self.assertTrue(check['warnings'])
            self.assertEqual(check['enforcement'],'advisory_only')
        for widths in ([float('nan'),.1],[-.01,.1],[.102,.1]):
            with self.assertRaises(ValueError):m.check_start_grippers('banana',widths)
    def test_ready_all_tasks_recomputed_train_only(self):
        for task in BUILTIN_TASKS:
            r=m.ready_pose(task);self.assertGreater(r['train_episodes'],30);m.limits(r['joints_rad'])
        np.testing.assert_allclose(m.ready_pose('plate')['joints_rad'],c.pos.READY,atol=1e-10)
    def test_settings_reject_invalid_and_protocol_distinguishes_duration(self):
        for kwargs in [dict(task='other'),dict(model='other'),dict(duration_s=float('nan')),dict(duration_s=0),dict(excursion_deg=91),dict(tcp_cm=41),dict(operator='')]:
            with self.assertRaises(ValueError):m.Settings(**kwargs).checked()
        self.assertNotEqual(m.digest(m.Settings(duration_s=30).protocol()),m.digest(m.Settings(duration_s=180).protocol()))
    def test_policy_manifest_all_nine(self):
        ids=[]
        for task in BUILTIN_TASKS:
            for model in BUILTIN_MODELS:
                r=m.catalog(task,model);ids.append(r['policy_sha256'])
                self.assertEqual(len(r['encoder_state_sha256']),64)
        self.assertEqual(len(set(ids)),9)
    def test_approval_tampering_expiration_and_scope(self):
        body=dict(prepared_unix_s=100,kind='ready');body['plan_sha256']=m.digest(body)
        a=dict(status='approved',source='local_desktop_confirm',operator='TEST',plan_sha256=body['plan_sha256'],approved_unix_s=101,operator_present_with_estop=True)
        m.check_approval(body,a,102)
        for field,value in [('source','background'),('plan_sha256','bad'),('operator',''),('operator_present_with_estop',False),('approved_unix_s',99)]:
            bad=dict(a);bad[field]=value
            with self.assertRaises(ValueError):m.check_approval(body,bad,102)
        with self.assertRaises(ValueError):m.check_approval(body,a,500)
        with self.assertRaises(ValueError):m.check_approval(dict(body,kind='formal'),a,102)
    def test_stop_and_heartbeat(self):
        with tempfile.TemporaryDirectory() as path:
            path=Path(path);heart=path/'heart';heart.touch();op=w.Operator(path,heart)
            self.assertTrue(op.permit());(path/'STOP').touch();op.last_check=0;self.assertFalse(op.permit())
        with tempfile.TemporaryDirectory() as path:
            path=Path(path);heart=path/'heart';heart.touch()
            import os
            os.utime(heart,(time.time()-3,time.time()-3));self.assertFalse(w.Operator(path,heart).permit())
    def test_default_ready_trajectories_bounded_and_continuous(self):
        for task in m.TASKS:
            goal=np.array(m.ready_pose(task)['joints_rad']);start=np.deg2rad(m.DEFAULT_DEG)
            path=w.positioning_plan(start,goal,True)
            np.testing.assert_array_equal(path['joints'][0],start);np.testing.assert_allclose(path['joints'][-1],goal,atol=1e-12)
            self.assertLessEqual(path['max_velocity_rad_s'],.05+1e-8);self.assertLessEqual(path['max_acceleration_rad_s2'],.05+1e-8)
            self.assertLess(np.abs(np.diff(path['joints'],axis=0)).max(),.00251)
    def test_controller_tracks_and_maps_gripper(self):
        state=np.array(m.ready_pose('plate')['joints_rad']);slew=c.Slew(state,[.1,.1]);target=state.copy();target[:,0]+=.1
        q=state.copy()
        for i in range(300):
            q,width=slew.next(target,[.03,.04],.05,q)
        np.testing.assert_allclose(q,target,atol=1e-4);np.testing.assert_allclose(width,[.03,.04],atol=1e-4)
        actions=np.tile(np.r_[state[0],1,state[1],0],(16,1));_,w=c.unpack(actions);np.testing.assert_equal(w[0],[0,.1])
    def test_history_groups_limits_and_excludes_reset(self):
        with tempfile.TemporaryDirectory() as path,patch.object(m,'RUNS',Path(path)):
            for i,(kind,duration,verdict) in enumerate([('formal',30,False),('formal',180,None),('formal',180,True),('ready',180,True)]):
                s=m.Settings(duration_s=duration);m.write(Path(path)/str(i)/'result.json',dict(kind=kind,started=True,settings=s.json(),protocol_sha256=m.digest(s.protocol()),adjudication={'success':verdict}))
            groups=m.summarize_records();self.assertEqual(len(groups),2)
            g=next(x for x in groups if x['duration_s']==180);self.assertEqual((g['started'],g['success'],g['pending']),(2,1,1))

if __name__=='__main__':unittest.main()
