"""Pure-data import and inference contract regressions; no robot imports."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from . import common as c,library as lib
from .policy_protocol import metadata,validate_service,actions

class LibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.base=Path(self.tmp.name);self.patch=patch.object(c,'PLATFORM',self.base/'platform');self.patch.start()
        self.source=self.base/'source';self.source.mkdir();c.refresh_catalog()
    def tearDown(self):self.patch.stop();c.refresh_catalog();self.tmp.cleanup()
    def task(self,**kw):
        d=dict(schema='nero.task.v1',id='test_task',label='Test task',instruction='Pick up the cup.',success_definition='In tray for 2 seconds',ready=dict(method='explicit',joints_deg=c.DEFAULT_DEG.tolist(),gripper_width_m=[.04,.1]));d.update(kw);return d
    def put(self,d,kind):
        path=self.source/f'{kind}.json';c.write(path,d);return lib.import_package(path,kind)
    def model(self,**kw):
        d=dict(schema='nero.model.v1',id='test_model',label='Test model',adapter='websocket',tasks=['test_task'],endpoint='ws://127.0.0.1:9000',service_model_id='fixture',checkpoint_sha256='a'*64,action_contract=lib.CONTRACT,
            runtime=dict(action_horizon=3,steps_per_replan=2,dt_s=.1,inference_timeout_s=2.,cameras=['main']));d.update(kw);return d
    def test_import_persist_compatibility_identity_and_tamper(self):
        self.put(self.task(),'task');self.put(self.model(),'model');c.refresh_catalog()
        self.assertIn('test_task',c.TASKS);self.assertEqual(list(lib.models_for('test_task')),['test_model'])
        self.assertFalse(lib.compatible('plate','test_model'))
        identity=c.catalog('test_task','test_model',True)
        self.assertEqual(identity['instruction'],'Pick up the cup.')
        self.assertEqual(identity['runtime']['action_horizon'],3)
        self.assertEqual(c.ready_gripper_target('test_task'),[.04,.1])
        with self.assertRaises(ValueError):self.put(self.task(),'task')
        dest=lib.root()/'tasks/test_task/task.json';data=c.read(dest);data['instruction']='modified';c.write(dest,data)
        with self.assertRaises(ValueError):c.ready_pose('test_task')
    def test_mean_copies_cache_and_recomputes(self):
        state=np.r_[np.deg2rad(c.DEFAULT_DEG[0]),.3,np.deg2rad(c.DEFAULT_DEG[1]),0.]
        states=np.stack([state,state.copy()]);states[1,0]+=.02
        np.savez(self.source/'starts.npz',start_states=states)
        self.put(self.task(ready=dict(method='training_mean',states_file='starts.npz',gripper_width_m=[0.,.1])),'task')
        (self.source/'starts.npz').unlink();pose=c.ready_pose('test_task')
        self.assertAlmostEqual(pose['joints_rad'][0][0],.01);self.assertEqual(pose['train_episodes'],2)
        self.assertAlmostEqual(pose['gripper_width_mean_m'][0],.07)
    def test_bad_inputs_never_committed(self):
        cases=[self.task(id='../escape'),self.task(id='plate'),self.task(ready=dict(method='explicit',joints_deg=[[999]*7]*2,gripper_width_m=[.1,.1])),self.task(ready=dict(method='explicit',joints_deg=[c.DEFAULT_DEG.tolist()],gripper_width_m=[.1,.1])),self.task(ready=dict(method='training_mean',states_file='../absent.npz',gripper_width_m=[.1,.1]))]
        for d in cases:
            with self.subTest(data=d),self.assertRaises((ValueError,FileNotFoundError)):self.put(d,'task')
        for d in [self.model(action_contract='delta'),self.model(checkpoint_sha256='bad'),self.model(endpoint='file:///tmp/foo'),self.model(runtime=dict(action_horizon=1,steps_per_replan=2))]:
            with self.assertRaises(ValueError):self.put(d,'model')
        self.assertFalse(list(lib.root().glob('*/*/*.json')))
    def test_protocol_horizon_identity_and_no_false_golden(self):
        d=self.model();cfg=lib.runtime_config(d);m=metadata('fixture','a'*64,cfg);validate_service(m,d)
        m['policy_sha256']='b'*64
        with self.assertRaises(ValueError):validate_service(m,d)
        self.assertEqual(actions(np.zeros((3,16)),3).shape,(3,16))
        for a in [np.zeros((16,16)),np.full((3,16),np.nan)]:
            with self.assertRaises(ValueError):actions(a,3)
        self.put(self.task(),'task');self.put(d,'model');self.assertEqual(lib.probe_paths('test_task','test_model'),(None,None))

if __name__=='__main__':unittest.main()
