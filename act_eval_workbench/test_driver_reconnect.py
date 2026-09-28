import os
from pathlib import Path
import threading
import tempfile
import json
import time
import unittest
from unittest.mock import patch, Mock
from . import driver_reconnect as d


class DriverReconnectTests(unittest.TestCase):
    def test_isolated_environment_never_scans_processes(self):
        with patch.dict(os.environ,{'ACT_EVAL_FAKE_DRIVER':'1'}),patch.object(d,'matching_drivers',side_effect=AssertionError('process scan')):
            self.assertTrue(d.recover(Path('/missing'),lambda:{},threading.Event(),print)['skipped'])

    def test_healthy_feedback_does_not_restart(self):
        state=dict(updated_unix_s=time.time(),arms={s:dict(age_s=.01) for s in ['right','left']})
        with patch.dict(os.environ,{},clear=True),patch.object(d,'matching_drivers',side_effect=AssertionError('restart')):
            result=d.recover(Path('/missing'),lambda:state,threading.Event(),print)
        self.assertTrue(result['ok']);self.assertEqual(result['restarted'],[])

    def test_stale_state_is_not_healthy(self):
        state=dict(updated_unix_s=time.time()-5,arms=dict(right=dict(age_s=.01)))
        self.assertEqual(d.fresh_arms(state),set())

    def test_active_publishers_and_incomplete_discovery_block_restart(self):
        for publishers in [{},dict.fromkeys(range(14),1)]:
            with self.assertRaises(RuntimeError):d.assert_idle(dict(updated_unix_s=time.time(),control_publishers=publishers))
        d.assert_idle(dict(updated_unix_s=time.time(),control_publishers=dict.fromkeys(range(14),0)))

    def test_recipe_cannot_auto_enable_or_swap_namespace(self):
        command=['/usr/bin/python3','/installed/agx_arm_ctrl_single','--ros-args','-r','__ns:=/right_arm',
                 '-p','auto_enable:=false','-p','control_enabled:=false']
        d.validate_command(command,'right')
        for bad in [command[:-2],command[:-1]+['control_enabled:=true']]:
            with self.assertRaises(RuntimeError):d.validate_command(bad,'right')
        with self.assertRaises(RuntimeError):d.validate_command(command,'left')

    def test_dead_readers_replaced_and_feedback_verified(self):
        for confirmed in [True,False,'needs_kill']:
            with tempfile.TemporaryDirectory() as directory,patch.dict(os.environ,{},clear=True):
                root=Path(directory);(root/'config').mkdir();(root/'logs').mkdir()
                recipes={s:dict(command=['/usr/bin/python3','/installed/agx_arm_ctrl_single','--ros-args','-r',f'__ns:=/{s}_arm','-p','auto_enable:=false','-p','control_enabled:=false'],environment={}) for s in ['right','left']}
                (root/'config/arm_driver_reconnect.json').write_text(json.dumps(recipes))
                old={'right':1234,'left':1235};arms={};clock=[0]
                def tick():clock[0]+=2;return clock[0]
                def matches(command,side):return [(old[side],command)] if side in old else []
                def kill(pid,sig):
                    if confirmed=='needs_kill' and sig==d.signal.SIGTERM:return
                    old.pop(next(s for s,p in old.items() if p==pid))
                def start(command,**kwargs):
                    side='right' if '__ns:=/right_arm' in command else 'left'
                    arms[side]=dict(age_s=.01);return Mock(pid=2000+len(arms))
                def state():return dict(updated_unix_s=time.time(),arms=arms,control_publishers=dict.fromkeys(map(str,range(14)),0))
                with patch.object(d.time,'monotonic',side_effect=tick),patch.object(d,'matching_drivers',side_effect=matches),patch.object(d,'dead_receive_thread',return_value=confirmed),patch.object(d.os,'kill',side_effect=kill) as terminate,patch.object(d.subprocess,'Popen',side_effect=start) as launch,patch.object(d.fcntl,'flock'):
                    result=d.recover(root,state,threading.Event(),lambda _:None)
                self.assertEqual(result['ok'],bool(confirmed))
                self.assertEqual(launch.call_count,2 if confirmed else 0)
                self.assertEqual(terminate.call_count,4 if confirmed=='needs_kill' else 2 if confirmed else 0)


if __name__=='__main__':unittest.main()
