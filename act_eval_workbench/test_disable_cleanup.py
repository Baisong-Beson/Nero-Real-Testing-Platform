"""Disable must remain available if opening initialization or cleanup fails."""
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase,main
from unittest.mock import patch
from . import worker as w

class Tests(TestCase):
    def test_initialization_failure_still_disables_all(self):
        with patch.object(w,'ResetIO',side_effect=RuntimeError('no feedback')),patch.object(w,'stop_services',return_value=dict(errors=[],gripper_disable_verified=True)) as disable:
            result=w.open_then_disable(Path('/unused'),None)
        disable.assert_called_once_with(True)
        self.assertFalse(result['opening']['completed']);self.assertTrue(result['errors']);self.assertTrue(result['gripper_disable_verified'])
    def test_close_failure_cannot_skip_disable(self):
        def fail():raise RuntimeError('close failed')
        io=SimpleNamespace(command_count=0,preflight=lambda:(_ for _ in ()).throw(RuntimeError('stale feedback')),close=fail)
        with patch.object(w,'ResetIO',return_value=io),patch.object(w,'stop_services',return_value=dict(errors=[],gripper_disable_verified=True)) as disable:
            result=w.open_then_disable(Path('/unused'),None)
        disable.assert_called_once_with(True);self.assertEqual(result['opening']['close_error'],'close failed')

if __name__=='__main__':main()
