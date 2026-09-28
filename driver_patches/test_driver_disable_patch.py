"""Test exact patched driver methods without importing ROS/SDK or CAN."""
import ast
from pathlib import Path
import sys
from types import SimpleNamespace as NS
import unittest

source=Path(sys.argv.pop(1))
tree=ast.parse(source.read_text(encoding='utf-8'))
methods=[node for cls in tree.body if isinstance(cls,ast.ClassDef) for node in cls.body
         if isinstance(node,ast.FunctionDef) and node.name in ('_disable_gripper_verified','_enable_callback')]
assert len(methods)==2

class Clock:
    def __init__(self):self.now=100.
    def time(self):return self.now
    def monotonic(self):return self.now
    def sleep(self,dt):self.now+=dt

class Tests(unittest.TestCase):
    def setUp(self):
        self.clock=Clock();env={'time':self.clock}
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),str(source),'exec'),env)
        self.calls=[];self.mode='normal';self.arm_ok=True
        def status():
            return NS(timestamp=self.clock.time() if self.mode!='stale' else 99.,
                      driver_enable_status=self.mode=='stuck' or len(self.calls)<2)
        def disable():self.calls.append('disable');return NS(driver_enable_status=True)
        self.obj=NS(effector_type='agx_gripper',gripper=NS(disable=disable,get_status=status),control_enabled=True)
        self.obj.get_logger=lambda:NS(info=lambda *_:None,error=lambda *_:None,warn=lambda *_:None)
        self.obj._check_arm_ready=lambda:self.arm_ok
        self.obj._enable_arm=lambda enable:True
        self.obj._disable_gripper_verified=lambda:env['_disable_gripper_verified'](self.obj)
        self.callback=lambda:env['_enable_callback'](self.obj,NS(data=False),NS(success=None,message=''))
    def test_new_feedback_required_cached_return_ignored(self):
        res=self.callback();self.assertTrue(res.success);self.assertGreaterEqual(len(self.calls),2)
        self.assertFalse(self.obj.control_enabled)
    def test_stale_false_cannot_pass(self):
        self.mode='stale';res=self.callback();self.assertFalse(res.success)
    def test_still_enabled_cannot_pass(self):
        self.mode='stuck';res=self.callback();self.assertFalse(res.success)
    def test_arm_unavailable_still_disables_gripper(self):
        self.arm_ok=False;res=self.callback();self.assertFalse(res.success);self.assertGreaterEqual(len(self.calls),2)
        self.assertIn('gripper disabled=True',res.message)
    def test_arm_exception_still_disables_gripper(self):
        def fail(_):raise RuntimeError('arm failure')
        self.obj._enable_arm=fail;res=self.callback();self.assertFalse(res.success);self.assertGreaterEqual(len(self.calls),2)
        self.assertIn('gripper disabled=True',res.message)

if __name__=='__main__':unittest.main()
