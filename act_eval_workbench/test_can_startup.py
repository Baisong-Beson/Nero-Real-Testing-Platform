import copy
import json
import os
import threading
import unittest
from unittest.mock import patch
from . import can_startup as c, hardware_startup as h


def links(up=False):
    return [dict(ifname=name, link_type='can', usb_serial=serial, flags=['UP'] if up else [],
                 linkinfo=dict(info_data=dict(state='ERROR-ACTIVE' if up else 'STOPPED',
                                              bittiming=dict(bitrate=c.BITRATE) if up else {})))
            for name, serial in c.ADAPTERS.items()]


class CanStartupTests(unittest.TestCase):
    def test_replug_unconfigured(self):
        commands=[a['command'][5:] for a in c.plan(links())]
        self.assertEqual(commands,[['type','can','bitrate','1000000'],['up']]*2)

    def test_repeated_start_does_not_cycle_healthy_bus(self):
        self.assertEqual(c.plan(links(True)),[])

    def test_one_down_only_touches_that_adapter(self):
        rows=links(True);rows[1]=links()[1]
        self.assertEqual([a['interface'] for a in c.plan(rows)],['can1','can1'])

    def test_swapped_down_names_use_temporary_names(self):
        rows=links();rows[0]['ifname']='can1';rows[1]['ifname']='can0'
        actions=c.plan(rows)
        for action in actions:
            row=next(r for r in rows if r['ifname']==action['interface'])
            self.assertEqual(row['usb_serial'],action['serial'])
            args=action['command'][5:]
            if args[0]=='name':
                self.assertNotIn(args[1],[r['ifname'] for r in rows]);row['ifname']=args[1]
            elif args[0]=='type':row['linkinfo']['info_data']['bittiming']={'bitrate':c.BITRATE}
            elif args[0]=='up':row['flags']=['UP'];row['linkinfo']['info_data']['state']='ERROR-ACTIVE'
        self.assertEqual(c.plan(rows),[])

    def test_missing_or_duplicate_adapter_refused(self):
        for rows in [[],links()[:1],links()+links()[:1]]:
            with self.assertRaises(RuntimeError):c.plan(rows)

    def test_unknown_target_occupant_refused(self):
        rows=links();rows[0]['ifname']='can2';rows.append(dict(ifname='can0',link_type='ether'))
        with self.assertRaises(RuntimeError):c.plan(rows)

    def test_running_wrong_rate_wrong_mapping_or_busoff_refused(self):
        for mode in ('rate','mapping','busoff'):
            rows=links(True)
            if mode=='rate':rows[0]['linkinfo']['info_data']['bittiming']['bitrate']=500000
            if mode=='mapping':rows[0]['ifname']='can2'
            if mode=='busoff':rows[0]['linkinfo']['info_data']['state']='BUS-OFF'
            with self.assertRaises(RuntimeError):c.plan(rows)

    def test_hot_unplug_during_setup_cannot_mutate_replacement(self):
        with patch.object(c,'inventory',side_effect=[links(),[]]),patch.object(c.subprocess,'run') as run:
            with self.assertRaisesRegex(RuntimeError,'变化'):c.activate()
            run.assert_not_called()

    def test_activation_checks_postcondition(self):
        with patch.object(c,'inventory',return_value=links()),patch.object(c.subprocess,'run'):
            with self.assertRaisesRegex(RuntimeError,'复查'):c.activate()

    def test_full_down_up_and_repeat(self):
        rows=links()
        def fake_run(command,**kwargs):
            row=next(r for r in rows if r['ifname']==command[4])
            args=command[5:]
            if args[0]=='type':row['linkinfo']['info_data']['bittiming']={'bitrate':c.BITRATE}
            else:row['flags']=['UP'];row['linkinfo']['info_data']['state']='ERROR-ACTIVE'
        with patch.object(c,'inventory',side_effect=lambda:copy.deepcopy(rows)),patch.object(c.subprocess,'run',side_effect=fake_run):
            self.assertEqual(len(c.activate()['commands']),4)
            self.assertEqual(c.activate()['commands'],[])

    def test_isolated_tests_never_touch_real_hardware(self):
        for env in [{'ACT_EVAL_FAKE_DRIVER':'1'}, {'ROS_DOMAIN_ID':'174'}]:
            with patch.dict(os.environ,env,clear=True),patch.object(c,'inventory',side_effect=AssertionError('real hardware')):
                self.assertTrue(h.recover(lambda _:None,threading.Event())['skipped'])

    def test_permission_missing_report_is_actionable(self):
        with patch.object(h,'isolated',return_value=False),patch.object(h,'camera_usb_status',return_value={'message':'USB missing'}),patch.object(c,'inventory',return_value=links()),patch.object(h.Path,'is_file',return_value=False):
            result=h.recover(lambda _:None,threading.Event(),allow_install=False)
        self.assertFalse(result['ok']);self.assertIn('管理员授权',result['error'])

    def test_healthy_startup_does_not_request_admin(self):
        with patch.object(h,'isolated',return_value=False),patch.object(h,'camera_usb_status',return_value={'message':'USB missing'}),patch.object(c,'inventory',return_value=links(True)),patch.object(h,'run_command') as run:
            self.assertTrue(h.recover(lambda _:None,threading.Event())['ok']);run.assert_not_called()

    def test_ssh_gui_without_auth_agent_uses_local_terminal(self):
        with patch.object(h,'run_command',side_effect=[(127,'','Error creating textual authentication agent: /dev/tty'),(0,'','')]) as run:
            h.install_helper(lambda _:None,threading.Event())
            self.assertEqual(run.call_args_list[1].args[0][0],'/usr/bin/gnome-terminal')
            self.assertIn('/usr/bin/sudo',run.call_args_list[1].args[0])

    def test_cancelled_authorization_does_not_open_second_prompt(self):
        with patch.object(h,'run_command',return_value=(126,'','Authentication cancelled')) as run:
            with self.assertRaises(RuntimeError):h.install_helper(lambda _:None,threading.Event())
            self.assertEqual(run.call_count,1)


if __name__=='__main__':unittest.main()
