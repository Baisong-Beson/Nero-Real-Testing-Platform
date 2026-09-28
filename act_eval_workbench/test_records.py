"""Record operations run only in temporary folders; never edit real experiments."""
import csv
import tempfile
from pathlib import Path
import unittest
from . import common as m
from . import records as r

class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)/'runs';self.output=Path(self.temp.name)/'exports'
        for name,kind,started,verdict,simulation,protocol in [
            ('a','formal',True,True,False,'p1'),('b','formal',True,False,False,'p1'),
            ('c','formal',True,None,False,'p2'),('d','ready',True,True,False,'p1'),
            ('e','formal',True,True,True,'p1'),('f','formal',False,None,False,'p1')]:
            m.write(self.root/name/'result.json',dict(kind=kind,started=started,runtime_completed=False,settings=m.Settings(operator='=2+2').json(),
                protocol_sha256=protocol,adjudication=dict(success=verdict,reason='中文依据'),simulation_only=simulation,
                video=dict(quality='minor_gaps',gap_warning_count=1,max_source_gap_s=.2666)))
            (self.root/name/'main_camera.avi').write_bytes(b'original evidence')
        m.write(self.root/'g_default'/'job_error.json',dict(error='prepare failed'))
        m.write(self.root/'g_default'/'config.json',dict(settings=m.Settings().json()))
        for name,kind in [('h','shadow'),('i','offline_simulation'),('j','disable'),('k','default'),('l','micro')]:
            m.write(self.root/name/'result.json',dict(kind=kind,started=True,runtime_completed=True))
        m.write(self.root/'m'/'result.json',dict(kind='formal',started=True,physical_motion_executed=False))
    def tearDown(self):self.temp.cleanup()
    def test_clear_restore_retains_all_bytes_and_updates_statistics(self):
        before={str(p):p.read_bytes() for p in self.root.glob('*/*')}
        self.assertEqual(len(r.scan(self.root)),3)
        r.set_archived(self.root,['a','b'],True,'test')
        self.assertEqual(len(r.scan(self.root)),1);self.assertEqual(len(r.scan(self.root,True)),3)
        groups=r.summarize(r.scan(self.root));self.assertEqual(len(groups),1);self.assertEqual(groups[0]['pending'],1)
        r.set_archived(self.root,['a','b'],False,'test')
        p1=next(g for g in r.summarize(r.scan(self.root)) if g['protocol']=='p1')
        self.assertEqual((p1['started'],p1['success'],p1['failed']),(2,1,1))
        for path,data in before.items():self.assertEqual(Path(path).read_bytes(),data)
        self.assertEqual(len(r.archive_state(self.root)['history']),2)
    def test_export_selection_utf8_formula_protection_full_json_and_hashes(self):
        out=r.export(self.root,self.output,['a','b'])
        self.assertTrue((out/'records.csv').read_bytes().startswith(b'\xef\xbb\xbf'))
        with (out/'records.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
        a=next(row for row in rows if row['记录编号']=='a')
        self.assertEqual(a['操作员'],"'=2+2");self.assertEqual(a['依据'],'中文依据');self.assertEqual(a['录像间隔提示数'],'1')
        data=m.read(out/'records.json');self.assertEqual(len(data['records']),2)
        self.assertEqual(data['record_scope'],'real_started_formal')
        self.assertEqual(data['groups'][0]['started'],2)
        self.assertEqual(data['records'][-1]['result']['settings']['operator'],'=2+2')
        for name,sha in m.read(out/'manifest.json')['files'].items():self.assertEqual(m.sha(out/name),sha)
        self.assertFalse((out/'main_camera.avi').exists())
    def test_export_all_excludes_cleared_and_preserves_pending_group(self):
        r.set_archived(self.root,['a'],True,'test');out=r.export(self.root,self.output)
        data=m.read(out/'records.json');self.assertEqual(len(data['records']),2)
        self.assertNotIn('a',[row['id'] for row in data['records']]);self.assertEqual(len(data['groups']),2)
    def test_bad_or_missing_selection_does_not_change_index(self):
        with self.assertRaises(ValueError):r.set_archived(self.root,['../other'],True,'test')
        self.assertFalse((self.root/'record_archive.json').exists())
        with self.assertRaises(ValueError):r.export(self.root,self.output,['missing'])
        self.assertFalse(self.output.exists())
    def test_only_started_real_model_execution_is_listed_and_exportable(self):
        self.assertEqual({row['id'] for row in r.scan(self.root)},{'a','b','c'})
        self.assertEqual(len(r.scan(self.root,include_diagnostics=True)),13)
        self.assertEqual(sum(g['started'] for g in r.summarize(r.scan(self.root,include_diagnostics=True))),3)
        for name in ('d','e','f','g_default','h','i','j','k','l','m'):
            with self.assertRaises(ValueError):r.export(self.root,self.output,[name])
            with self.assertRaises(ValueError):r.set_archived(self.root,[name],True,'test')

if __name__=='__main__':unittest.main()
