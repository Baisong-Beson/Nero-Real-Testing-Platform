import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from . import common as m, storage as s


class Tests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)
        for name,value in [('PREFERENCES',self.root/'config/preferences.json'),('DEFAULT_EXPORTS',self.root/'exports')]:
            p=patch.object(m,name,value);p.start();self.addCleanup(p.stop)

    def test_first_save_and_restarted_process_preference(self):
        self.assertEqual(s.initial_export_directory(),self.root/'exports')
        chosen=s.create_folder(self.root,'中文 path');s.remember_export_directory(chosen)
        self.assertEqual(s.initial_export_directory(),chosen)
        self.assertEqual(m.read(m.PREFERENCES)['last_export_directory'],str(chosen))

    def test_missing_and_unwritable_previous_fall_back(self):
        chosen=s.create_folder(self.root,'disk');s.remember_export_directory(chosen);chosen.rmdir()
        self.assertEqual(s.initial_export_directory(),self.root/'exports')
        chosen.mkdir();real=s.usable_directory
        with patch.object(s,'usable_directory',side_effect=lambda path:False if Path(path)==chosen else real(path)):
            self.assertEqual(s.initial_export_directory(),self.root/'exports')
        self.assertEqual(m.read(m.PREFERENCES)['last_export_directory'],str(chosen))

    def test_corrupt_preferences_recover_and_other_keys_survive(self):
        m.PREFERENCES.parent.mkdir();m.PREFERENCES.write_text('{broken')
        self.assertEqual(s.initial_export_directory(),self.root/'exports')
        m.write(m.PREFERENCES,{'unrelated_setting':42});s.remember_export_directory(self.root)
        self.assertEqual(s.preferences()['unrelated_setting'],42)

    def test_folder_creation_collision_and_invalid_names(self):
        folder=s.create_folder(self.root,'trial 01');self.assertTrue(folder.is_dir())
        with self.assertRaises(FileExistsError):s.create_folder(self.root,'trial 01')
        for name in ('','..','.', '../outside','a/b','a\\b'):
            with self.assertRaises(ValueError):s.create_folder(self.root,name)

    def test_invalid_selection_does_not_change_preference(self):
        s.remember_export_directory(self.root);before=m.PREFERENCES.read_bytes()
        with self.assertRaises(OSError):s.remember_export_directory(self.root/'missing')
        self.assertEqual(m.PREFERENCES.read_bytes(),before)

    def test_defaults_all_belong_to_platform(self):
        for path in (m.ROOT,m.EXPORT,m.RUNS):self.assertTrue(path.is_relative_to(m.PLATFORM))

    def test_task_model_survive_restart_and_export_save(self):
        s.remember_export_directory(self.root);s.remember_selection('banana','g2')
        self.assertEqual(s.initial_selection(),dict(task='banana',model='g2'))
        self.assertEqual(s.initial_export_directory(),self.root)
        s.remember_export_directory(self.root)
        self.assertEqual(s.initial_selection(),dict(task='banana',model='g2'))
        self.assertEqual(m.ready_gripper_target(s.initial_selection()['task']),[0.,.1])
        s.remember_selection('plate','p0')
        self.assertEqual(m.ready_gripper_target(s.initial_selection()['task']),[.1,.1])

    def test_invalid_saved_selection_falls_back_without_corrupting_preferences(self):
        for value in [None,[],{'task':'unknown','model':'unknown'}]:
            m.write(m.PREFERENCES,dict(last_selection=value,last_export_directory=str(self.root)))
            self.assertEqual(s.initial_selection(),dict(task='plate',model='mpi-base'))
        before=m.PREFERENCES.read_bytes()
        with self.assertRaises(ValueError):s.remember_selection('banana','unknown')
        self.assertEqual(m.PREFERENCES.read_bytes(),before)


if __name__=='__main__':unittest.main()
