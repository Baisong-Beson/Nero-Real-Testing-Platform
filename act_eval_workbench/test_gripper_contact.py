import unittest
from .gripper_contact import ContactFeedback

class Tests(unittest.TestCase):
    def feed(self,force=1.,enabled=True,fault=False,width=.04205):
        f=ContactFeedback()
        for i in range(101):f.add(1000+i*.01,i*.01,width,force,enabled,fault)
        return f
    def test_fresh_stable_force_contact(self):
        row=self.feed().contact(1.,1001.,.5,.04205)
        self.assertEqual(row['mode'],'force_contact');self.assertFalse(row['object_presence_verified'])
    def test_force_disabled_fault_open_do_not_count(self):
        for kwargs in [dict(force=0.),dict(enabled=False),dict(fault=True),dict(width=.1)]:
            with self.subTest(kwargs=kwargs):self.assertIsNone(self.feed(**kwargs).contact(1.,1001.,.5,kwargs.get('width',.04205)))
    def test_stale_duplicate_precommand_and_moving_rejected(self):
        f=self.feed()
        self.assertIsNone(f.contact(1.3,1001.3,.5,.04205))
        self.assertIsNone(f.contact(1.,1001.,.9,.04205))
        self.assertIsNone(f.contact(1.,1001.,.5,.02))
        f=ContactFeedback()
        for i in range(100):f.add(1000.,i*.01,.04205,1.,True,False)
        self.assertIsNone(f.contact(1.,1001.,0.,.04205))
        f=ContactFeedback()
        for i in range(101):f.add(1000+i*.01,i*.01,.06-i*.0001,1.,True,False)
        self.assertIsNone(f.contact(1.,1001.,.5,.05))
    def test_interrupted_contact_must_settle_again(self):
        f=self.feed();f.add(1001.01,1.01,.04205,0.,True,False)
        f.add(1001.02,1.02,.04205,1.,True,False)
        self.assertIsNone(f.contact(1.02,1001.02,.5,.04205))

if __name__=='__main__':unittest.main()
