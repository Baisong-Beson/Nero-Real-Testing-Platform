"""Recognize stable force contact; contact does not identify the held object."""
from collections import deque
import math

CONTACT_FORCE_N=.5
SETTLE_S=.3
WIDTH_STABILITY_M=.0005

class ContactFeedback:
    def __init__(self):self.rows=deque(maxlen=300)
    def add(self,stamp,received,width,force,enabled,fault):
        if not all(math.isfinite(x) for x in (stamp,received,width,force)) or not -.001<=width<=.101:
            self.rows.clear();return
        if self.rows and stamp<=self.rows[-1]['stamp']:return
        self.rows.append(dict(stamp=stamp,received=received,width_m=width,force_n=force,enabled=bool(enabled),fault=bool(fault)))
    def contact(self,now,ros_now,since,measured_width):
        if not self.rows:return None
        last=self.rows[-1]
        if not 0<=now-last['received']<=.2 or not -.05<=ros_now-last['stamp']<=.2:
            return None
        # Demand a continuous post-command interval of fresh contact feedback.
        selected=[]
        for row in reversed(self.rows):
            if row['received']<since or not row['enabled'] or row['fault'] or abs(row['force_n'])<CONTACT_FORCE_N:break
            if selected and selected[-1]['stamp']-row['stamp']>.1:break
            selected.append(row)
            if last['stamp']-row['stamp']>=SETTLE_S:break
        if len(selected)<3 or last['stamp']-selected[-1]['stamp']<SETTLE_S:return None
        widths=[row['width_m'] for row in selected]
        if max(widths)-min(widths)>WIDTH_STABILITY_M or abs(last['width_m']-measured_width)>.002:return None
        # Do not accept an unchanged fully open claw as a successful closure.
        if not .002<measured_width<.095:return None
        return dict(mode='force_contact',width_m=last['width_m'],force_n=last['force_n'],
            stable_s=last['stamp']-selected[-1]['stamp'],object_presence_verified=False)
