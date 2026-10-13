"""Read-only wrist image stream health, independent of robot control."""
from collections import deque
import math

WRIST_TOPICS={side:f'/{side}_wrist/color/image_raw' for side in ('left','right')}
COLORS={'ok':'#198754','warning':'#bc7900','error':'#c73646','unknown':'#7b8794'}

class FrameHealth:
    def __init__(self,topic):
        self.topic=topic;self.stamp=None;self.received=None;self.arrivals=deque(maxlen=120)
        self.frames=0;self.error=None;self.width=None;self.height=None
    def observe(self,stamp,received,width,height,step,size,encoding):
        channels={'rgb8':3,'bgr8':3,'rgba8':4,'bgra8':4,'mono8':1,'mono16':2,'16UC1':2,'yuv422':2}
        if (not math.isfinite(stamp) or not math.isfinite(received) or width<=0 or height<=0
                or encoding not in channels or step<width*channels[encoding] or size<step*height):
            self.error='图像数据异常';return
        if self.stamp is not None and stamp<self.stamp:self.error='时间戳回退';return
        if self.stamp is not None and stamp==self.stamp:return
        self.stamp=stamp;self.received=received;self.width=width;self.height=height
        self.frames+=1;self.error=None;self.arrivals.append(received)
    def snapshot(self,ros_now,now,publishers):
        recent=[t for t in self.arrivals if now-t<=2.]
        fps=(len(recent)-1)/(recent[-1]-recent[0]) if len(recent)>1 and recent[-1]>recent[0] else 0.
        age=None if self.stamp is None else ros_now-self.stamp
        received_age=None if self.received is None else now-self.received
        if publishers==0:health,label='error','未连接'
        elif publishers!=1:health,label='error','发布源冲突'
        elif self.error:health,label='error',self.error
        elif age is None:health,label='warning','等待图像'
        elif age<-.1 or received_age<0:health,label='error','时钟异常'
        elif max(age,received_age)>1.5:health,label='error','图像中断'
        elif max(age,received_age)>.5:health,label='warning','图像延迟'
        elif fps<5:health,label='warning','帧率偏低' if len(recent)>1 else '检测中'
        else:health,label='ok','通畅'
        return dict(topic=self.topic,health=health,label=label,age_s=age,received_age_s=received_age,
                    fps=fps,frames=self.frames,publishers=publishers,width=self.width,height=self.height)

def indicator(row,telemetry_age_s):
    if not row or not -.1<=telemetry_age_s<=.8:return COLORS['unknown'],'监视未连接'
    health=row.get('health','unknown');label=row.get('label','检测中')
    if health in ('ok','warning') and row.get('fps',0)>0:label+=f' · {row["fps"]:.1f} fps'
    return COLORS.get(health,COLORS['unknown']),label
