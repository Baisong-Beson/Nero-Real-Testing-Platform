"""Convert fresh ROS wrist frames to the common RGB inference contract."""
import numpy as np
import cv2
from .library import CAMERAS

def wrist_images(required,frames,now,ros_now,main_stamp):
    result={}
    for key in required:
        if key=='main':continue
        if key not in frames:raise RuntimeError('模型需要相机但尚未收到图像：'+key)
        msg,received=frames[key];stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
        if not -.05<=ros_now-stamp<=.5 or not 0<=now-received<=.5 or abs(stamp-main_stamp)>.25:raise RuntimeError('模型相机图像过期或不同步：'+key)
        channels={'rgb8':3,'bgr8':3,'rgba8':4,'bgra8':4,'mono8':1}
        if msg.encoding not in channels:raise ValueError('暂不支持腕部相机编码：'+msg.encoding)
        n=channels[msg.encoding]
        if min(msg.height,msg.width)<1 or msg.step<msg.width*n or len(msg.data)<msg.height*msg.step:raise ValueError('腕部相机图像尺寸无效')
        a=np.frombuffer(bytes(msg.data),np.uint8)[:msg.height*msg.step].reshape(msg.height,msg.step)[:,:msg.width*n].reshape(msg.height,msg.width,n)
        if msg.encoding=='rgb8':rgb=a.copy()
        else:rgb=cv2.cvtColor(a,{'bgr8':cv2.COLOR_BGR2RGB,'rgba8':cv2.COLOR_RGBA2RGB,'bgra8':cv2.COLOR_BGRA2RGB,'mono8':cv2.COLOR_GRAY2RGB}[msg.encoding])
        result[CAMERAS[key]]=rgb
    return result
