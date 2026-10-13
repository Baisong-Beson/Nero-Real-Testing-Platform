"""Verify live topic reassignment on a strictly isolated ROS domain."""
import os
import subprocess
import sys
import time
import cv2
import numpy as np
from .common import new_session, read, write


def main():
    if os.environ.get('ROS_DOMAIN_ID')!='174' or os.environ.get('ROS_LOCALHOST_ONLY')!='1':
        raise RuntimeError('isolated ROS 174 required')
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import Image
    directory=new_session('camera_assignment_simulation');heart=directory/'heartbeat';heart.touch()
    config=directory/'bindings.json'
    red=dict(topic='/test_camera_red/image',transport='raw')
    blue=dict(topic='/test_camera_blue/image',transport='raw')
    write(config,dict(roles={'main':red,'right':blue,'left':None}))
    rclpy.init();node=rclpy.create_node('synthetic_camera_assignment')
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
    pubs=[node.create_publisher(Image,row['topic'],qos) for row in (red,blue)]
    log=(directory/'telemetry.log').open('w')
    proc=subprocess.Popen([sys.executable,'-m','act_eval_workbench.telemetry','--directory',str(directory),'--camera-config',str(config)],stdout=log,stderr=subprocess.STDOUT)
    def drive(seconds):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            heart.touch()
            for pub,color in zip(pubs,[(240,0,0),(0,0,240)]):
                msg=Image();msg.header.stamp=node.get_clock().now().to_msg()
                msg.width=64;msg.height=48;msg.step=192;msg.encoding='rgb8'
                msg.data=np.full((48,64,3),color,np.uint8).tobytes();pub.publish(msg)
            rclpy.spin_once(node,timeout_sec=.001);time.sleep(.065)
        assert proc.poll() is None,'telemetry exited'
        return read(directory/'telemetry.json')
    try:
        before=drive(4)
        assert np.allclose(cv2.imread(str(directory/'camera.jpg'))[0,0],(0,0,240),atol=3)
        assert before['wrist_cameras']['left']['label']=='未分配'
        write(config,dict(roles={'main':blue,'right':red,'left':None}))
        after=drive(3)
        assert after['main_camera']['topic']==blue['topic']
        assert np.allclose(cv2.imread(str(directory/'camera.jpg'))[0,0],(240,0,0),atol=3)
        assert np.allclose(cv2.imread(str(directory/'camera_right.jpg'))[0,0],(0,0,240),atol=3)
        write(config,dict(roles={'main':None,'right':red,'left':None}))
        disabled=drive(2)
        assert disabled['main_camera']['frames_decoded']==0
        assert not (directory/'camera.jpg').exists()
        assert sum(disabled['control_publishers'].values())==0
        write(directory/'summary.json',dict(passed=True,physical_motion_executed=False,
            before=before['main_camera'],after=after['main_camera'],disabled=disabled['main_camera']))
        print('CAMERA_ASSIGNMENT_STREAM_TEST_PASSED',directory)
    finally:
        (directory/'STOP_TELEMETRY').touch()
        try:proc.wait(timeout=4)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=4)
        log.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
