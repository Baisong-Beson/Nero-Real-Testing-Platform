"""Production telemetry + JPEG decode/stall/recovery, localhost ROS domain 174."""
import os
import subprocess
import sys
import time
import cv2
import numpy as np
from .common import new_session, read, write
from .camera_reconnect import fresh, TOPIC


def main():
    if os.environ.get('ROS_DOMAIN_ID')!='174' or os.environ.get('ROS_LOCALHOST_ONLY')!='1':
        raise RuntimeError('isolated ROS 174 required')
    import rclpy
    from sensor_msgs.msg import CompressedImage
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    directory=new_session('main_camera_simulation');heart=directory/'heartbeat';heart.touch()
    rclpy.init();node=rclpy.create_node('synthetic_main_camera')
    pub=node.create_publisher(CompressedImage,TOPIC,QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT))
    pixels=np.full((48,64,3),(10,100,200),np.uint8);ok,encoded=cv2.imencode('.jpg',pixels);assert ok
    log=(directory/'telemetry.log').open('w')
    proc=subprocess.Popen([sys.executable,'-m','act_eval_workbench.telemetry','--directory',str(directory)],stdout=log,stderr=subprocess.STDOUT)
    def drive(seconds,mode='good'):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            heart.touch()
            if mode!='pause':
                msg=CompressedImage();msg.header.stamp=node.get_clock().now().to_msg();msg.format='jpeg'
                msg.data=encoded.tobytes() if mode=='good' else b''
                pub.publish(msg)
            rclpy.spin_once(node,timeout_sec=.001);time.sleep(.065)
        assert proc.poll() is None,'telemetry crashed'
        return read(directory/'telemetry.json')
    try:
        healthy=drive(3.5);assert fresh(healthy),healthy
        assert healthy['main_camera']['frames_decoded']>=3 and (directory/'camera.jpg').exists()
        invalid=drive(.8,'bad');assert not fresh(invalid) and invalid['main_camera']['decode_error'],invalid
        restored=drive(1.2);assert fresh(restored),restored
        stalled=drive(1.2,'pause');assert not fresh(stalled),stalled
        recovered=drive(1.2);assert fresh(recovered),recovered
        assert len(recovered['control_publishers'])>=14 and sum(recovered['control_publishers'].values())==0
        write(directory/'summary.json',dict(passed=True,physical_motion_executed=False,
            healthy=healthy['main_camera'],invalid=invalid['main_camera'],stalled=stalled['main_camera'],recovered=recovered['main_camera']))
        print('MAIN_CAMERA_STREAM_TEST_PASSED',directory,flush=True)
    finally:
        (directory/'STOP_TELEMETRY').touch()
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=3)
        log.close();node.destroy_node();rclpy.shutdown()


if __name__=='__main__':main()
