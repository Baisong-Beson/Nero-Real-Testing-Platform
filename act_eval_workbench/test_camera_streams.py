"""Exercise production telemetry on localhost ROS 174 with synthetic images."""
import os
import subprocess
import sys
import time
from .common import new_session,read,write

def main():
    if os.environ.get('ROS_DOMAIN_ID')!='174' or os.environ.get('ROS_LOCALHOST_ONLY')!='1':raise RuntimeError('isolated ROS 174 required')
    import rclpy
    from sensor_msgs.msg import Image
    from rclpy.qos import QoSProfile,ReliabilityPolicy
    directory=new_session('wrist_camera_simulation');heart=directory/'heartbeat';heart.touch()
    rclpy.init();node=rclpy.create_node('synthetic_wrist_cameras')
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
    pubs={a:node.create_publisher(Image,f'/{a}_wrist/color/image_raw',qos) for a in ('left','right')}
    log=(directory/'telemetry.log').open('w');proc=subprocess.Popen([sys.executable,'-m','act_eval_workbench.telemetry','--directory',str(directory)],stdout=log,stderr=subprocess.STDOUT)
    def drive(seconds,right=True):
        deadline=time.monotonic()+seconds
        while time.monotonic()<deadline:
            heart.touch();msg=Image();msg.header.stamp=node.get_clock().now().to_msg()
            msg.width=64;msg.height=48;msg.encoding='rgb8';msg.step=192;msg.data=bytes(9216)
            pubs['left'].publish(msg)
            if right:pubs['right'].publish(msg)
            rclpy.spin_once(node,timeout_sec=.001);time.sleep(.065)
        return read(directory/'telemetry.json')
    try:
        healthy=drive(4)
        assert all(healthy['wrist_cameras'][a]['health']=='ok' for a in pubs),healthy
        stalled=drive(2.2,False)
        assert stalled['wrist_cameras']['left']['health']=='ok',stalled
        assert stalled['wrist_cameras']['right']['label']=='图像中断',stalled
        recovered=drive(2.2)
        assert all(recovered['wrist_cameras'][a]['health']=='ok' for a in pubs),recovered
        assert sum(recovered['control_publishers'].values())==0
        write(directory/'summary.json',dict(passed=True,physical_motion_executed=False,healthy=healthy['wrist_cameras'],
            stalled=stalled['wrist_cameras'],recovered=recovered['wrist_cameras']))
        print('WRIST_STREAM_TEST_PASSED',directory,flush=True)
    finally:
        (directory/'STOP_TELEMETRY').touch()
        try:proc.wait(timeout=3)
        except subprocess.TimeoutExpired:proc.terminate();proc.wait(timeout=3)
        log.close();node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
