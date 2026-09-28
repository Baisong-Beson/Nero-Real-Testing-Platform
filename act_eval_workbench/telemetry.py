"""Subscription-only live preview. No control publishers or service clients."""
import argparse
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .common import write
from .camera_health import FrameHealth,WRIST_TOPICS

def run(directory):
    import rclpy
    from rclpy.qos import QoSProfile,ReliabilityPolicy
    from sensor_msgs.msg import JointState,CompressedImage,Image
    from agx_arm_msgs.msg import GripperStatus
    rclpy.init();node=rclpy.create_node('nero_desktop_readonly_preview')
    entries={};grippers={};image=[None];subscriptions=[];main_camera=dict(frames_decoded=0,decode_error=False)
    wrists={side:FrameHealth(topic) for side,topic in WRIST_TOPICS.items()}
    def wrist_camera(msg,side):
        wrists[side].observe(msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,time.monotonic(),
            msg.width,msg.height,msg.step,len(msg.data),msg.encoding)
    def gripper(msg,arm):
        grippers[arm]=dict(enabled=bool(msg.driver_enable_status),width_m=float(msg.width),
            fault=bool(msg.driver_error_status),stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9)
    def joint(msg,arm):
        try:
            mapping=dict(zip(msg.name,msg.position));q=[float(mapping[f'joint{i}']) for i in range(1,8)]
            if not np.isfinite(q).all():raise ValueError('nonfinite joints')
            entries[arm]=dict(joints_deg=np.rad2deg(q).tolist(),width_m=float(mapping['gripper']),
                stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,received=time.monotonic())
        except (KeyError,ValueError):pass
    def camera(msg):
        image[0]=(bytes(msg.data),msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,time.monotonic())
    qos=QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)
    for arm in ('right','left'):
        subscriptions.append(node.create_subscription(JointState,f'/{arm}_arm/feedback/joint_states',lambda msg,a=arm:joint(msg,a),qos))
        subscriptions.append(node.create_subscription(GripperStatus,f'/{arm}_arm/feedback/gripper_status',lambda msg,a=arm:gripper(msg,a),qos))
    subscriptions.append(node.create_subscription(CompressedImage,'/zed_m/zed_node/rgb/color/rect/image/compressed',camera,qos))
    for side,topic in WRIST_TOPICS.items():
        subscriptions.append(node.create_subscription(Image,topic,lambda msg,s=side:wrist_camera(msg,s),qos))
    last=0.;last_image_stamp=None;publishers={}
    try:
        while not (directory/'STOP_TELEMETRY').exists():
            if time.time()-(directory/'heartbeat').stat().st_mtime>4:break
            rclpy.spin_once(node,timeout_sec=.02)
            now=time.monotonic()
            if now-last<.2:continue
            last=now;ros_now=node.get_clock().now().nanoseconds/1e9
            arms={a:dict(row,age_s=ros_now-row['stamp'],received_age_s=now-row['received']) for a,row in entries.items()}
            if image[0] and last_image_stamp!=image[0][1]:
                try:frame=cv2.imdecode(np.frombuffer(image[0][0],np.uint8),cv2.IMREAD_COLOR)
                except cv2.error:frame=None
                if frame is not None:
                    main_camera.update(frames_decoded=main_camera['frames_decoded']+1,decode_error=False,
                        source_unix_s=image[0][1],received_monotonic_s=image[0][2],width=int(frame.shape[1]),height=int(frame.shape[0]))
                    frame=cv2.resize(frame,(768,432));ok,data=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,80])
                    if ok:
                        temp=directory/'camera.tmp';temp.write_bytes(data.tobytes());temp.replace(directory/'camera.jpg')
                        last_image_stamp=image[0][1]
                else:main_camera['decode_error']=True
            control_topics={f'/{arm}_arm/control/{mode}' for arm in ('right','left') for mode in ('joint_states','move_c','move_j','move_js','move_l','move_mit','move_p')}
            control_topics.update(n for n,_ in node.get_topic_names_and_types() if n.startswith(('/right_arm/control/','/left_arm/control/')))
            publishers={n:node.count_publishers(n) for n in sorted(control_topics)}
            write(directory/'telemetry.json',dict(updated_unix_s=time.time(),arms=arms,camera_age_s=ros_now-image[0][1] if image[0] else None,
                main_camera=dict(main_camera,source_age_s=ros_now-main_camera.get('source_unix_s',0),received_age_s=now-main_camera.get('received_monotonic_s',0)),
                grippers={a:dict(row,age_s=ros_now-row['stamp']) for a,row in grippers.items()},
                wrist_cameras={side:health.snapshot(ros_now,now,node.count_publishers(health.topic)) for side,health in wrists.items()},
                control_publishers=publishers,mode='readonly_subscriptions',motor_enabled='not_available_from_joint_state'))
    finally:node.destroy_node();rclpy.shutdown()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True);a=p.parse_args();run(a.directory)
