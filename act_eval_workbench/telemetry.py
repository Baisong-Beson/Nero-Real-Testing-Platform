"""Subscription-only multi-camera preview. No robot control endpoints."""
import argparse
import os
import json
from pathlib import Path
import time
import cv2
import numpy as np
from .common import write, PLATFORM
from .camera_health import FrameHealth, WRIST_TOPICS
from .camera_preview import decode_raw, write_preview, VIEWS
from .camera_devices import load_config

DEFAULT_SOURCES = {
    'main': dict(topic='/zed_m/zed_node/rgb/color/rect/image/compressed', transport='compressed'),
    **{side: dict(topic=topic, transport='raw') for side, topic in WRIST_TOPICS.items()}
}


def camera_sources(camera_config=None):
    if camera_config is not None:return json.loads(camera_config.read_text()).get('roles',{})
    if os.environ.get('ROS_DOMAIN_ID','0')!='0':return DEFAULT_SOURCES
    config = load_config(PLATFORM)
    return config['roles'] if config.get('roles') else DEFAULT_SOURCES


def run(directory, camera_config=None):
    import rclpy
    from rclpy.qos import QoSProfile, ReliabilityPolicy
    from sensor_msgs.msg import JointState, CompressedImage, Image
    from agx_arm_msgs.msg import GripperStatus
    rclpy.init()
    node = rclpy.create_node('nero_desktop_readonly_preview')
    qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
    entries = {}; grippers = {}; subscriptions = []
    sources = {}; camera_subs = {}; images = {}; processed = {}; health = {}; previews = {}
    source_check = 0.; last = 0.

    def gripper(msg, arm):
        grippers[arm] = dict(enabled=bool(msg.driver_enable_status), width_m=float(msg.width),
            fault=bool(msg.driver_error_status), stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9)

    def joint(msg, arm):
        try:
            mapping = dict(zip(msg.name, msg.position))
            q = [float(mapping[f'joint{i}']) for i in range(1,8)]
            if not np.isfinite(q).all():raise ValueError('nonfinite joints')
            entries[arm] = dict(joints_deg=np.rad2deg(q).tolist(), width_m=float(mapping['gripper']),
                stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9, received=time.monotonic())
        except (KeyError,ValueError):pass

    def camera(msg, role, transport):
        received = time.monotonic()
        images[role] = (msg, received, transport)
        if transport == 'raw':
            health[role].observe(msg.header.stamp.sec+msg.header.stamp.nanosec/1e9, received,
                msg.width, msg.height, msg.step, len(msg.data), msg.encoding)

    def refresh_sources():
        try:desired = camera_sources(camera_config)
        except (OSError,ValueError):return  # Retain known subscriptions during an invalid update.
        for role in VIEWS:
            source = desired.get(role)
            if role in sources and source == sources[role]:continue
            if role in camera_subs:node.destroy_subscription(camera_subs.pop(role))
            sources[role] = source
            images.pop(role,None);processed.pop(role,None)
            health[role] = FrameHealth(source['topic'] if source else '')
            previews[role] = dict(frames_decoded=0,decode_error=False)
            path=directory/({'main':'camera.jpg','left':'camera_left.jpg','right':'camera_right.jpg'}[role])
            path.unlink(missing_ok=True)
            if source:
                transport=source.get('transport','raw')
                camera_subs[role]=node.create_subscription(CompressedImage if transport=='compressed' else Image,
                    source['topic'], lambda msg,r=role,t=transport:camera(msg,r,t),qos)

    for arm in ('right','left'):
        subscriptions.append(node.create_subscription(JointState,f'/{arm}_arm/feedback/joint_states',lambda msg,a=arm:joint(msg,a),qos))
        subscriptions.append(node.create_subscription(GripperStatus,f'/{arm}_arm/feedback/gripper_status',lambda msg,a=arm:gripper(msg,a),qos))
    refresh_sources()
    try:
        while not (directory/'STOP_TELEMETRY').exists():
            if time.time()-(directory/'heartbeat').stat().st_mtime>4:break
            rclpy.spin_once(node,timeout_sec=.02)
            now=time.monotonic()
            if now-source_check>1:
                source_check=now;refresh_sources()
            if now-last<.2:continue
            last=now;ros_now=node.get_clock().now().nanoseconds/1e9
            arms={a:dict(row,age_s=ros_now-row['stamp'],received_age_s=now-row['received']) for a,row in entries.items()}
            for role,(msg,received,transport) in images.items():
                if processed.get(role)==received:continue
                processed[role]=received;row=previews[role]
                stamp=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9
                try:
                    frame=(cv2.imdecode(np.frombuffer(msg.data,np.uint8),cv2.IMREAD_COLOR)
                           if transport=='compressed' else decode_raw(msg))
                    if frame is None:raise ValueError('Invalid image')
                    if transport=='compressed':
                        h,w=frame.shape[:2]
                        health[role].observe(stamp,received,w,h,w*3,h*w*3,'bgr8')
                    write_preview(directory,role,frame)
                    row.update(frames_decoded=row['frames_decoded']+1,decode_error=False,
                        source_unix_s=stamp,received_monotonic_s=received,
                        width=int(frame.shape[1]),height=int(frame.shape[0]))
                except (ValueError,cv2.error):row['decode_error']=True
            snapshots={role:dict(row,source_age_s=ros_now-row.get('source_unix_s',0),
                received_age_s=now-row.get('received_monotonic_s',0),
                topic=(sources.get(role) or {}).get('topic','')) for role,row in previews.items()}
            control_topics={f'/{arm}_arm/control/{mode}' for arm in ('right','left')
                            for mode in ('joint_states','move_c','move_j','move_js','move_l','move_mit','move_p')}
            control_topics.update(n for n,_ in node.get_topic_names_and_types() if n.startswith(('/right_arm/control/','/left_arm/control/')))
            publishers={n:node.count_publishers(n) for n in sorted(control_topics)}
            wrist_health={side:h.snapshot(ros_now,now,node.count_publishers(h.topic) if h.topic else 0)
                          for side,h in health.items() if side!='main'}
            for side in wrist_health:
                if not sources.get(side):wrist_health[side].update(label='未分配',health='unknown')
            write(directory/'telemetry.json',dict(updated_unix_s=time.time(),arms=arms,
                camera_age_s=snapshots['main']['source_age_s'] if snapshots['main']['frames_decoded'] else None,
                main_camera=snapshots['main'],previews=snapshots,
                grippers={a:dict(row,age_s=ros_now-row['stamp']) for a,row in grippers.items()},
                wrist_cameras=wrist_health,control_publishers=publishers,
                mode='readonly_subscriptions',motor_enabled='not_available_from_joint_state'))
    finally:node.destroy_node();rclpy.shutdown()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--camera-config',type=Path);a=p.parse_args();run(a.directory,a.camera_config)
