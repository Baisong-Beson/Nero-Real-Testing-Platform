"""ZED Mini recovery: exact camera driver only; verify decoded, advancing frames."""
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from .driver_reconnect import assert_idle
from .hardware_startup import isolated

SERIAL = '10027024'
TOPIC = '/zed_m/zed_node/rgb/color/rect/image/compressed'


def fresh(state):
    camera = state.get('main_camera', {})
    ages = [time.time()-state.get('updated_unix_s', 0),
            camera.get('source_age_s', 999), camera.get('received_age_s', 999)]
    return (camera.get('frames_decoded', 0) > 0 and not camera.get('decode_error')
            and all(isinstance(v, (float, int)) and math.isfinite(v) and -.05 <= v < .8 for v in ages))


def usb_snapshot(usb_root=Path('/sys/bus/usb/devices'), video_root=Path('/sys/class/video4linux')):
    candidates = []
    for path in usb_root.glob('*'):
        try:
            if (path/'idVendor').read_text().strip() != '2b03':continue
            serial=(path/'serial').read_text().strip() if (path/'serial').exists() else ''
            candidates.append(dict(path=path.name,product=(path/'product').read_text().strip(),serial=serial,
                product_id=(path/'idProduct').read_text().strip(),devnum=(path/'devnum').read_text().strip(),speed=(path/'speed').read_text().strip()))
        except OSError:
            continue
    hids=[d for d in candidates if d['serial']==SERIAL and d['product_id']=='f681']
    uvcs=[d for d in candidates if d['product_id'] in ('f680','f682')]
    # ZED Mini's UVC device has no USB serial descriptor. Match its HID's
    # USB2/USB3 port-chain pair; accept a unique Mini UVC as fallback. The SDK
    # launch is still explicitly pinned to SERIAL, never to a /dev/video index.
    pairs={d['path'].partition('-')[2] for d in hids}
    selected=[d for d in uvcs if d['serial']==SERIAL or (hids and
              (d['path'].partition('-')[2] in pairs or len(uvcs)==1))]
    devices=hids+selected
    selected_paths={d['path'] for d in selected}
    videos = []
    for node in video_root.glob('video*'):
        for parent in (node.resolve(), *node.resolve().parents):
            try:
                if parent.name in selected_paths and (parent/'idVendor').read_text().strip() == '2b03':
                    videos.append('/dev/'+node.name);break
            except OSError:
                continue
    return dict(serial=SERIAL, devices=devices, video_devices=sorted(videos))


def usb_key(usb):
    return json.dumps(usb, sort_keys=True)


def camera_processes():
    records = []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            args = [a.decode() for a in (path/'cmdline').read_bytes().split(b'\0') if a]
            launcher = (len(args) >= 5 and args[:5] == ['/usr/bin/python3', '/opt/ros/humble/bin/ros2', 'launch', 'zed_wrapper', 'zed_camera.launch.py']
                        and 'camera_model:=zedm' in args and 'camera_name:=zed_m' in args)
            container = (bool(args) and Path(args[0]).name in ('component_container', 'component_container_isolated')
                         and '__ns:=/zed_m' in args and '__node:=zed_container' in args)
            if launcher or container:
                # Only the dedicated ZED group is considered; no robot launches.
                ticks = (path/'stat').read_text().split(') ')[1].split()[19]
                records.append(dict(pid=int(path.name), start_ticks=ticks, args=args, kind='launcher' if launcher else 'container'))
        except (OSError, UnicodeError, IndexError):
            continue
    return records


def same_process(record):
    return record in camera_processes()


def stop_camera(records, telemetry, cancel, notify):
    if sum(r['kind']=='launcher' for r in records) > 1 or sum(r['kind']=='container' for r in records) > 1:
        raise RuntimeError('检测到多个主相机驱动，未自动终止；请检查重复启动')
    for sig, seconds in [(signal.SIGINT, 4), (signal.SIGTERM, 2), (signal.SIGKILL, 2)]:
        alive = [record for record in records if same_process(record)]
        if not alive:return
        assert_idle(telemetry())
        for record in alive:
            if same_process(record):
                notify(f'清理主相机旧连接 PID {record["pid"]}')
                try:os.kill(record['pid'], sig)
                except ProcessLookupError:pass
        deadline = time.monotonic()+seconds
        while time.monotonic() < deadline:
            if not any(same_process(record) for record in records):return
            if cancel.is_set():raise RuntimeError('主相机恢复已取消')
            cancel.wait(.1)
    if any(same_process(record) for record in records):
        raise RuntimeError('旧主相机驱动未退出，停止重复启动')


def validate_recipe(recipe):
    args = recipe['command']
    if recipe.get('serial') != SERIAL or args[:5] != ['/usr/bin/python3', '/opt/ros/humble/bin/ros2', 'launch', 'zed_wrapper', 'zed_camera.launch.py']:
        raise RuntimeError('主相机恢复配置不匹配')
    for value in ['camera_model:=zedm', 'camera_name:=zed_m', 'node_name:=zed_node', 'serial_number:='+SERIAL]:
        if value not in args:raise RuntimeError('主相机型号、命名空间或序列号配置不匹配')


def wait_frames(telemetry, cancel, seconds):
    deadline = time.monotonic()+seconds
    previous = None;count = 0
    while True:
        if cancel.is_set():raise RuntimeError('主相机恢复已取消')
        state = telemetry();camera = state.get('main_camera', {})
        stamp = camera.get('source_unix_s')
        if fresh(state) and stamp is not None:
            if previous is None or stamp > previous:count += 1;previous = stamp
            elif stamp < previous:count = 0;previous = stamp
            if count >= 3:return camera
        else:count = 0;previous = None
        if time.monotonic() >= deadline:return None
        cancel.wait(.1)


def recover(platform, telemetry, cancel, notify):
    if isolated():return dict(ok=True, skipped=True, message='隔离测试：跳过主相机恢复')
    result = dict(ok=False, restarted=False, serial=SERIAL, topic=TOPIC)
    try:
        if fresh(telemetry()):
            result.update(ok=True, message='主相机画面实时，保留现有连接');return result
        # Allow DDS discovery / an existing open() to deliver frames first.
        if wait_frames(telemetry, cancel, 2):
            result.update(ok=True, message='主相机画面已连接');return result
        result['usb'] = usb_snapshot()
        if not result['usb']['video_devices']:
            result.update(waiting_usb=True, message='主相机视频 USB 未识别：请重插相机自身数据线/检查供电；识别后自动恢复')
            return result
        recipe = json.loads((platform/'config/main_camera_reconnect.json').read_text())
        validate_recipe(recipe)
        with open('/tmp/plate_approved_motion.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert_idle(telemetry())
            existing = camera_processes()
            result['previous_processes'] = existing
            stop_camera(existing, telemetry, cancel, notify)
            if cancel.is_set():raise RuntimeError('主相机恢复已取消')
            if camera_processes():raise RuntimeError('另一程序已启动主相机，停止重复启动')
            assert_idle(telemetry())
            env = dict(os.environ);env.update(recipe['environment'])
            env.pop('PYTHONNOUSERSITE', None);env.pop('PYTHONHOME', None)
            env['ROS_DOMAIN_ID']='0';env['ROS_LOG_DIR']=str(platform/'logs/ros')
            log = platform/'logs'/f'zed_reconnect_{time.time_ns()}.log'
            notify('正在启动 ZED Mini 主相机，等待连续新图像…')
            with log.open('w') as stream:
                process = subprocess.Popen(recipe['command'], env=env, cwd=platform, stdout=stream,
                                           stderr=subprocess.STDOUT, start_new_session=True)
            result.update(restarted=True, pid=process.pid, log=str(log))
            frames = wait_frames(telemetry, cancel, 30)
            if not frames:
                raise RuntimeError('启动后未收到连续可解码图像，请查看相机日志：'+str(log))
            result.update(ok=True, frames=frames, message='主相机已自动恢复，连续新图像验证通过')
    except Exception as exc:
        result.update(error=str(exc), message='主相机未恢复：'+str(exc))
    return result


class ReconnectWatch:
    """One automatic attempt per USB enumeration or healthy-to-stale episode."""
    def __init__(self):
        self.attempted_key=None;self.stale_since=None;self.last_check=0.

    def completed(self, usb):
        self.attempted_key=usb_key(usb)

    def due(self, state, usb, now):
        if fresh(state):
            self.attempted_key=None;self.stale_since=None;return False
        if self.stale_since is None:self.stale_since=now
        if now-self.stale_since < 3 or not usb['video_devices']:return False
        key=usb_key(usb)
        if key == self.attempted_key:return False
        self.attempted_key=key;return True
