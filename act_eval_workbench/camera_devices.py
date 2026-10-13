"""Local camera assignment and driver lifecycle; never issues robot commands."""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import threading
import time

ROLES = ('main', 'left', 'right')


def config_path(platform):
    return Path(platform) / 'config' / 'camera_bindings.json'


def load_config(platform):
    try:
        return json.loads(config_path(platform).read_text())
    except FileNotFoundError:
        return dict(version=1, auto_rebind=True, roles={}, driver={})


def save_config(platform, config):
    path = config_path(platform)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    temp.replace(path)


def discover_devices():
    devices = []
    result = subprocess.run(['rs-enumerate-devices', '-s'], capture_output=True, text=True, timeout=8)
    if result.returncode:
        raise RuntimeError('相机扫描失败：' + result.stderr[-300:])
    for line in result.stdout.splitlines():
        match = re.match(r'\s*(Intel.*?RealSense.*?)\s+(\d{8,})\s+([\d.]+)\s*$', line)
        if match:
            name, serial, _ = match.groups()
            devices.append(dict(id='realsense:'+serial, kind='realsense', serial=serial, name=name.strip()))
    for path in Path('/sys/bus/usb/devices').glob('*'):
        try:
            if (path/'idVendor').read_text().strip() != '2b03':continue
            serial = (path/'serial').read_text().strip()
            if not serial.isdigit():continue
            devices.append(dict(id='zed:'+serial, kind='zed', serial=serial, name='ZED Mini'))
        except OSError:
            continue
    return list({d['id']: d for d in devices}.values())


def binding(device, role):
    row = dict(device)
    if device['kind'] == 'realsense':
        row.update(topic=f'/{role}_wrist/color/image_raw' if role != 'main' else '/main_view/color/image_raw', transport='raw')
    elif device['kind'] == 'zed':
        # This platform's ZED driver is a shared, independently managed source.
        row.update(topic='/zed_m/zed_node/rgb/color/rect/image/compressed', transport='compressed')
    else:
        raise ValueError('不支持的相机类型')
    return row


def validate_roles(roles):
    ids = [d['id'] for d in roles.values() if d]
    if len(ids) != len(set(ids)):
        raise ValueError('同一台相机只能分配给一个视角')
    if any(role not in ROLES for role in roles):raise ValueError('未知视角')
    if sum(bool(d and d['kind']=='zed') for d in roles.values())>1:
        raise ValueError('当前平台支持一台 ZED 主设备')
    for row in roles.values():
        if row and (not row['serial'].isdigit() or row['kind'] not in ('realsense', 'zed')):
            raise ValueError('无效的相机标识')


def automatic_replacements(roles, devices, missing_roles):
    """Only replace one previously seen missing camera with one unassigned peer."""
    present = {d['id']:d for d in devices}
    assigned = {d['id'] for d in roles.values() if d}
    replacements = {}
    for kind in ('realsense', 'zed'):
        missing = [r for r in missing_roles if roles.get(r) and roles[r]['kind']==kind and roles[r]['id'] not in present]
        candidates = [d for d in devices if d['kind']==kind and d['id'] not in assigned]
        if len(missing)==1 and len(candidates)==1:
            replacements[missing[0]] = binding(candidates[0], missing[0])
    return replacements


def camera_processes(role):
    node = f'__node:={role}_wrist' if role != 'main' else '__node:=main_view'
    records = []
    for path in Path('/proc').glob('[0-9]*'):
        try:
            args = [a.decode() for a in (path/'cmdline').read_bytes().split(b'\0') if a]
            if args and Path(args[0]).name == 'realsense2_camera_node' and node in args:
                records.append((int(path.name), (path/'stat').read_text().split(') ')[1].split()[19]))
        except (OSError, UnicodeError, IndexError):pass
    return records


def systemctl(*args, check=True):
    env = dict(os.environ, XDG_RUNTIME_DIR=f'/run/user/{os.getuid()}',
               DBUS_SESSION_BUS_ADDRESS=f'unix:path=/run/user/{os.getuid()}/bus')
    return subprocess.run(['systemctl','--user',*args], env=env, capture_output=True, text=True, timeout=18, check=check)


def configure_drivers(platform, previous, current):
    old_zed=next((d for d in previous.get('roles',{}).values() if d and d['kind']=='zed'),None)
    new_zed=next((d for d in current['roles'].values() if d and d['kind']=='zed'),None)
    if new_zed and (not old_zed or old_zed['serial']!=new_zed['serial']):
        from . import camera_reconnect as zed
        recipe_path=Path(platform)/'config/main_camera_reconnect.json'
        recipe=json.loads(recipe_path.read_text())
        records=zed.camera_processes()
        if sum(r['kind']=='launcher' for r in records)>1 or sum(r['kind']=='container' for r in records)>1:
            raise RuntimeError('存在重复 ZED 驱动')
        for record in records:
            if zed.same_process(record):os.kill(record['pid'],signal.SIGINT)
        deadline=time.monotonic()+6
        while zed.camera_processes() and time.monotonic()<deadline:time.sleep(.1)
        if zed.camera_processes():raise RuntimeError('旧 ZED 驱动尚未退出')
        recipe['serial']=new_zed['serial']
        recipe['command']=[('serial_number:='+new_zed['serial']) if a.startswith('serial_number:=') else a for a in recipe['command']]
        temp=recipe_path.with_suffix('.tmp');temp.write_text(json.dumps(recipe,indent=2));temp.replace(recipe_path)
        zed.validate_recipe(recipe)
        env=dict(os.environ,**recipe.get('environment',{}));env['ROS_DOMAIN_ID']='0'
        log=Path(platform)/'logs'/f'camera_zed_{time.time_ns()}.log'
        with log.open('w') as stream:
            subprocess.Popen(recipe['command'],env=env,cwd=platform,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
    for role in ROLES:
        old, new = previous.get('roles',{}).get(role), current['roles'].get(role)
        if old == new:continue
        # Only this platform's explicitly named camera drivers are stopped.
        unit_name = f'nero-camera-{role}.service'
        systemctl('stop', unit_name, check=False)
        if role=='right':systemctl('stop','nero-right-wrist-camera.service',check=False)
        if role=='right':systemctl('disable','nero-right-wrist-camera.service',check=False)
        existing = camera_processes(role)
        if len(existing)>1:raise RuntimeError('存在重复相机驱动，请先关闭重复实例')
        for pid, start in existing:
            if (pid,start) in camera_processes(role):os.kill(pid,signal.SIGINT)
        deadline = time.monotonic()+5
        while camera_processes(role) and time.monotonic()<deadline:time.sleep(.1)
        if camera_processes(role):raise RuntimeError('旧相机驱动尚未退出')
        if not new or new['kind']!='realsense':
            systemctl('disable',unit_name,check=False)
            continue
        driver = current.get('driver',{})
        if not Path(driver.get('executable','')).is_file() or not Path(driver.get('setup','')).is_file():
            raise RuntimeError('未找到 RealSense 驱动环境')
        unit = Path.home()/'.config/systemd/user'/unit_name
        unit.parent.mkdir(parents=True, exist_ok=True)
        script = Path(platform)/'act_eval_workbench/camera_devices.py'
        unit.write_text('[Unit]\nDescription=Platform camera source\n\n[Service]\nType=simple\n'
            +f'ExecStart=/usr/bin/python3 "{script}" --run-driver "{config_path(platform)}" {role}\n'
            +'Restart=on-failure\nRestartSec=3\nKillSignal=SIGINT\nTimeoutStopSec=10\nEnvironment=ROS_DOMAIN_ID=0\n'
            +'\n[Install]\nWantedBy=default.target\n',encoding='utf-8')
        systemctl('daemon-reload')
        systemctl('enable','--now',unit_name)


class CameraManager:
    def __init__(self, platform, idle, notify):
        self.platform=Path(platform);self.idle=idle;self.notify=notify
        self.devices=[];self.error='';self.pending=set();self.seen=set();self.lock=threading.Lock();self.updating=False
        self.stopped=threading.Event();self.refresh=threading.Event()
        self.config=load_config(platform)
        self.seen={d['id'] for d in self.config.get('roles',{}).values() if d}
        self.thread=threading.Thread(target=self.watch,daemon=True);self.thread.start()

    def apply(self, roles, auto_rebind):
        validate_roles(roles)
        with self.lock:
            self.updating=True
            try:
                if not self.idle():raise RuntimeError('请在任务结束后更改相机分配')
                old=load_config(self.platform)
                new=dict(old, roles=roles, auto_rebind=auto_rebind)
                save_config(self.platform,new)
                try:
                    configure_drivers(self.platform,old,new)
                except Exception:
                    save_config(self.platform,old)
                    try:configure_drivers(self.platform,new,old)
                    except Exception as rollback:self.notify('相机驱动恢复失败：'+str(rollback))
                    raise
                self.config=new;self.pending.clear();self.seen={d['id'] for d in self.devices}
                self.notify('相机分配已保存')
            finally:self.updating=False

    def watch(self):
        while not self.stopped.is_set():
            try:
                devices=discover_devices();self.devices=devices;self.error=''
                present={d['id'] for d in devices}
                roles=self.config.get('roles',{})
                for role,row in roles.items():
                    if row and row['id'] in self.seen and row['id'] not in present:self.pending.add(role)
                    if row and row['id'] in present:self.pending.discard(role)
                self.seen=present
                changes=automatic_replacements(roles,devices,self.pending)
                if changes and self.config.get('auto_rebind',True) and self.idle():
                    self.apply(dict(roles,**changes),True)
                    self.notify('已自动更新替换相机的设备绑定')
            except Exception as exc:self.error=str(exc)
            self.refresh.wait(4);self.refresh.clear()

    def close(self):
        self.stopped.set();self.refresh.set()


def run_driver(config, role):
    data=json.loads(Path(config).read_text());row=data['roles'][role];validate_roles({role:row})
    driver=data['driver'];node=f'{role}_wrist' if role!='main' else 'main_view'
    args=[driver['executable'],'--ros-args','-r','__node:='+node,'-r','__ns:=/',
          '-p','serial_no:=_'+row['serial'],'-p','enable_color:=true','-p','enable_depth:=false',
          '-p','enable_infra1:=false','-p','enable_infra2:=false','-p','pointcloud.enable:=false',
          '-p','align_depth.enable:=false','-p','depth_module.color_profile:=640,480,15',
          '-p','rgb_camera.color_profile:=640,480,15','-r','~/color/image_rect_raw:=~/color/image_raw']
    os.execv('/bin/bash',['bash','-c','source /opt/ros/humble/setup.bash; source "$1"; shift; exec "$@"',
                         'camera-driver',driver['setup'],*args])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--run-driver',nargs=2)
    args=parser.parse_args()
    if args.run_driver:run_driver(*args.run_driver)
