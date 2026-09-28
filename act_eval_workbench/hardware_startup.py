"""Nonblocking desktop startup orchestration and USB diagnostics; no ROS controls."""
import json
import os
from pathlib import Path
import subprocess
import time
from . import can_startup


def isolated():
    return os.environ.get('ACT_EVAL_FAKE_DRIVER') == '1' or os.environ.get('ROS_DOMAIN_ID', '0') != '0'


def camera_usb_status():
    cameras = []
    base = Path('/sys/bus/usb/devices')
    for directory in base.glob('*'):
        try:
            product = (directory / 'product').read_text().strip()
            if 'RealSense' not in product and 'ZED' not in product:
                continue
            cameras.append(dict(product=product, serial=(directory/'serial').read_text().strip(),
                                speed_mbps=(directory/'speed').read_text().strip(), path=directory.name))
        except OSError:
            continue
    videos = sorted(str(p) for p in Path('/dev').glob('video*'))
    message = ('USB 视频设备未枚举：请检查 USB 3.0 接口、数据线和扩展坞供电'
               if not videos else f'USB 视频节点 {len(videos)} 个；是否出图以实时监视为准')
    return dict(cameras=cameras, video_devices=videos, message=message)


def run_command(command, cancel, timeout=20):
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    deadline = time.monotonic() + timeout
    try:
        while True:
            if cancel.is_set():
                raise RuntimeError('连接检查已取消')
            try:
                stdout, stderr = process.communicate(timeout=.2)
                return process.returncode, stdout, stderr
            except subprocess.TimeoutExpired:
                if time.monotonic() > deadline:
                    raise RuntimeError('系统授权/接口激活超时；可点击重新检查连接重试')
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate(timeout=2)


def install_helper(notify, cancel):
    installer = Path(__file__).with_name('install_can_startup.py')
    code, out, err = run_command(['/usr/bin/pkexec', '/usr/bin/python3', '-I', str(installer)], cancel, 180)
    # An SSH-launched GUI has no polkit agent in its login session. Ask through
    # the existing local terminal server instead; never read/store the password.
    if code and any(s in err for s in ('authentication agent', 'controlling terminal', '/dev/tty')):
        notify('系统图形授权不可用，已打开本机终端；请在该窗口输入管理员密码（输入时不显示字符）')
        code, out, err = run_command(['/usr/bin/gnome-terminal', '--wait', '--title=CAN 自动激活首次配置',
                                      '--', '/usr/bin/sudo', '/usr/bin/python3', '-I', str(installer)], cancel, 180)
    if code:
        raise RuntimeError('CAN 自动配置未完成，请完成本机管理员授权后重试。' + (err or out)[-700:])
    notify(out.strip() or 'CAN 自动激活助手已安装')


def recover(notify, cancel, allow_install=True):
    if isolated():
        return dict(ok=True, skipped=True, message='隔离测试环境：跳过真实 CAN 激活')
    result = dict(ok=False, camera_usb=camera_usb_status(), physical_motion_executed=False)
    try:
        rows = can_startup.inventory()
        result['can_before'] = rows
        actions = can_startup.plan(rows)
        if not actions:
            result.update(ok=True, message='CAN：右臂 can0 / 左臂 can1 已就绪，保留当前接口')
            return result
        notify('正在激活 CAN 接口；首次使用需要一次系统管理员授权')
        helper = can_startup.HELPER
        installed = helper.is_file() and helper.stat().st_uid == 0 and not helper.stat().st_mode & 0o022
        if not installed:
            if not allow_install:
                raise RuntimeError('尚未安装 CAN 自动激活助手，需要在本机桌面完成一次管理员授权')
            install_helper(notify, cancel)
        code, out, err = run_command(['/usr/bin/sudo', '-n', str(helper)], cancel)
        if code:
            raise RuntimeError((err or out or 'CAN 激活失败')[-1500:])
        result['activation'] = json.loads(out)
        result['can_after'] = can_startup.inventory()
        if can_startup.plan(result['can_after']):
            raise RuntimeError('CAN 复查未通过')
        result.update(ok=True, message=result['activation']['message'])
    except Exception as exc:
        result.update(error=str(exc), message='CAN 未就绪：' + str(exc))
    finally:
        notify(result['camera_usb']['message'])
    return result
