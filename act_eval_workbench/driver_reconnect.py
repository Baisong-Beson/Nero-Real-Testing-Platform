"""Recover only missing drivers or positively identified dead CAN receive threads."""
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def fresh_arms(state):
    if not 0 <= time.time()-state.get('updated_unix_s', 0) < 2:
        return set()
    return {side for side, row in state.get('arms', {}).items()
            if -.05 <= row.get('age_s', 999) <= .3}


def assert_idle(state):
    publishers = state.get('control_publishers', {})
    if not 0 <= time.time()-state.get('updated_unix_s', 0) < 2 or len(publishers) < 14:
        raise RuntimeError('缺少完整实时控制发布者检查，未重启驱动')
    if any(publishers.values()):
        raise RuntimeError('存在机器人控制发布者，未重启驱动')


def matching_drivers(command, side):
    matches = []
    for directory in Path('/proc').glob('[0-9]*'):
        try:
            args = [s.decode() for s in (directory/'cmdline').read_bytes().split(b'\0') if s]
            if len(args) > 1 and args[:2] == command[:2] and f'__ns:=/{side}_arm' in args:
                matches.append((int(directory.name), args))
        except (OSError, UnicodeError):
            pass
    return matches


def dead_receive_thread(pid):
    path = (Path('/proc')/str(pid)/'fd/2').resolve()
    if not path.is_file():
        return False
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-16384))
        tail = stream.read().decode(errors='replace')
    return '_read_loop' in tail and 'CanOperationError' in tail and 'No such device' in tail


def validate_command(command, side):
    if len(command) < 2 or command[0] != '/usr/bin/python3' or Path(command[1]).name != 'agx_arm_ctrl_single':
        raise RuntimeError('机械臂恢复配置不是已知驱动')
    if f'__ns:=/{side}_arm' not in command:
        raise RuntimeError('机械臂恢复配置的左右命名空间错误')
    if command[-4:] != ['-p', 'auto_enable:=false', '-p', 'control_enabled:=false']:
        raise RuntimeError('恢复驱动必须禁止自动使能并关闭控制入口')


def recover(platform, telemetry, cancel, notify):
    if os.environ.get('ACT_EVAL_FAKE_DRIVER') == '1' or os.environ.get('ROS_DOMAIN_ID', '0') != '0':
        return dict(ok=True, skipped=True)
    restarted = []
    try:
        # Give existing healthy drivers/discovery a chance; return immediately
        # when both streams are already fresh, without a fixed startup delay.
        deadline = time.monotonic()+3
        while True:
            state = telemetry()
            if fresh_arms(state) >= {'right', 'left'}:
                return dict(ok=True, restarted=[], message='双臂实时反馈已连接，保留现有驱动')
            if cancel.is_set():raise RuntimeError('连接恢复已取消')
            if time.monotonic() >= deadline:break
            cancel.wait(.1)
        configuration = json.loads((platform/'config/arm_driver_reconnect.json').read_text())
        with open('/tmp/plate_approved_motion.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            assert_idle(telemetry())
            pending = []
            for side in ('right', 'left'):
                if side in fresh_arms(telemetry()):continue
                recipe = configuration[side];command = recipe['command']
                validate_command(command, side)
                existing = matching_drivers(command, side)
                if len(existing) > 1:raise RuntimeError(f'{side}: 存在重复驱动，未自动终止')
                if existing and not dead_receive_thread(existing[0][0]):
                    raise RuntimeError(f'{side}: 缺少反馈，但未确认接收线程退出，请检查驱动日志')
                pending.append((side, recipe, existing))
            for side, recipe, existing in pending:
                if cancel.is_set():raise RuntimeError('连接恢复已取消')
                assert_idle(telemetry())
                if existing:
                    pid, args = existing[0]
                    now = matching_drivers(recipe['command'], side)
                    if now != existing:raise RuntimeError('驱动进程已变化，请重新检查')
                    notify(f'{side} 驱动接收线程因拔线退出，正在重建连接（自动使能关闭）')
                    os.kill(pid, signal.SIGTERM)
                    deadline = time.monotonic()+3
                    while matching_drivers(recipe['command'], side):
                        if time.monotonic() >= deadline:
                            # A crashed SDK reader can leave non-daemon threads
                            # blocking normal process exit. Kill only the same
                            # already-proven failed PID, with controls still idle.
                            assert_idle(telemetry())
                            if matching_drivers(recipe['command'], side) != existing:
                                raise RuntimeError('驱动进程已变化，停止终止旧进程')
                            notify(f'{side} 故障驱动正常退出超时，清理其残留进程 {pid}')
                            os.kill(pid, signal.SIGKILL)
                            deadline = time.monotonic()+2
                            while matching_drivers(recipe['command'], side):
                                if time.monotonic() >= deadline:raise RuntimeError('故障驱动残留进程未退出')
                                cancel.wait(.05)
                            break
                        cancel.wait(.05)
                if cancel.is_set():raise RuntimeError('连接恢复已取消')
                if matching_drivers(recipe['command'], side):raise RuntimeError('驱动已被另一进程重启，停止重复启动')
                env = dict(os.environ)
                env.pop('PYTHONNOUSERSITE', None);env.pop('PYTHONHOME', None)
                env.update(recipe['environment']);env['ROS_DOMAIN_ID']='0'
                env['ROS_LOG_DIR']=str(platform/'logs/ros')
                log = platform/'logs'/f'driver_{side}_reconnect_{time.time_ns()}.log'
                with log.open('w') as stream:
                    process = subprocess.Popen(recipe['command'], cwd=platform, env=env,
                                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                restarted.append(dict(side=side, pid=process.pid, log=str(log)))
            deadline = time.monotonic()+15
            while not fresh_arms(telemetry()) >= {'right', 'left'}:
                if cancel.is_set():raise RuntimeError('连接恢复已取消')
                if time.monotonic() >= deadline:raise RuntimeError('驱动已启动但尚无双臂实时反馈，请检查机器人电源及驱动日志')
                cancel.wait(.1)
        return dict(ok=True, restarted=restarted, message='双臂驱动重连成功，实时反馈已恢复')
    except Exception as exc:
        return dict(ok=False, restarted=restarted, message='机械臂连接未恢复：'+str(exc))
