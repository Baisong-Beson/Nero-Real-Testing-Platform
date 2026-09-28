#!/usr/bin/python3 -I
"""One-time admin install: fixed CAN helper only, never blanket passwordless sudo."""
import os
from pathlib import Path
import pwd
import re
import subprocess
import tempfile


def install():
    if os.geteuid() != 0:
        raise RuntimeError('请使用系统管理员授权运行安装程序')
    uid = int(os.environ.get('PKEXEC_UID') or os.environ.get('SUDO_UID') or '-1')
    if uid <= 0:
        raise RuntimeError('请从普通桌面账户通过 pkexec 或 sudo 安装')
    user = pwd.getpwuid(uid).pw_name
    if not re.fullmatch(r'[a-z_][a-z0-9_-]*', user):
        raise RuntimeError('Unsupported account name')
    source = Path(__file__).resolve().with_name('can_startup.py')
    code = source.read_bytes()
    compile(code, str(source), 'exec')
    destination = Path('/usr/local/libexec/nero-can-activate')
    destination.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    for directory in [destination.parent, Path('/etc/sudoers.d')]:
        if directory.is_symlink() or directory.stat().st_uid != 0 or directory.stat().st_mode & 0o022:
            raise RuntimeError(f'管理员目录权限异常: {directory}')
    rule = f'{user} ALL=(root) NOPASSWD: {destination} ""\n'
    # Validate sudoers before replacing either installed file.
    with tempfile.NamedTemporaryFile(dir='/etc/sudoers.d', prefix='.nero-can-', delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(rule.encode())
    try:
        temporary.chmod(0o440)
        subprocess.run(['/usr/sbin/visudo', '-cf', str(temporary)], check=True)
        with tempfile.NamedTemporaryFile(dir=destination.parent, prefix='.nero-can-', delete=False) as stream:
            staged = Path(stream.name)
            stream.write(code)
        try:
            staged.chmod(0o755)
            staged.replace(destination)
        finally:
            staged.unlink(missing_ok=True)
        temporary.replace('/etc/sudoers.d/nero-can-activate')
    finally:
        temporary.unlink(missing_ok=True)
    print('CAN 自动激活已配置。仅授权固定双 CAN 接口助手，无参数、无机器人运动功能。')


if __name__ == '__main__':
    install()
