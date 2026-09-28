#!/usr/bin/python3 -I
"""Fixed-device CAN setup. Also installed as a root-owned, no-argument helper.

Uses ip link only: no CAN payload, ROS service, motor enable, or motion command.
"""
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

BITRATE = 1000000
ADAPTERS = {'can0': '0022001D4148570D20343133',
            'can1': '003800284148571320343133'}
HELPER = Path('/usr/local/libexec/nero-can-activate')
IP = '/usr/sbin/ip'


def usb_serial(interface):
    device = (Path('/sys/class/net') / interface / 'device').resolve()
    for parent in (device, *device.parents):
        path = parent / 'serial'
        if path.is_file():
            return path.read_text().strip()
    return None


def inventory():
    proc = subprocess.run([IP, '-details', '-json', 'link', 'show'],
                          capture_output=True, text=True, timeout=5, check=True)
    rows = json.loads(proc.stdout)
    for row in rows:
        if row.get('link_type') == 'can':
            row['usb_serial'] = usb_serial(row['ifname'])
    return rows


def plan(rows):
    """Validate the entire mapping before changing anything, including swapped names."""
    names = {r['ifname']: r for r in rows}
    chosen = {}
    for target, serial in ADAPTERS.items():
        matches = [r for r in rows if r.get('link_type') == 'can'
                   and r.get('usb_serial') == serial]
        if len(matches) != 1:
            raise RuntimeError(f'{target}: 未找到唯一的 CAN 转接器 {serial}，请检查 USB 连接/供电')
        chosen[target] = matches[0]
    for target, row in chosen.items():
        occupant = names.get(target)
        if occupant and occupant.get('usb_serial') not in ADAPTERS.values():
            raise RuntimeError(f'{target} 被其他设备占用，不改动未知接口')
        if 'UP' in row.get('flags', []):
            data = row.get('linkinfo', {}).get('info_data', {})
            if row['ifname'] != target or data.get('bittiming', {}).get('bitrate') != BITRATE:
                raise RuntimeError(f'{row["ifname"]} 正在运行但名称/速率不匹配，不中断现有总线')
            if data.get('state') != 'ERROR-ACTIVE':
                raise RuntimeError(f'{target}: CAN 总线状态 {data.get("state")}，检查接线/终端电阻')
    actions = []

    def add(name, serial, *args):
        actions.append(dict(interface=name, serial=serial,
                            command=[IP, 'link', 'set', 'dev', name, *args]))

    renamed = {}
    for i, (target, row) in enumerate(chosen.items()):
        name, serial = row['ifname'], ADAPTERS[target]
        if name != target:
            temporary = f'nero_can_tmp{i}'
            if temporary in names:
                raise RuntimeError(f'临时接口名 {temporary} 已存在，停止自动重命名')
            add(name, serial, 'name', temporary)
            renamed[target] = temporary
    for target, temporary in renamed.items():
        add(temporary, ADAPTERS[target], 'name', target)
    for target, row in chosen.items():
        if 'UP' not in row.get('flags', []):
            add(target, ADAPTERS[target], 'type', 'can', 'bitrate', str(BITRATE))
            add(target, ADAPTERS[target], 'up')
    return actions


def activate():
    before = inventory()
    actions = plan(before)
    completed = []
    for action in actions:
        # Detect unplug/replug or renumbering during setup, before each mutation.
        row = next((r for r in inventory() if r['ifname'] == action['interface']), {})
        if row.get('link_type') != 'can' or row.get('usb_serial') != action['serial']:
            raise RuntimeError('CAN 设备在激活过程中变化，请重新检查连接')
        subprocess.run(action['command'], capture_output=True, text=True, timeout=5, check=True)
        completed.append(action['command'])
    after = inventory()
    if plan(after):
        raise RuntimeError('CAN 激活后复查仍未就绪')
    return dict(ok=True, before=before, after=after, commands=completed,
                message='CAN：右臂 can0 / 左臂 can1 已激活，均为 1 Mbps')


def main():
    try:
        if len(sys.argv) != 1:
            raise RuntimeError('This helper accepts no arguments')
        if os.geteuid() != 0:
            raise RuntimeError('需要管理员权限激活 CAN')
        # Root-owned lock serializes multiple desktop/terminal launches.
        with open('/run/lock/nero-can-activate.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            result = activate()
    except Exception as exc:
        detail = getattr(exc, 'stderr', '') or str(exc)
        result = dict(ok=False, error=detail)
    print(json.dumps(result, ensure_ascii=False))
    return 0 if result['ok'] else 1


if __name__ == '__main__':
    sys.exit(main())
