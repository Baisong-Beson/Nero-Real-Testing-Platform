"""Reversible record clearing and reproducible CSV/JSON exports; no robot IO."""
import csv
from datetime import datetime
from pathlib import Path
import time
import uuid
from . import common as m

def archive_state(root):
    path=Path(root)/'record_archive.json'
    if not path.exists():return dict(version=1,archived={},history=[])
    state=m.read(path)
    if state.get('version')!=1 or not isinstance(state.get('archived'),dict):raise ValueError('记录归档索引损坏，请检查 record_archive.json')
    return state

def is_experiment(result):
    """Only a real, started model-controlled attempt is an experiment record."""
    return (result.get('kind')=='formal' and bool(result.get('started'))
            and not result.get('simulation_only') and result.get('physical_motion_executed') is not False)

def scan(root=None,include_archived=False,include_diagnostics=False):
    root=Path(root or m.RUNS);hidden=archive_state(root)['archived'];rows=[]
    if not root.exists():return rows
    for folder in sorted(root.iterdir(),reverse=True):
        if not folder.is_dir() or folder.is_symlink():continue
        path=folder/'result.json';failed_prepare=not path.exists()
        if failed_prepare:path=folder/'job_error.json'
        if not path.exists() or (not include_archived and folder.name in hidden):continue
        try:
            data=m.read(path)
            if not isinstance(data,dict):continue
            if failed_prepare:
                config=m.read(folder/'config.json') if (folder/'config.json').exists() else {}
                data=dict(data,kind=folder.name.rsplit('_',1)[-1],settings=config.get('settings',{}),started=False,runtime_completed=False)
        except (ValueError,OSError):continue
        if not include_diagnostics and not is_experiment(data):continue
        rows.append(dict(id=folder.name,directory=str(folder),source_file=str(path),result=data,
                         archived=folder.name in hidden,source_sha256=m.sha(path)))
    return rows

def set_archived(root,ids,archived,operator):
    root=Path(root).resolve();ids=list(dict.fromkeys(ids))
    known={row['id']:row for row in scan(root,True)}
    if not ids:raise ValueError('没有可处理的记录')
    if any(name not in known for name in ids):raise ValueError('所选记录已改变，请刷新后重试')
    state=archive_state(root);event=dict(action='clear' if archived else 'restore',ids=ids,operator=operator,unix_s=time.time())
    for name in ids:
        if archived:state['archived'][name]=dict(operator=operator,unix_s=event['unix_s'],source_sha256=known[name]['source_sha256'])
        else:state['archived'].pop(name,None)
    state.setdefault('history',[]).append(event);m.write(root/'record_archive.json',state)
    return len(ids)

def summarize(rows):
    groups={}
    for row in rows:
        r=row['result']
        if not is_experiment(r):continue
        s=r.get('settings',{});protocol=r.get('protocol_sha256','unknown')
        key=(s.get('task','unknown'),s.get('model','unknown'),s.get('layout',''),protocol)
        g=groups.setdefault(key,dict(task=key[0],model=key[1],layout=key[2],protocol=protocol,
            target=s.get('repeats_per_group',20),duration_s=s.get('duration_s',0),started=0,success=0,failed=0,pending=0))
        g['started']+=1;v=r.get('adjudication',{}).get('success')
        g['success' if v is True else 'failed' if v is False else 'pending']+=1
    return list(groups.values())

def error_text(result):
    return result.get('error') or '; '.join(result.get('errors',[]))

def csv_value(value):
    # Preserve user text as text when opened by Excel, not as a formula.
    if isinstance(value,str) and value.lstrip().startswith(('=','+','-','@')):return "'"+value
    return value

def csv_write(path,rows,fields):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({key:csv_value(row.get(key,'')) for key in fields} for row in rows)

def export(root,destination,ids=None):
    rows=scan(root)
    if ids is not None:
        requested=set(ids);rows=[r for r in rows if r['id'] in requested]
        if {r['id'] for r in rows}!=requested:raise ValueError('所选记录已改变，请刷新后重试')
    if not rows:raise ValueError('没有可导出的记录')
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=True)
    out=destination/('ACT_results_'+time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6]);out.mkdir()
    details=[]
    for row in rows:
        r=row['result'];s=r.get('settings',{});v=r.get('adjudication',{});video=r.get('video',{})
        stamp=r.get('started_unix_s');finished=r.get('finished_unix_s')
        details.append(dict(记录编号=row['id'],开始时间=datetime.fromtimestamp(stamp).isoformat(timespec='seconds') if stamp else row['id'][:15],
            类型=r.get('kind',''),任务=s.get('task',''),模型=s.get('model',''),操作员=s.get('operator',''),布局=s.get('layout',''),
            推理上限秒=s.get('duration_s',''),已开始=bool(r.get('started')),运行完成=bool(r.get('runtime_completed')),
            实际运行秒=r.get('active_duration_s',round(finished-stamp,3) if stamp and finished else ''),
            结果='成功' if v.get('success') is True else '失败' if v.get('success') is False else '未判定',依据=v.get('reason',''),
            错误=error_text(r),模拟=bool(r.get('simulation_only') or r.get('physical_motion_executed') is False),
            控制点=r.get('commands_sent',''),录像质量=video.get('quality','error' if video.get('error') else '未标注'),
            录像间隔提示数=video.get('gap_warning_count',0),最大录像间隔秒=video.get('max_source_gap_s',''),
            参数协议=r.get('protocol_sha256',''),原始目录=row['directory'],原始结果SHA256=row['source_sha256']))
    groups=summarize(rows);stats=[]
    for g in groups:
        stats.append(dict(任务=g['task'],模型=g['model'],布局=g['layout'],推理上限秒=g['duration_s'],参数协议=g['protocol'],
            目标次数=g['target'],已开始=g['started'],成功=g['success'],失败=g['failed'],待判定=g['pending'],
            成功率_成功除已开始=f'{g["success"]/g["started"]:.4%}'))
    csv_write(out/'records.csv',details,list(details[0]))
    csv_write(out/'summary.csv',stats,['任务','模型','布局','推理上限秒','参数协议','目标次数','已开始','成功','失败','待判定','成功率_成功除已开始'])
    m.write(out/'records.json',dict(schema='act_results_export.v2',record_scope='real_started_formal',exported_unix_s=time.time(),source_root=str(Path(root).resolve()),
        scope='selected' if ids is not None else 'all_active',records=rows,groups=groups))
    (out/'README.md').write_text('# ACT 实验结果导出\n\nrecords.csv 为逐条记录，summary.csv 为本次导出范围内的真实正式实验统计，可用 Excel 打开；records.json 保存完整结果和来源校验。\n\n列表、导出和成功率仅包含已开始的真实 formal 模型推理执行，包含执行后中止/失败的试验。复位、失能、预检失败、离线模拟、只读推理和假驱动测试不进入实验记录；对应诊断文件仍保留在原目录。未判定不算成功。不同参数协议分组。已清除归档记录不在本次导出范围。\n\n原始录像和逐点日志没有复制进本导出包，仍在 records.csv 的“原始目录”；请连同原始目录备份后再移动到其他机器。录像间隔提示表示证据存在短暂缺帧，不代表任务成功。\n',encoding='utf-8')
    m.write(out/'manifest.json',dict(files={p.name:m.sha(p) for p in out.iterdir() if p.is_file()},records=len(rows)))
    return out
