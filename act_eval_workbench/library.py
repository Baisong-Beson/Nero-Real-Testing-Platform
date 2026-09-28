"""Managed, immutable model/task packages. Imports never run package code."""
from pathlib import Path
import json
import re
import shutil
import tempfile
import time
from urllib.parse import urlparse
import numpy as np

BUILTIN_TASKS=('plate','banana','holder')
BUILTIN_MODELS=('mpi-base','p0','g2')
CONTRACT='nero.dual7.absolute_radians.closedness.v1'
CAMERAS={'main':'observation/exterior_image_1_left',
         'left_wrist':'observation/left_wrist_image', 'right_wrist':'observation/right_wrist_image'}

def common():
    from . import common as c
    return c

def root():return common().PLATFORM/'library'

def identifier(value):
    if not isinstance(value,str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',value):
        raise ValueError('ID 仅允许小写字母、数字、下划线和短横线，长度 1–64')
    return value

def text_field(data,key):
    value=data.get(key)
    if not isinstance(value,str) or not value.strip() or len(value)>16000:raise ValueError('请填写 '+key)
    return value.strip()

def local_file(base,name):
    if not isinstance(name,str) or not name:raise ValueError('缺少包内文件路径')
    base=Path(base).resolve();p=(base/name).resolve()
    if not p.is_relative_to(base) or not p.is_file():raise ValueError('文件缺失或路径越过导入包目录：'+name)
    return p

def runtime_config(data):
    cfg=dict(action_horizon=16,steps_per_replan=4,dt_s=.05,inference_timeout_s=2.,cameras=['main'])
    cfg.update(data.get('runtime',{}))
    if set(cfg)!= {'action_horizon','steps_per_replan','dt_s','inference_timeout_s','cameras'}:
        raise ValueError('runtime 含未知参数')
    for key,lo,hi in [('action_horizon',1,256),('steps_per_replan',1,256)]:
        v=cfg[key]
        if type(v) is not int or not lo<=v<=hi:raise ValueError(key+' 必须为 1–256 的整数')
    if cfg['steps_per_replan']>cfg['action_horizon']:raise ValueError('执行步数不得超过预测长度')
    for key,lo,hi in [('dt_s',.02,1.),('inference_timeout_s',.05,30.)]:
        v=cfg[key]
        if type(v) not in (int,float) or not np.isfinite(v) or not lo<=v<=hi:raise ValueError('无效 '+key)
    if not isinstance(cfg['cameras'],list) or not cfg['cameras'] or len(set(cfg['cameras']))!=len(cfg['cameras']) or not set(cfg['cameras'])<=set(CAMERAS) or 'main' not in cfg['cameras']:
        raise ValueError('cameras 需要包含 main，可加 left_wrist / right_wrist')
    return cfg

def validate(data,kind,base):
    if not isinstance(data,dict) or data.get('schema')!=f'nero.{kind}.v1':raise ValueError(f'需要 nero.{kind}.v1 格式的 JSON')
    identifier(data.get('id'));text_field(data,'label')
    if data['id'] in (BUILTIN_TASKS if kind=='task' else BUILTIN_MODELS):raise ValueError('不能覆盖内置 ID，请使用新的 ID')
    c=common();files=[]
    if kind=='task':
        text_field(data,'instruction');text_field(data,'success_definition')
        ready=data.get('ready',{})
        if ready.get('method')=='explicit':
            q=np.deg2rad(ready.get('joints_deg'))
            if q.shape!=(2,7):raise ValueError('起手位需要右、左各 7 个关节角')
            c.limits(q)
        elif ready.get('method')=='training_mean':
            p=local_file(base,ready.get('states_file'));files.append(p)
            with np.load(p,allow_pickle=False) as z:states=np.asarray(z['start_states'],float)
            if states.ndim!=2 or states.shape[1]!=16 or not 1<=len(states)<=100000 or not np.isfinite(states).all():raise ValueError('start_states 必须为 N×16 有限数组，每条示教一条起始状态')
            # These are measured demonstrations, not commands. Validate the
            # computed target; retain out-of-envelope source poses for auditing.
            mean=states.mean(0);c.limits(np.stack([mean[:7],mean[8:15]]))
            if np.any((states[:,[7,15]]<0)|(states[:,[7,15]]>1)):raise ValueError('训练夹爪闭合度应在 0–1')
        else:raise ValueError('ready.method 需要 explicit 或 training_mean')
        w=np.asarray(ready.get('gripper_width_m'),float)
        if w.shape!=(2,) or not np.isfinite(w).all() or np.any((w<0)|(w>.1)):raise ValueError('起手夹爪宽度应为右、左两个 0–0.1 m 数值')
    else:
        if data.get('action_contract')!=CONTRACT:raise ValueError('模型必须适配 NERO 双臂绝对弧度 / 夹爪闭合度协议')
        runtime_config(data)
        tasks=data.get('tasks',[])
        if not isinstance(tasks,list) or not tasks or any(x!='*' and (not isinstance(x,str) or not re.fullmatch(r'[a-z0-9][a-z0-9_-]{0,63}',x)) for x in tasks):raise ValueError('tasks 需要任务 ID 列表；通用语言模型可用 ["*"]')
        if data.get('adapter')=='websocket':
            uri=urlparse(text_field(data,'endpoint'))
            if uri.scheme not in ('ws','wss') or not uri.hostname or uri.username or uri.password or uri.fragment:raise ValueError('endpoint 需要不含密码的 ws:// 或 wss:// 地址')
            if not re.fullmatch(r'[0-9a-f]{64}',str(data.get('checkpoint_sha256',''))):raise ValueError('checkpoint_sha256 需要实际权重的 SHA256（64 位小写十六进制）')
            text_field(data,'service_model_id')
            if data.get('checkpoint_file'):
                p=local_file(base,data['checkpoint_file']);files.append(p)
                if c.sha(p)!=data['checkpoint_sha256']:raise ValueError('导入权重文件 SHA256 不匹配')
        elif data.get('adapter')=='sealed_policy':
            for key in ('policy_file','verification_file'):
                files.append(local_file(base,data.get(key)))
            v=c.read(local_file(base,data['verification_file']))
            if c.sha(local_file(base,data['policy_file']))!=v['policy_sha256']:raise ValueError('权重与 verification.json 不匹配')
            cfg=runtime_config(data)
            if cfg['action_horizon']!=16 or cfg['cameras']!=['main']:raise ValueError('现有封存策略适配器需要 16 步与主相机')
        elif data.get('adapter')=='openpi_nero':
            manifest_path=local_file(base,data.get('checkpoint_manifest'));files.append(manifest_path)
            manifest=c.read(manifest_path)
            if manifest.get('schema')!='nero.checkpoint.files.v1' or not isinstance(manifest.get('files'),dict) or not manifest['files']:raise ValueError('缺少检查点文件清单')
            if c.digest(manifest['files'])!=data.get('checkpoint_sha256'):raise ValueError('检查点清单摘要不匹配')
            checkpoint=Path(base)/data.get('checkpoint_dir','checkpoint')
            for name,expected in manifest['files'].items():
                p=local_file(checkpoint,name)
                if not p.is_relative_to(Path(base).resolve()):raise ValueError('检查点路径越过包目录')
                if c.sha(p)!=expected:raise ValueError('检查点文件校验失败：'+name)
                files.append(p)
            if not any(x.startswith('params/') for x in manifest['files']) or not any(x.startswith('assets/') for x in manifest['files']):raise ValueError('检查点需要 params 和归一化 assets')
            if not re.fullmatch(r'pi0[5]?_nero_[a-z0-9_]+',str(data.get('train_config',''))):raise ValueError('需要已适配 NERO 的训练配置')
            cfg=runtime_config(data)
            if set(cfg['cameras'])!=set(CAMERAS) or cfg['dt_s']!=.05:raise ValueError('本地 NERO 模型需要三路相机和训练时的 20 Hz 动作间隔')
            widths=np.asarray(data.get('native_gripper_max_width_m'),float)
            if widths.shape!=(2,) or not np.isfinite(widths).all() or np.any((widths<.08)|(widths>.11)):raise ValueError('需要左右夹爪训练标定宽度')
            if type(data.get('noise_seed',0)) is not int or not 0<=data.get('noise_seed',0)<2**32:raise ValueError('noise_seed 应为非负 32 位整数')
        else:raise ValueError('adapter 需要 websocket、sealed_policy 或 openpi_nero')
    if data.get('probe_file'):
        p=local_file(base,data['probe_file']);files.append(p)
        validate_probe(p,runtime_config(data)['cameras'] if kind=='model' else ['main'])
    if data.get('golden_file'):
        if kind!='model' or not data.get('probe_file'):raise ValueError('golden 需要模型包自带 probe_file')
        p=local_file(base,data['golden_file']);files.append(p)
        with np.load(p,allow_pickle=False) as z:g=z['expected_raw_action_chunks']
        with np.load(local_file(base,data['probe_file']),allow_pickle=False) as z:n=len(z['state'])
        if g.shape!=(n,runtime_config(data)['action_horizon'],16) or not np.isfinite(g).all():raise ValueError('golden 动作形状或数值错误')
    return list(dict.fromkeys(files))

def validate_probe(path,cameras):
    with np.load(path,allow_pickle=False) as z:
        states=z['state']
        if states.ndim!=2 or states.shape[1]!=16 or not 1<=len(states)<=1000 or not np.isfinite(states).all():raise ValueError('probe state 需要 N×16 有限数组')
        common().limits(np.stack([states[:,:7],states[:,8:15]],axis=1))
        if np.any((states[:,[7,15]]<0)|(states[:,[7,15]]>1)):raise ValueError('probe 夹爪闭合度应在 0–1')
        for camera in cameras:
            image=z['rgb' if camera=='main' else 'rgb_'+camera]
            if image.dtype!=np.uint8 or image.ndim!=4 or image.shape[0]!=len(states) or image.shape[-1]!=3 or min(image.shape[1:3])<1:raise ValueError('probe 相机需要 N×H×W×3 uint8 RGB：'+camera)

def import_package(path,kind):
    c=common();path=Path(path).expanduser().resolve()
    if path.is_dir():path=path/f'{kind}.json'
    if kind not in ('task','model'):raise ValueError('invalid package kind')
    data=c.read(path);files=validate(data,kind,path.parent)
    parent=root()/(kind+'s');parent.mkdir(parents=True,exist_ok=True);target=parent/data['id']
    if target.exists():raise ValueError('ID 已存在，未覆盖。请更换 ID 后导入：'+data['id'])
    labels=scan(kind)
    if data['label'] in [x['label'] for x in labels.values()] or data['label'] in (c.TASKS.values() if kind=='task' else c.MODELS.values()):raise ValueError('名称已存在，请使用不同名称')
    stage=Path(tempfile.mkdtemp(prefix='.import-',dir=parent))
    try:
        for src in files:
            dst=stage/src.relative_to(path.parent);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
        c.write(stage/f'{kind}.json',data)
        validate(data,kind,stage)
        hashes={str(p.relative_to(stage)).replace('\\','/'):c.sha(p) for p in stage.rglob('*') if p.is_file()}
        c.write(stage/'IMPORT.json',dict(imported_unix_s=time.time(),source=str(path),files=hashes))
        # The legacy loader expects a sealed tree; it remains an isolated adapter.
        if kind=='model' and data['adapter']=='sealed_policy':
            export=stage/'sealed';policy=export/'task/model/policy.pt';policy.parent.mkdir(parents=True)
            shutil.copy2(stage/data['policy_file'],policy)
            c.write(export/'manifest.json',{'files':{'task/model/policy.pt':{'sha256':c.sha(policy)}}})
            c.write(export/'EXPORT_COMPLETE',{'manifest_sha256':c.sha(export/'manifest.json')})
        stage.rename(target)
    except BaseException:
        shutil.rmtree(stage);raise
    c.refresh_catalog();return data['id']

def scan(kind):
    result={}
    for p in sorted((root()/(kind+'s')).glob(f'*/{kind}.json')):
        if p.parent.name.startswith('.'):continue
        try:
            d=common().read(p)
            if d.get('id')!=p.parent.name or d.get('schema')!=f'nero.{kind}.v1':continue
            seal=common().read(p.parent/'IMPORT.json')
            if common().sha(p)!=seal['files'][f'{kind}.json']:continue
            if not isinstance(d.get('label'),str) or (kind=='task' and not isinstance(d.get('success_definition'),str)):continue
            result[d['id']]=d
        except (ValueError,OSError,KeyError,TypeError):continue
    return result

def package(kind,key,verify=False):
    p=root()/(kind+'s')/identifier(key);c=common();data=c.read(p/f'{kind}.json');seal=c.read(p/'IMPORT.json')
    if c.sha(p/f'{kind}.json')!=seal['files'][f'{kind}.json']:raise ValueError('导入配置已改变，请使用新 ID 重新导入')
    if verify:
        for name,expected in seal['files'].items():
            if c.sha(local_file(p,name))!=expected:raise ValueError('导入文件校验失败：'+name)
    return data,p,seal

def task_definition(task):
    c=common()
    if task in BUILTIN_TASKS:return dict(id=task,label=c.TASKS[task],instruction=c.TASKS[task],success_definition=c.SUCCESS[task],ready={'gripper_width_m':[0. if task=='banana' else .1,.1]})
    return package('task',task)[0]

def compatible(task,model):
    if model in BUILTIN_MODELS:return task in BUILTIN_TASKS
    try:
        data=package('model',model)[0];return '*' in data['tasks'] or task in data['tasks']
    except (OSError,ValueError,KeyError):return False

def models_for(task):return {k:v for k,v in common().MODELS.items() if compatible(task,k)}

def task_pose(task):
    data,p,_=package('task',task,True);r=data['ready'];c=common()
    if r['method']=='training_mean':
        with np.load(p/r['states_file'],allow_pickle=False) as z:states=np.asarray(z['start_states'],float)
        mean=states.mean(0);q=c.limits(np.stack([mean[:7],mean[8:15]]));std=states.std(0);n=len(states)
        widths=.1*(1-states[:,[7,15]]);estimator='每条示教一帧，起始状态等权均值'
    else:
        q=c.limits(np.deg2rad(r['joints_deg']));widths=np.array([r['gripper_width_m']]);std=np.zeros(16);n=0;estimator='导入任务指定的参考姿态'
    w=widths.mean(0)
    return dict(task=task,joints_rad=q.tolist(),joints_deg=np.rad2deg(q).tolist(),gripper_closedness=(1-w/.1).tolist(),
        gripper_width_mean_m=w.tolist(),gripper_width_range_m=np.stack([widths.min(0),widths.max(0)],1).tolist(),
        train_episodes=n,estimator=estimator,ready_state_std=std.tolist(),train_excursion_deg=np.zeros((2,7)).tolist(),
        manifest_sha256=c.sha(p/'task.json'),states_file_sha256=c.sha(p/r['states_file']) if n else None)

def identity(task,model,verify=False):
    c=common();d,p,seal=package('model',model,verify)
    if not compatible(task,model):raise ValueError('模型未声明兼容此任务')
    cfg=runtime_config(d)
    if d['adapter']=='sealed_policy':
        v=c.read(p/d['verification_file']);policy_sha=v['policy_sha256'];encoder=v['encoder_state_sha256']
    else:policy_sha=d['checkpoint_sha256'];encoder=''
    task_data=task_definition(task)
    return dict(task=task,model=model,adapter=d['adapter'],policy_sha256=policy_sha,encoder_state_sha256=encoder,
        manifest_sha256=c.sha(p/'model.json'),policy_path=str(p/'sealed/task/model/policy.pt') if d['adapter']=='sealed_policy' else str(p/d['checkpoint_file']) if d.get('checkpoint_file') else None,
        model_definition=d,task_definition=task_data,task_sha256=c.digest(task_data),runtime=cfg,
        instruction=task_data['instruction'],action_contract=CONTRACT,package_path=str(p))

def probe_paths(task,model):
    c=common()
    if model in BUILTIN_MODELS:
        manifest=c.read(c.EXPORT/'manifest.json')['files'];names=[f'probes/{task}.npz',f'{task}/{model}/golden.npz']
        for n in names:
            if c.sha(c.EXPORT/n)!=manifest[n]['sha256']:raise ValueError('probe/golden 哈希校验失败')
        return c.EXPORT/names[0],c.EXPORT/names[1]
    d,p,_=package('model',model,True)
    if d.get('probe_file'):return p/d['probe_file'],p/d['golden_file'] if d.get('golden_file') else None
    if task not in BUILTIN_TASKS:
        t,tp,_=package('task',task,True)
        if t.get('probe_file'):return tp/t['probe_file'],None
    return None,None

def observation(identity,rgb,state,extra=None):
    return dict({'observation/exterior_image_1_left':rgb,'observation/state':np.asarray(state,np.float32),
                 'prompt':identity.get('instruction',task_definition(identity['task'])['instruction']),
                 'task_id':identity['task']},**(extra or {}))

def export_templates(directory):
    c=common();directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    target=directory/('nero_import_templates_'+time.strftime('%Y%m%d_%H%M%S'));target.mkdir()
    c.write(target/'task.json',dict(schema='nero.task.v1',id='my_task',label='我的任务',instruction='Pick up the object and place it in the tray.',success_definition='物体完全释放并在托盘内稳定 2 秒',
        ready=dict(method='explicit',joints_deg=c.DEFAULT_DEG.tolist(),gripper_width_m=[.1,.1])))
    c.write(target/'model.json',dict(schema='nero.model.v1',id='my_vla',label='我的 VLA',adapter='websocket',tasks=['*'],endpoint='ws://127.0.0.1:9000',
        service_model_id='my_vla',checkpoint_sha256='REPLACE_WITH_ACTUAL_CHECKPOINT_SHA256',action_contract=CONTRACT,
        runtime=dict(action_horizon=16,steps_per_replan=4,dt_s=.05,inference_timeout_s=2.,cameras=['main'])))
    c.write(target/'sealed_model.json',dict(schema='nero.model.v1',id='my_trained_policy',label='我的训练模型',adapter='sealed_policy',tasks=['plate'],
        action_contract=CONTRACT,policy_file='policy.pt',verification_file='verification.json',probe_file='probe.npz',golden_file='golden.npz',
        runtime=dict(action_horizon=16,steps_per_replan=4,dt_s=.05,inference_timeout_s=2.,cameras=['main'])))
    shutil.copy2(Path(__file__).with_name('NERO_MODEL_TASK_IMPORT.md'),target/'使用说明.md')
    return target

def import_sealed(policy,model_id,label,tasks):
    """Bring a completed training export into managed storage, never raw pickle."""
    policy=Path(policy).resolve();verification=policy.with_name('verification.json');c=common()
    if not verification.is_file():raise ValueError('此权重缺少同目录 verification.json。其他架构请使用 VLA 服务适配器与 model.json 导入。')
    data=dict(schema='nero.model.v1',id=model_id,label=label,tasks=tasks,adapter='sealed_policy',
              action_contract=CONTRACT,policy_file='policy.pt',verification_file='verification.json')
    with tempfile.TemporaryDirectory(prefix='nero-model-import-') as tmp:
        stage=Path(tmp);shutil.copy2(policy,stage/'policy.pt');shutil.copy2(verification,stage/'verification.json')
        probes=policy.parents[2]/'probes'/f'{policy.parent.parent.name}.npz'
        if probes.is_file():shutil.copy2(probes,stage/'probe.npz');data['probe_file']='probe.npz'
        golden=policy.with_name('golden.npz')
        if probes.is_file() and golden.is_file():shutil.copy2(golden,stage/'golden.npz');data['golden_file']='golden.npz'
        c.write(stage/'model.json',data);return import_package(stage/'model.json','model')
