from __future__ import annotations
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid
import numpy as np

PACKAGE=Path(__file__).resolve().parent
PLATFORM=Path(os.environ.get('NERO_PLATFORM_ROOT',str(PACKAGE.parent))).expanduser().resolve()
ROOT=PLATFORM/'runtime/openpi'
EXPORT=PLATFORM/'models/act_export'
RUNS=PLATFORM/'runs'
DEFAULT_EXPORTS=PLATFORM/'exports'
PREFERENCES=PLATFORM/'config/preferences.json'
LEGACY=ROOT/'examples/nero_pi05_bridge/plate_ready_v2'
TASKS={'plate':'双臂端盘子','banana':'右手向左手传香蕉','holder':'将笔放入矩形篮子'}
MODELS={'mpi-base':'MPI-base','p0':'P0 · step 6000','g2':'G2 · scene seed42 step2000'}
SUCCESS={'plate':'盘子完全离桌，无人扶持稳定至少 2 秒',
         'banana':'右手释放，左手独立持蕉至少 2 秒',
         'holder':'笔释放后稳定留在矩形篮内至少 2 秒'}
DEFAULT_DEG=np.array([[0,90,90,90,0,0,0],[0,90,-90,90,0,0,0]],float)
LOWER=np.array([-2.68526,-1.72,-2.73,-.99,-2.73,-.71,-1.5507963])
UPPER=np.array([2.68526,1.72,2.73,2.12,2.73,.93,1.5507963])

def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    temp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def new_session(kind):
    RUNS.mkdir(parents=True,exist_ok=True)
    path=RUNS/(time.strftime('%Y%m%d_%H%M%S')+'_'+str(time.time_ns()%1_000_000_000)+f'_{kind}')
    path.mkdir(exist_ok=False)
    return path

@dataclass(frozen=True)
class Settings:
    task: str='plate'
    model: str='mpi-base'
    duration_s: float=180.
    excursion_deg: float=45.
    tcp_cm: float=20.
    operator: str='现场操作员'
    layout: str='layout_001'
    repeats_per_group: int=20
    relative_limits_enabled: bool=False
    def checked(self):
        if self.task not in TASKS or self.model not in MODELS:raise ValueError('请选择已有任务与模型')
        if not isinstance(self.relative_limits_enabled,bool):raise ValueError('相对范围开关必须为布尔值')
        for k,lo,hi in [('duration_s',1,600),('excursion_deg',5,90),('tcp_cm',2,40)]:
            v=getattr(self,k)
            if not isinstance(v,(int,float)) or isinstance(v,bool) or not math.isfinite(v) or not lo<=v<=hi:
                raise ValueError(f'{k} 应在 {lo}–{hi} 范围内')
        if not 1<=self.repeats_per_group<=1000:raise ValueError('每组次数无效')
        if not self.operator.strip() or not self.layout.strip():raise ValueError('请填写操作员和布局编号')
        return self
    def json(self):return asdict(self.checked())
    def protocol(self):
        from .library import BUILTIN_MODELS,package,runtime_config,task_definition
        cfg=runtime_config(package('model',self.model)[0]) if self.model not in BUILTIN_MODELS else dict(dt_s=.05,steps_per_replan=4,action_horizon=16,inference_timeout_s=.15,cameras=['main'])
        return dict(version='nero_desktop.driver_targets.v7',duration_s=self.duration_s,
                    model_runtime=cfg,task_definition=task_definition(self.task),
                    target_filter='firmware_move_j_absolute_targets',
                    reset_control='firmware_move_j_segment_endpoints',
                    ready_gripper_control='close_command_with_contact_feedback',
                    training_start_check='advisory_only',startup_gripper_command='preserve_existing',
                    raw_joint_limit_scope='executed_prefix_only_unused_tail_warning',
                    relative_limits_enabled=self.relative_limits_enabled,
                    excursion_deg=self.excursion_deg if self.relative_limits_enabled else None,
                    tcp_cm=self.tcp_cm if self.relative_limits_enabled else None,
                    link_displacement_cm=self.tcp_cm+5 if self.relative_limits_enabled else None,
                    dt_s=cfg['dt_s'],steps_per_replan=cfg['steps_per_replan'],joint_speed_rad_s=None,joint_accel_rad_s2=None,
                    target_lead_rad=None,hard_lead_rad=None,gripper_speed_m_s=None,gripper_force_n=1.,startup_hold_s=1.,
                    measured_speed_abort_rad_s=None,speed_owner='driver_startup_speed_percent_and_firmware',
                    scheduler_gap_max_s=.35,scheduler_rebase_after_s=.15,video_gap_warning_s=.25,video_gap_abort_s=.5)

def limits(q):
    q=np.asarray(q,float)
    if q.shape[-2:]!=(2,7) or not np.isfinite(q).all():raise ValueError('关节角必须是有限的 2×7 数组')
    if np.any(q<LOWER) or np.any(q>UPPER):raise ValueError('关节目标超出已验证软限位')
    return q

def ready_pose(task):
    if task not in TASKS:raise ValueError('未知任务')
    from .library import BUILTIN_TASKS,task_pose
    if task not in BUILTIN_TASKS:return task_pose(task)
    report=read(PACKAGE/'assets'/f'{task}_ready.json')
    states_path=PACKAGE/'assets'/f'{task}_train_states.npz'
    if sha(states_path)!=report['states_file_sha256']:raise ValueError('训练姿态缓存哈希不匹配')
    with np.load(states_path,allow_pickle=False) as data:
        starts=np.array([data[row['episode']][0] for row in report['evidence']],dtype=float)
    mean=starts.mean(0)
    if len(starts)!=report['train_episodes'] or not np.allclose(mean,report['ready_state_mean'],atol=1e-9,rtol=0):
        raise ValueError('训练起始帧复算不一致')
    q=limits(np.stack([mean[:7],mean[8:15]]))
    return dict(task=task,joints_rad=q.tolist(),joints_deg=np.rad2deg(q).tolist(),
                gripper_closedness=mean[[7,15]].tolist(),train_episodes=len(starts),
                gripper_width_mean_m=(.1*(1-mean[[7,15]])).tolist(),
                gripper_width_range_m=np.stack([.1*(1-starts[:,[7,15]].max(0)),.1*(1-starts[:,[7,15]].min(0))],axis=1).tolist(),
                estimator=report['estimator'],ready_state_std=report['ready_state_std'],
                train_excursion_deg=report['training_max_joint_excursion_from_mean_ready_deg'],
                manifest_sha256=report['manifest_sha256'],states_file_sha256=report['states_file_sha256'])

def ready_gripper_target(task,pose=None):
    """Command widths, not observed widths around a held object (closedness 1/0)."""
    if task not in TASKS:raise ValueError('未知任务')
    from .library import task_definition
    return task_definition(task)['ready']['gripper_width_m']

def check_start_grippers(task,widths,pose=None):
    pose=pose or ready_pose(task);w=np.asarray(widths,float)
    if w.shape!=(2,) or not np.isfinite(w).all() or np.any((w<-.001)|(w>.101)):
        raise ValueError('夹爪反馈无效：需要物理范围内的有限开度')
    bounds=np.array([[.095,.101],[.095,.101]])
    from .library import BUILTIN_TASKS
    if task not in BUILTIN_TASKS:bounds=np.asarray(pose['gripper_width_range_m'])+[-.002,.002]
    if task=='banana':
        # Demonstrations begin with the banana already in the right gripper.
        # Recorded training range plus 2 mm measurement/object allowance.
        bounds[0]=np.array(pose['gripper_width_range_m'][0])+[-.002,.002]
    within=bool(np.all((w>=bounds[:,0])&(w<=bounds[:,1])))
    reference='；'.join(f'{arm} {lo*1000:.1f}–{hi*1000:.1f} mm' for arm,(lo,hi) in zip(('右爪','左爪'),bounds))
    warnings=[] if within else [f'当前夹爪开度右 {w[0]*1000:.1f} / 左 {w[1]*1000:.1f} mm 偏离训练起手参考（{reference}）；仅提示，不阻止推理。']
    if task=='banana':warnings.append('香蕉任务需要右手实际持蕉；模型闭合指令为1（驱动0 mm），持物实测开度可以大于0，开度本身不能证明持物。')
    return dict(expected_width_range_m=bounds.tolist(),actual_width_m=w.tolist(),within_training_range=within,
                enforcement='advisory_only',warnings=warnings,object_presence_verified=False)

def catalog(task,model,verify_policy=False):
    if task not in TASKS or model not in MODELS:raise ValueError('任务或模型不在封存清单中')
    from .library import BUILTIN_MODELS,BUILTIN_TASKS,identity
    if model not in BUILTIN_MODELS:return identity(task,model,verify_policy)
    if task not in BUILTIN_TASKS:raise ValueError('内置模型仅兼容原有任务，请选择为此任务导入的模型')
    seal=read(EXPORT/'EXPORT_COMPLETE')
    if sha(EXPORT/'manifest.json')!=seal['manifest_sha256']:raise ValueError('导出清单封存校验失败')
    manifest=read(EXPORT/'manifest.json')['files'];prefix=f'{task}/{model}'
    vpath=EXPORT/prefix/'verification.json'
    if sha(vpath)!=manifest[prefix+'/verification.json']['sha256']:raise ValueError('模型身份记录哈希不匹配')
    data=read(vpath);expected=manifest[prefix+'/policy.pt']['sha256']
    if data['policy_sha256']!=expected:raise ValueError('策略身份不一致')
    if verify_policy and sha(EXPORT/prefix/'policy.pt')!=expected:raise ValueError('策略文件哈希不匹配')
    return dict(task=task,model=model,instruction=TASKS[task],policy_sha256=expected,encoder_state_sha256=data['encoder_state_sha256'],
                manifest_sha256=seal['manifest_sha256'],policy_path=str(EXPORT/prefix/'policy.pt'))

def source_identity():
    files=list(PACKAGE.glob('*.py'))+[LEGACY/n for n in ('plate_startpose.py','plate_approved_step.py','plate_can_observer.py','plate_shadow_audit_v2.json')]
    files+=list((PACKAGE/'assets').glob('*_ready.json'))+list((PACKAGE/'assets').glob('*_train_states.npz'))
    files+=list((PLATFORM/'library/tasks').glob('*/task.json'))+list((PLATFORM/'library/models').glob('*/model.json'))
    bridge=ROOT/'examples/nero_pi05_bridge/nero_pi05_bridge'
    files+=[bridge/n for n in ('policy_client.py','observation.py','act_shadow_ros.py','nero_fk.py')]
    return {str(p.resolve()):sha(p) for p in files}

def approve(plan,directory,operator):
    # Called only by the local confirmation button, never by prepare/background jobs.
    if not operator.strip():raise ValueError('缺少操作员')
    approval=dict(status='approved',source='local_desktop_confirm',operator=operator,
                  plan_sha256=plan['plan_sha256'],approved_unix_s=time.time(),operator_present_with_estop=True)
    path=Path(directory)/'approval.json'
    if path.exists():raise ValueError('该计划已有许可，请使用新计划')
    write(path,approval)
    return approval

def check_approval(plan,approval,now=None):
    now=time.time() if now is None else now
    clean={k:v for k,v in plan.items() if k!='plan_sha256'}
    if plan.get('plan_sha256')!=digest(clean):raise ValueError('计划指纹已改变')
    if not 0<=now-plan['prepared_unix_s']<=1800:raise ValueError('计划过期，请重新准备')
    if approval.get('status')!='approved' or approval.get('source')!='local_desktop_confirm':raise ValueError('缺少本次桌面确认')
    if approval.get('plan_sha256')!=plan['plan_sha256'] or not approval.get('operator','').strip():raise ValueError('许可与计划不匹配')
    if approval.get('operator_present_with_estop') is not True:raise ValueError('尚未确认现场就绪')
    stamp=approval.get('approved_unix_s',-1)
    if not plan['prepared_unix_s']<=stamp<=now or now-stamp>300:raise ValueError('许可过期')

def summarize_records():
    from .records import scan,summarize
    return [dict(g,protocol=g['protocol'][:12]) for g in summarize(scan(RUNS))]

_BUILTIN_TASK_LABELS=TASKS.copy()
_BUILTIN_MODEL_LABELS=MODELS.copy()
_BUILTIN_SUCCESS=SUCCESS.copy()
def refresh_catalog():
    from .library import scan
    TASKS.clear();TASKS.update(_BUILTIN_TASK_LABELS)
    MODELS.clear();MODELS.update(_BUILTIN_MODEL_LABELS)
    SUCCESS.clear();SUCCESS.update(_BUILTIN_SUCCESS)
    for k,d in scan('task').items():
        if k not in _BUILTIN_TASK_LABELS:TASKS[k]=d['label'];SUCCESS[k]=d['success_definition']
    for k,d in scan('model').items():
        if k not in _BUILTIN_MODEL_LABELS:MODELS[k]=d['label']

refresh_catalog()
