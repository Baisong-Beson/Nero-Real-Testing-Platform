"""Architecture independent NERO observation/action contract."""
import numpy as np
from .library import CONTRACT, runtime_config

SCHEMA='nero.policy.v1'

def metadata(model_id,checkpoint_sha256,runtime):
    runtime=runtime_config({'runtime':runtime})
    return dict(schema=SCHEMA,service_model_id=model_id,policy_sha256=checkpoint_sha256,
                commandable=False,action_contract=CONTRACT,runtime=runtime)

def validate_service(actual,definition):
    expected=metadata(definition['service_model_id'],definition['checkpoint_sha256'],runtime_config(definition))
    for k,v in expected.items():
        if actual.get(k)!=v or (k=='commandable' and actual.get(k) is not False):raise ValueError(f'模型服务元数据不匹配 {k}: 期望 {v}，得到 {actual.get(k)}')

def validate_metadata(actual,task,model,policy_sha):
    # Existing sealed models retain their exact, stronger identity validation.
    from .common import catalog
    expected=catalog(task,model)
    if not expected.get('adapter'):
        from nero_pi05_bridge.act_shadow_ros import validate_metadata as legacy
        return legacy(actual,task,model,policy_sha)
    for k,v in dict(schema=SCHEMA,task=task,model=model,policy_sha256=policy_sha,
                    commandable=False,action_contract=CONTRACT,runtime=expected['runtime'],
                    task_sha256=expected['task_sha256'],manifest_sha256=expected['manifest_sha256']).items():
        if actual.get(k)!=v or (k=='commandable' and actual.get(k) is not False):raise ValueError('平台模型身份或协议不匹配：'+k)

def actions(value,horizon):
    a=np.asarray(value,dtype=np.float32)
    if a.shape!=(horizon,16) or not np.isfinite(a).all():raise ValueError(f'模型应返回有限的绝对动作 [{horizon},16]')
    return a

def validate_observation(obs,cameras):
    from .library import CAMERAS
    state=np.asarray(obs['observation/state'])
    if state.shape!=(16,) or not np.isfinite(state).all():raise ValueError('state 需要 16 维有限数值')
    if not isinstance(obs.get('prompt'),str) or not obs['prompt'].strip():raise ValueError('缺少任务语言指令 prompt')
    for camera in cameras:
        a=np.asarray(obs[CAMERAS[camera]])
        if a.dtype!=np.uint8 or a.ndim!=3 or a.shape[-1]!=3:raise ValueError('相机需要 H×W×3 uint8 RGB：'+camera)
