"""Local NERO-trained pi-family policy, with explicit camera/gripper mapping."""
from pathlib import Path
import time
import numpy as np
from .policy_protocol import actions,validate_observation

def native_observation(obs,max_widths):
    state=np.asarray(obs['observation/state'],np.float32).copy()
    widths=.1*(1-state[[7,15]])
    state[[7,15]]=np.clip(1-widths/np.asarray(max_widths),0,1)
    return {'observation/state':state,'observation/exterior_image_1_left':obs['observation/exterior_image_1_left'],
        'observation/wrist_image_left':obs['observation/right_wrist_image'],
        'observation/wrist_image_right':obs['observation/left_wrist_image'],'prompt':obs['prompt']}

def platform_actions(native,max_widths):
    a=np.asarray(native,np.float32).copy()
    # Preserve raw model values; physical [0,1] clipping remains in the common
    # controller. Calibration maps widths, not left/right channel order.
    a[..., [7,15]]=1-np.asarray(max_widths)/.1*(1-a[..., [7,15]])
    return a

class NativePolicy:
    def __init__(self,identity):
        from openpi.training import config
        from openpi.policies import policy_config
        d=identity['model_definition'];cfg=config.get_config(d['train_config']);meta=cfg.policy_metadata
        if meta.get('mode')!='nero_native' or meta.get('arm')!='both' or meta.get('robot_action_dimensions')!=16 or meta.get('action_semantics')!='absolute_joint_target_rad+absolute_gripper_closedness':
            raise ValueError('此模型配置不是 NERO 双臂绝对关节模型，禁止直接接入机器人动作')
        if cfg.model.action_horizon!=identity['runtime']['action_horizon']:raise ValueError('预测长度与训练配置不一致')
        self.identity=identity;self.max_widths=d['native_gripper_max_width_m']
        checkpoint=Path(identity['package_path'])/d['checkpoint_dir']
        self.policy=policy_config.create_trained_policy(cfg,checkpoint,sample_kwargs={'num_steps':10})
        self.noise=np.random.default_rng(d.get('noise_seed',0)).standard_normal((cfg.model.action_horizon,cfg.model.action_dim)).astype(np.float32)
        self.metadata=dict(identity['model_definition'],native_metadata=meta)
        print('已加载本地 NERO 双臂模型；固定采样噪声种子 '+str(d.get('noise_seed',0)),flush=True)
    def infer(self,obs):
        validate_observation(obs,self.identity['runtime']['cameras']);start=time.monotonic()
        result=self.policy.infer(native_observation(obs,self.max_widths),noise=self.noise)
        a=actions(platform_actions(result['actions'],self.max_widths),self.identity['runtime']['action_horizon'])
        return {'actions':a,'server_timing':{'infer_ms':(time.monotonic()-start)*1000,'native_policy_timing':result.get('policy_timing')}}
