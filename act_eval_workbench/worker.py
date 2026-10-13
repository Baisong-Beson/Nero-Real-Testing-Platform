"""One job per process. Import/prepare/shadow never creates motion publishers."""
from __future__ import annotations
import argparse
import asyncio
import contextlib
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback
import cv2
import numpy as np
from .common import *
from . import controller as c
from . import reset_motion as reset

class Operator:
    def __init__(self,directory,heartbeat):
        self.directory=Path(directory);self.heartbeat=Path(heartbeat);self.closed=False
        self.text=self;self.last_write=0.;self.last_check=0.;self.cancelled=False
    @property
    def stopped(self):
        if self.cancelled:return True
        now=time.monotonic()
        if now-self.last_check>.05:
            self.last_check=now
            try:
                age=time.time()-self.heartbeat.stat().st_mtime
                self.cancelled=(self.directory/'STOP').exists() or not -.1<=age<=2.
            except OSError:self.cancelled=True
        return self.cancelled
    def pump(self):pass
    def set(self,text):
        now=time.monotonic()
        if now-self.last_write>=.2:
            write(self.directory/'progress.json',dict(text=str(text),updated_unix_s=time.time()))
            self.last_write=now
    def permit(self):return not self.stopped and not self.closed

class PositionIO(c.diag.CameraIO):
    def __init__(self,output,gui=None):
        super().__init__(output,gui)
        # Joint-only positioning preserves the gripper; opening is unnecessary.
        self.feedback=c.pos.Feedback(require_open=False)
    def snapshot(self,stationary=False):
        q,w=super().snapshot(stationary)
        if hasattr(self,'micro_origin'):
            offset=float(np.max(np.abs(q-self.micro_origin)))
            self.micro_max_observed=max(getattr(self,'micro_max_observed',0.),offset)
            if offset>np.deg2rad(.5):raise RuntimeError('微动测试实测变化超过 0.5°，停止')
        return q,w
    def preflight(self):
        for attempt in range(3):
            try:return super().preflight()
            except RuntimeError as exc:
                if 'need 0.5 s of unique stationary feedback' not in str(exc) or attempt==2:raise
                self.event('waiting_for_stationary_history',attempt=attempt+1)
                if self.interrupted or (self.gui and self.gui.stopped):raise RuntimeError('定位准备已取消')
    def call_pair(self,service,value,permit=None):
        result=super().call_pair(service,value,permit)
        if service=='enable_agx_arm' and value:
            # Hold measured angles immediately after enabling, before opening gates.
            super().call_pair('emergency_stop',None,permit)
        return result

class ResetIO(c.FormalIO):
    """Open, position arms, then restore the task gripper target at the goal."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        from .gripper_contact import ContactFeedback
        from agx_arm_msgs.msg import GripperStatus
        from rclpy.qos import QoSProfile,ReliabilityPolicy
        self.right_contact=ContactFeedback()
        def receive(msg):
            fault=any(getattr(msg,key) for key in ('voltage_too_low','motor_overheating','driver_overcurrent','driver_overheating','driver_error_status'))
            self.right_contact.add(msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,self.now(),
                msg.width,msg.force,msg.driver_enable_status,fault)
        self.subs.append(self.node.create_subscription(GripperStatus,'/right_arm/feedback/gripper_status',receive,
            QoSProfile(depth=10,reliability=ReliabilityPolicy.BEST_EFFORT)))
    def gripper_contact(self,since,widths):
        return self.right_contact.contact(self.now(),self.node.get_clock().now().nanoseconds/1e9,since,widths[0])
    def snapshot(self,stationary=False):
        try:
            q,widths=super().snapshot(stationary)
            if hasattr(self,'reset_path'):
                if not hasattr(self,'reset_actual_geometry'):
                    self.reset_actual_geometry=c.Geometry(self.reset_path['start'],self.reset_path['gripper_widths_m'][0])
                self.reset_actual_geometry.check(q,widths)
            return q,widths
        except RuntimeError as exc:
            if 'measured speed' in str(exc) and not getattr(self,'speed_fault_logged',False):
                self.speed_fault_logged=True
                self.event('reset_speed_fault_feedback',error=str(exc),history={arm:[
                    dict(source_unix_s=r['stamp'],received_monotonic_s=r['received'],joints_rad=r['q'].tolist())
                    for r in rows if self.now()-r['received']<=.8] for arm,rows in self.feedback.rows.items()})
            raise
    def preflight(self):
        return c.driver_preflight(self)

    def publish(self,q,actual,index):
        _,measured=self.snapshot()
        widths=self.reset_path['gripper_widths_m'][index]
        opening_end=self.reset_path['opening_last_index']
        if index<=opening_end and np.max(np.abs(actual-self.reset_path['start']))>np.deg2rad(.5):
            raise RuntimeError('开爪期间关节偏移超过 0.5°，停止')
        final_index=self.reset_path.get('final_gripper_index')
        if final_index is not None and index==final_index:
            if np.max(np.abs(q-self.reset_path['goal']))>1e-8 or np.max(np.abs(actual-q))>reset.FINAL_TOL:
                raise RuntimeError('双臂尚未到达起手位，停止恢复任务夹爪状态')
        elif np.max(np.abs(q-self.reset_path['start']))>1e-8 and np.max(np.abs(np.asarray(measured)-.1))>.002:
            raise RuntimeError('双夹爪未张开到 100 mm（容差 2 mm），停止关节复位')
        self.publish_full(q,widths,actual,measured,index,-2,index)

def reset_plan(start,goal,widths,via_default=False,opening_only=False,final_widths=None,contact_grasp=False):
    widths=np.asarray(widths,dtype=float)
    if widths.shape!=(2,) or not np.isfinite(widths).all() or np.any(widths<-.001) or np.any(widths>.101):
        raise ValueError('夹爪起始宽度无效')
    widths=np.clip(widths,0,.1)
    final_widths=np.asarray([.1,.1] if final_widths is None else final_widths,dtype=float)
    if final_widths.shape!=(2,) or not np.isfinite(final_widths).all() or np.any((final_widths<0)|(final_widths>.1)):
        raise ValueError('夹爪目标宽度无效')
    if opening_only and np.any(final_widths!=.1):raise ValueError('失能前开爪流程必须保持双爪张开')
    if contact_grasp and (opening_only or not np.array_equal(final_widths,[0.,.1])):
        raise ValueError('持蕉接触判据仅适用于右爪闭合0 mm、左爪张开100 mm')
    opening=reset.opening(widths)
    hold=np.repeat(np.asarray(start)[None,:,:],len(opening),axis=0)
    path=reset.trajectory(start,goal,via_default)
    qs=hold if opening_only else np.concatenate([hold,path['joints'][1:]])
    ws=opening if opening_only else np.concatenate([opening,np.full((len(path['joints'])-1,2),.1)])
    path.update(joints=qs,goal=np.asarray(start if opening_only else goal),
                times=None,duration_s=None,
                gripper_widths_m=ws,opening_last_index=len(opening)-1,gripper_goal_m=final_widths.tolist(),opening_duration_s=None,
                final_gripper_index=None,gripper_completion='right_closed_or_contact' if contact_grasp else 'position')
    path['segment_end_indices']=[] if opening_only else [len(opening)-1+i for i in path['segment_end_indices']]
    if np.any(final_widths!=.1):
        closing=reset.gripper_segment(np.full(2,.1),final_widths)[1:]
        path['joints']=np.concatenate([qs,np.repeat(np.asarray(goal)[None,:,:],len(closing),axis=0)])
        path['gripper_widths_m']=np.concatenate([ws,closing])
        path['final_gripper_index']=len(path['joints'])-1
    return path

def reset_geometry(path):
    from nero_pi05_bridge.nero_fk import NeroFK
    if sha(c.pos.URDF)!=c.pos.URDF_SHA:raise RuntimeError('URDF changed')
    fk=NeroFK(c.pos.URDF);box=read(c.pos.TOOLS/'plate_shadow_audit_v2.json')['workspace_aabb']
    lo=np.array([box[k+'_min'] for k in 'xyz']);hi=np.array([box[k+'_max'] for k in 'xyz'])
    tcp=[]
    for qs,ws in zip(path['joints'],path['gripper_widths_m']):
        tcp.append([fk.tcp_xyz(qs[i],ws[i]).copy() for i in range(2)])
    tcp=np.asarray(tcp);step=float(np.linalg.norm(np.diff(tcp,axis=0),axis=-1).max(initial=0))
    if not np.isfinite(tcp).all() or np.any(tcp<lo) or np.any(tcp>hi) or step>.01:
        raise RuntimeError('复位轨迹末端超出工作范围或步长超过 10 mm')
    return dict(max_tcp_step_m=step,urdf_sha256=c.pos.URDF_SHA,collision_checked=False,
                gripper_widths_audited=True,gripper_goal_m=path['gripper_goal_m'])

def open_then_disable(directory,operator):
    """Best effort opening; always independently attempt all-motor disable."""
    io=None;opening=dict(completed=False)
    try:
        io=ResetIO(directory,operator)
        q,widths=io.preflight()
        if np.max(np.abs(np.asarray(widths)-.1))<=.002:
            # Already open: no enabling or repeated opening/holding for 14 s.
            opening.update(completed=True,already_open=True,final_widths_m=widths)
            io.event('grippers_already_open',widths_m=widths,skipped_motor_enable=True)
        else:
            path=reset_plan(q,q,widths,opening_only=True);reset_geometry(path)
            write(directory/'opening_plan.json',serialize(path));io.reset_path=path
            answer=reset.run_motion(io,path,lambda:operator.permit() and not io.interrupted,
                lambda n,total:operator.set(f'失能前张开双夹爪 {100*n/total:.0f}%；关节保持当前位置'))
            opening.update(completed=answer['ok'],final_widths_m=io.snapshot()[1])
    except Exception as exc:opening['error']=f'{type(exc).__name__}: {exc}'
    finally:
        if io:
            opening['commands_sent']=io.command_count
            try:io.close()
            except Exception as exc:opening['close_error']=str(exc)
    result=stop_services(True);result['opening']=opening
    if not opening['completed']:result['errors'].append('开爪未完成，已继续尝试全失能：'+opening.get('error','unknown'))
    if opening.get('close_error'):result['errors'].append('开爪连接收尾异常，已继续尝试全失能：'+opening['close_error'])
    return result

@contextlib.contextmanager
def motion_lock():
    fake=os.environ.get('ACT_EVAL_FAKE_DRIVER')=='1'
    if fake and (os.environ.get('ROS_DOMAIN_ID')!='174' or os.environ.get('ROS_LOCALHOST_ONLY')!='1'):
        raise RuntimeError('fake motion lock requires isolated ROS domain')
    with open('/tmp/act_eval_fake_motion_174.lock' if fake else '/tmp/plate_approved_motion.lock','a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:raise RuntimeError('另一执行程序仍持有运动锁；请先关闭已结束的旧结果窗口')
        yield

def segment(start,goal):
    start=limits(start);goal=limits(goal);dt=c.pos.DT
    distance=float(np.max(np.abs(goal-start)))
    ramp=math.ceil(max(12.,1.875*distance/c.pos.MAX_V,math.sqrt(5.773503*distance/c.pos.MAX_A))/dt)*dt
    times=np.arange(round((ramp+2)/dt)+1)*dt;u=np.clip((times-1)/ramp,0,1)
    blend=10*u**3-15*u**4+6*u**5
    return start+blend[:,None,None]*(goal-start)

def positioning_plan(start,goal,via_default=False):
    qs=segment(start,np.deg2rad(DEFAULT_DEG)) if via_default else segment(start,goal)
    if via_default:qs=np.concatenate([qs,segment(np.deg2rad(DEFAULT_DEG),goal)[1:]])
    limits(qs);v=np.diff(qs,axis=0)/c.pos.DT;a=np.diff(v,axis=0)/c.pos.DT
    if np.abs(v).max()>c.pos.MAX_V+1e-8 or np.abs(a).max()>c.pos.MAX_A+1e-8:raise ValueError('定位轨迹速度/加速度越限')
    return dict(stage='desktop_position',start=np.asarray(start),goal=np.asarray(goal),joints=qs,
                times=np.arange(len(qs))*c.pos.DT,duration_s=(len(qs)-1)*c.pos.DT,
                max_velocity_rad_s=float(np.abs(v).max()),max_acceleration_rad_s2=float(np.abs(a).max()))

def serialize(plan):return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in plan.items()}

def wait_recording_ready(io,rec,operator,minimum,start,tolerance):
    """Allow the async writer to commit its first frame before enabling motion."""
    deadline=io.now()+3.
    while rec.frames<minimum:
        if rec.error:raise RuntimeError('视频记录失败：'+rec.error)
        if not operator.permit() or io.interrupted:raise RuntimeError('等待录像期间已取消')
        io.pump();io.check_publishers();q,_=io.snapshot(stationary=True)
        if np.max(np.abs(q-np.asarray(start)))>tolerance:raise RuntimeError('等待录像期间起点发生变化')
        if io.now()>=deadline:raise RuntimeError('视频记录未就绪：录像线程未写入首批帧')
        io.idle()

def prepare(directory,kind,settings,port,operator):
    if not operator.permit():raise RuntimeError('界面停止或心跳过期')
    metadata=catalog(settings.task,settings.model) if kind=='formal' else dict(task=settings.task,model=settings.model,policy_sha256='',encoder_state_sha256='')
    c.configure(settings,metadata,port)
    body=dict(kind=kind,settings=settings.json(),protocol=settings.protocol(),protocol_sha256=digest(settings.protocol()),
              prepared_unix_s=time.time(),source_sha256=c.identity(),model_identity=metadata,
              server_port=port,success_definition=SUCCESS[settings.task],simulation_only=os.environ.get('ACT_EVAL_FAKE_DRIVER')=='1')
    if kind=='disable':
        body.update(description='保持双臂当前位置，由夹爪驱动张开到 100 mm，反馈确认张开后双臂关节与夹爪一起失能。已张开则直接全失能。会松开持物并撤去关节支撑，请确保已支撑好。开爪失败仍继续尝试全失能。',gripper_policy='open_then_disable',reset_profile=reset.PROFILE.copy())
    else:
        out=directory/'preparation';out.mkdir(exist_ok=False)
        io=c.FormalIO(out,operator) if kind=='formal' else ResetIO(out,operator) if kind in ('ready','default') else PositionIO(out,operator)
        try:
            q,widths=io.preflight();row=io.camera();image=c.diag.decode_compressed_image(row['data'])
            if hasattr(io,'driver_parameters'):
                body['driver_parameters']=io.driver_parameters
                body['protocol']['driver_parameters']=io.driver_parameters
                body['protocol_sha256']=digest(body['protocol'])
            body.update(start_joints_rad=q.tolist(),start_widths_m=widths,image_shape=list(image.shape),start_tolerance_rad=float(np.deg2rad(.2)))
            if kind=='formal':
                ready=ready_pose(settings.task)
                deviation=float(np.max(np.abs(q-np.array(ready['joints_rad']))))
                body['start_pose_check']=dict(enforcement='advisory_only',max_deviation_deg=float(np.rad2deg(deviation)),within_reference=deviation<=.08)
                body['gripper_start_check']=check_start_grippers(settings.task,widths,ready)
                body['start_warnings']=body['gripper_start_check']['warnings'].copy()
                if deviation>.08:body['start_warnings'].insert(0,f'当前关节最大偏离训练起手位 {np.rad2deg(deviation):.1f}°；将从当前位置推理，不自动复位。偏离训练起点可能影响任务表现。')
                actions,image,capture=asyncio.run(c.connected_prediction(io))
                aq,aw=c.unpack(actions,limit_steps=c.BOUNDS['steps_per_replan']);geometry=c.Geometry(q,widths);geometry.check(aq[0],aw[0])
                body.update(ready_pose=ready,first_raw_actions=actions.tolist(),capture=capture,
                            description=f'{TASKS[settings.task]} / {MODELS[settings.model]}：从已确认的当前位置开始，关节保持 1 秒且夹爪维持已有指令，随后双臂及夹爪连续策略控制，最长 {settings.duration_s:g} 秒；结束保持，不自动松开、复位。')
            else:
                goal=np.deg2rad(DEFAULT_DEG)
                if kind=='ready':
                    body['ready_pose']=ready_pose(settings.task);goal=np.array(body['ready_pose']['joints_rad'])
                if kind=='micro':
                    goal=q.copy();goal[:,3]+=np.deg2rad(.3)
                is_reset=kind in ('ready','default')
                final_widths=ready_gripper_target(settings.task,body['ready_pose']) if kind=='ready' else [.1,.1]
                task_grip=kind=='ready' and any(w!=.1 for w in final_widths)
                path=reset_plan(q,goal,widths,via_default=kind=='ready',final_widths=final_widths,contact_grasp=settings.task=='banana' and kind=='ready') if is_reset else positioning_plan(q,goal)
                audit=reset_geometry(path) if is_reset else c.diag.geometry(path,widths)
                description=('保持当前关节，先张开双夹爪到 100 mm，反馈确认到位后双臂'+('经默认位到任务起手位' if kind=='ready' else '到默认位')+'。速度由底层驱动的现有配置控制，已取消工作台低速插值；会松开持物，自动使能，到位后保持。' if is_reset else '双臂 joint4 增加 0.3°；夹爪不动作。')
                if kind=='ready' and settings.task=='banana':
                    description+='双臂到位停稳后，右爪发送闭合度1（驱动0 mm）指令，左爪保持100 mm；右爪闭合到位或检测到稳定接触力后完成，持物实测开度不必为0。接触不能识别持物，推理前请确认右手实际持蕉。'
                elif task_grip:description+=f'关节到位后恢复任务夹爪宽度：右 {final_widths[0]*1000:g} / 左 {final_widths[1]*1000:g} mm，等待位置反馈确认。'
                body.update(trajectory=serialize(path),geometry=audit,gripper_commands=2*len(path['joints']) if is_reset else 0,
                            description=description+('复位逐段等待真实到位，不设固定耗时；不受推理总时长限制。' if is_reset else f'标称轨迹约 {path["duration_s"]:.1f} 秒；微动保留 3 秒倒计时。')+'请核对整个运动通路。',
                            gripper_policy='task_target_after_joint_arrival' if task_grip else 'open_to_100mm' if is_reset else 'preserve_current_state')
            if not cv2.imwrite(str(directory/'before.png'),cv2.cvtColor(image,cv2.COLOR_RGB2BGR)):raise RuntimeError('保存现场图像失败')
            body['image_sha256']=sha(directory/'before.png')
        finally:io.close()
    body['plan_sha256']=digest(body);write(directory/'plan.json',body)
    operator.set('计划已准备，等待现场确认。')
    return body

def stop_services(disable=False):
    """No publishers; close gates, request current-pose hold, optionally disable."""
    import rclpy
    from std_srvs.srv import SetBool,Empty
    rclpy.init();node=rclpy.create_node('nero_desktop_stop_disable')
    result=dict(motor_disable_requested=disable,responses={},errors=[],hardware_emergency_stop=False)
    grippers={};subscriptions=[]
    if disable:
        from agx_arm_msgs.msg import GripperStatus
        from rclpy.qos import QoSProfile,ReliabilityPolicy
        def receive(msg,arm):
            grippers[arm]=dict(enabled=bool(msg.driver_enable_status),width_m=float(msg.width),
                              source_unix_s=msg.header.stamp.sec+msg.header.stamp.nanosec/1e9,received=time.monotonic())
        for arm in c.pos.ARMS:
            subscriptions.append(node.create_subscription(GripperStatus,f'/{arm}_arm/feedback/gripper_status',
                lambda msg,a=arm:receive(msg,a),QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)))
    try:
        stages=[('control_enable',False),('emergency_stop',None)]+([('enable_agx_arm',False)] if disable else [])
        clients={(a,s):node.create_client(Empty if v is None else SetBool,f'/{a}_arm/{s}') for s,v in stages for a in c.pos.ARMS}
        deadline=time.monotonic()+3
        while not all(x.service_is_ready() for x in clients.values()) and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.01)
        for service,value in stages:
            futures={};requested_unix=time.time()
            for arm in c.pos.ARMS:
                client=clients[arm,service]
                if not client.service_is_ready():result['errors'].append(f'{arm}/{service}: unavailable');continue
                req=Empty.Request() if value is None else SetBool.Request()
                if value is not None:req.data=value
                futures[arm]=client.call_async(req)
            deadline=time.monotonic()+(8 if service=='enable_agx_arm' and value is False else 3)
            while not all(f.done() for f in futures.values()) and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.01)
            for arm,f in futures.items():
                key=f'{arm}/{service}'
                if not f.done() or f.result() is None:result['errors'].append(key+': no response');continue
                response=f.result();result['responses'][key]=str(response)
                if value is not None and not response.success:result['errors'].append(key+': rejected')
        if disable:
            deadline=time.monotonic()+2
            def verified():
                return all(a in grippers and not grippers[a]['enabled'] and
                           requested_unix<=grippers[a]['source_unix_s']<=time.time()+.05 and
                           time.monotonic()-grippers[a]['received']<=.3 for a in c.pos.ARMS)
            while not verified() and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.01)
            result['gripper_feedback']=grippers
            result['gripper_disable_verified']=verified()
            if not result['gripper_disable_verified']:
                result['errors'].append('夹爪失能未由新鲜反馈确认；旧驱动只失能关节，请更新驱动后重试')
        result['control_publishers']={n:node.count_publishers(n) for n,_ in node.get_topic_names_and_types() if n.startswith(('/right_arm/control/','/left_arm/control/'))}
    finally:node.destroy_node();rclpy.shutdown()
    return result

def execute(directory,operator):
    body=read(directory/'plan.json');approval=read(directory/'approval.json');check_approval(body,approval)
    if body.get('simulation_only',False)!=(os.environ.get('ACT_EVAL_FAKE_DRIVER')=='1'):
        raise RuntimeError('模拟计划与真机环境不可互换')
    if body['source_sha256']!=c.identity():raise RuntimeError('源码或数据已改变，请重新准备计划')
    settings=Settings(**body['settings']).checked();c.configure(settings,body['model_identity'],body['server_port'])
    with motion_lock():
        run=directory/'execution';run.mkdir(exist_ok=False);write(run/'approval_consumed.json',approval)
        source=run/'executed_source';source.mkdir()
        # Preserve full paths in manifest to disambiguate duplicate basenames.
        for index,(path,expected) in enumerate(body['source_sha256'].items()):
            shutil.copy2(path,source/f'{index:03d}_{Path(path).name}')
        write(source/'manifest.json',body['source_sha256'])
        result=dict(kind=body['kind'],settings=settings.json(),protocol_sha256=body['protocol_sha256'],
                    plan_sha256=body['plan_sha256'],started=False,runtime_completed=False,started_unix_s=time.time(),
                    simulation_only=body['simulation_only'],adjudication=dict(success=None,reason='等待现场与录像判定'))
        io=rec=None
        try:
            if not operator.permit():raise RuntimeError('界面已停止或心跳丢失')
            if body['kind']=='disable':
                result.update(open_then_disable(run,operator));result.update(started=True,runtime_completed=not result['errors'])
            else:
                io=c.FormalIO(run,operator) if body['kind']=='formal' else ResetIO(run,operator) if body['kind'] in ('ready','default') else PositionIO(run,operator)
                rec=c.VideoRecorder(run,body['image_shape']);io.recorder=rec
                q,w=io.preflight();io.camera();io.monitor_camera=True
                if body.get('driver_parameters') is not None and body['driver_parameters']!=io.driver_parameters:
                    raise RuntimeError('驱动速度或启动配置已改变，请重新准备计划')
                if body['kind']=='micro':io.micro_origin=q.copy();io.micro_max_observed=0.
                if np.max(np.abs(q-np.array(body['start_joints_rad'])))>body['start_tolerance_rad'] or np.max(np.abs(np.array(w)-body['start_widths_m']))>.001:
                    raise RuntimeError('当前位置或夹爪与批准计划不一致，请重新准备')
                is_reset=body['kind'] in ('ready','default')
                deadline=io.now()+(0. if is_reset else 3.)
                while io.now()<deadline:
                    io.pump();io.check_publishers();q,_=io.snapshot(stationary=True)
                    if not operator.permit():raise RuntimeError('倒计时已取消')
                    if np.max(np.abs(q-np.array(body['start_joints_rad'])))>body['start_tolerance_rad']:raise RuntimeError('倒计时期间起点发生变化')
                    operator.set(f'{max(1,math.ceil(deadline-io.now()))} 秒后开始：{body["description"]}');io.idle()
                wait_recording_ready(io,rec,operator,1 if is_reset else 10,body['start_joints_rad'],body['start_tolerance_rad'])
                io.event('execution_ready',fixed_countdown_s=0. if is_reset else 3.,recorded_frames=rec.frames)
                check_approval(body,approval)
                if body['kind']=='formal':
                    if os.environ.get('ACT_EVAL_FAKE_DRIVER')!='1':
                        io.can_observer=c.plate_can_observer.CanObserver(run)
                    result.update(asyncio.run(c.run_attempt(io,dict(body,trial_id=directory.name),operator.permit)))
                else:
                    path={k:np.array(v) if k in ('start','goal','joints','times','gripper_widths_m') else v for k,v in body['trajectory'].items()}
                    if isinstance(io,ResetIO):reset_geometry(path);io.reset_path=path
                    else:c.diag.geometry(path,w)
                    runner=reset.run_motion if isinstance(io,ResetIO) else c.pos.run_motion
                    answer=runner(io,path,lambda:operator.permit() and not io.interrupted,lambda n,total:operator.set(f'{body["kind"]} {n}/{total} ({100*n/total:.0f}%)'))
                    result.update(answer);result['runtime_completed']=answer['ok']
        except Exception as exc:
            result.update(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc())
        finally:
            if io:
                result.update(started=bool(getattr(io,'physical_started',False) or io.command_count),commands_sent=io.command_count,max_tracking_error_rad=io.max_tracking_error)
                if body['kind']=='micro':result['micro_max_observed_deg']=float(np.rad2deg(getattr(io,'micro_max_observed',0.)))
                end=io.now()+(0. if body['kind'] in ('ready','default') else 1.)
                while io.now()<end:io.pump();io.idle()
                try:
                    q,w=io.snapshot(stationary=True);result.update(final_joints_deg=np.rad2deg(q).tolist(),final_widths_m=w)
                    cv2.imwrite(str(directory/'after.png'),cv2.cvtColor(c.diag.decode_compressed_image(io.camera()['data']),cv2.COLOR_RGB2BGR))
                except Exception as exc:result['final_observation_error']=str(exc)
                if getattr(io,'can_observer',None):result['can_evidence']=io.can_observer.close()
                io.monitor_camera=False;io.recorder=None;io.close()
            if rec:
                result['video']=rec.close()
                if result['video']['error']:
                    result.update(recording_error=result['video']['error'],runtime_completed=False)
                    result.setdefault('error','录像记录失败：'+result['video']['error'])
            result['finished_unix_s']=time.time();write(directory/'result.json',result)
            from .records import is_experiment
            operator.last_write=0.
            operator.set('已结束。'+result.get('error','请记录实验结果。' if is_experiment(result) else '操作日志已保存，不计入实验记录。'))
        return result

async def shadow(directory,settings,port,operator):
    c.configure(settings,catalog(settings.task,settings.model),port)
    out=directory/'shadow';out.mkdir();io=c.FormalIO(out,operator);client=c.make_client();rec=None;spinner=None
    result=dict(kind='shadow',settings=settings.json(),physical_motion_executed=False,inferences=0)
    try:
        io.preflight();io.camera();spinner=asyncio.create_task(c.pump_while(io));await client.connect()
        c.validate_metadata(client.metadata,settings.task,settings.model,c.POLICY_SHA)
        if client.metadata.get('encoder_state_sha256')!=c.ENCODER_SHA:raise ValueError('encoder identity mismatch')
        def guard():
            if not operator.permit():raise RuntimeError('只读推理已停止')
            if spinner.done():spinner.result()
            io.check_publishers();io.snapshot()
        first=True;began=time.monotonic();latencies=[];violations=[]
        while time.monotonic()-began<settings.duration_s:
            actions,image,capture=await c.infer(io,client,guard,enforce_limits=False)
            if first:
                q,w=io.snapshot();geometry=c.Geometry(q,w);rec=c.VideoRecorder(out,image.shape);io.recorder=rec;first=False
            qs,ws=c.unpack(actions,joint_limits=False)
            for step in range(c.BOUNDS['steps_per_replan']):
                try:geometry.check(qs[step],ws[step])
                except (ValueError,RuntimeError) as exc:violations.append(dict(inference=result['inferences'],step=step,error=str(exc)))
            io.event('shadow_prediction',actions=actions.tolist(),capture=capture)
            result['inferences']+=1;latencies.append(capture['inference_ms'])
            operator.set(f'只读推理 {time.monotonic()-began:.1f}/{settings.duration_s:g} s · {result["inferences"]} 次 · 越界 {len(violations)}')
            await asyncio.sleep(.1)
        result.update(runtime_completed=True,inference_ms=dict(p50=float(np.percentile(latencies,50)),p95=float(np.percentile(latencies,95)),max=max(latencies)),violations=violations)
    except Exception as exc:result.update(runtime_completed=False,error=str(exc))
    finally:
        if spinner:spinner.cancel();await asyncio.gather(spinner,return_exceptions=True)
        await client.close();io.recorder=None
        if rec:result['video']=rec.close()
        io.close();write(directory/'result.json',result)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','execute','shadow','stop'])
    p.add_argument('--directory',type=Path,required=True);p.add_argument('--kind',choices=['ready','default','formal','disable','micro'])
    p.add_argument('--heartbeat',type=Path);a=p.parse_args();directory=a.directory.resolve()
    if os.environ.get('ACT_EVAL_FAKE_DRIVER')=='1' and (os.environ.get('ROS_DOMAIN_ID')!='174' or os.environ.get('ROS_LOCALHOST_ONLY')!='1'):
        raise RuntimeError('fake driver tests must use isolated ROS domain 174 on localhost')
    try:
        if a.mode=='stop':
            result=stop_services(False);write(directory/'stop_result.json',result);return 2 if result['errors'] else 0
        config=read(directory/'config.json');settings=Settings(**config['settings']).checked();operator=Operator(directory,a.heartbeat)
        if a.mode=='prepare':prepare(directory,a.kind,settings,config['port'],operator);return 0
        if a.mode=='shadow':result=asyncio.run(shadow(directory,settings,config['port'],operator))
        else:result=execute(directory,operator)
        return 2 if result.get('error') or result.get('errors') else 0
    except Exception as exc:
        write(directory/'job_error.json',dict(error=f'{type(exc).__name__}: {exc}',traceback=traceback.format_exc()))
        print(traceback.format_exc(),flush=True);return 2
if __name__=='__main__':raise SystemExit(main())
