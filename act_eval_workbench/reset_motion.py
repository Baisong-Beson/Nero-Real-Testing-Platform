"""Firmware-planned reset segments; spatial auditing is independent of timing."""
import numpy as np
from . import controller as c
from .common import DEFAULT_DEG,limits

DT=.05
MAX_GAP=.35
FINAL_TOL=.001
STALL_TIMEOUT=5.
PROFILE=dict(controller='firmware_move_j',joint_velocity_rad_s=None,
    joint_acceleration_rad_s2=None,tracking_target_lead_rad=None,
    gripper_velocity_m_s=None,gripper_acceleration_m_s2=None,
    measured_speed_abort_rad_s=None,feedback_poll_s=.005,max_command_gap_s=MAX_GAP,
    stall_timeout_s=STALL_TIMEOUT,final_tolerance_rad=FINAL_TOL)

def segment(start,goal):
    count=max(1,int(np.ceil(np.max(np.abs(goal-start))/.01)))
    return start+np.linspace(0,1,count+1)[:,None,None]*(goal-start)

def trajectory(start,goal,via_default=False):
    start=limits(start);goal=limits(goal)
    qs=segment(start,np.deg2rad(DEFAULT_DEG)) if via_default else segment(start,goal)
    ends=[len(qs)-1]
    if via_default:
        qs=np.concatenate([qs,segment(np.deg2rad(DEFAULT_DEG),goal)[1:]])
        ends.append(len(qs)-1)
    limits(qs)
    return dict(stage='desktop_reset_driver_v6',start=start,goal=goal,joints=qs,
        segment_end_indices=ends,times=None,duration_s=None,
        samples_are_spatial_only=True,max_velocity_rad_s=None,
        max_acceleration_rad_s2=None,reset_profile=PROFILE.copy())

def gripper_segment(widths,goal):
    # Dense geometry audit only; execution requests final width once.
    widths=np.asarray(widths);goal=np.asarray(goal)
    count=int(np.ceil(np.max(np.abs(goal-widths))/.005))
    if count==0:return np.asarray([widths])
    return widths+np.linspace(0,1,count+1)[:,None]*(goal-widths)

def opening(widths):return gripper_segment(widths,np.full(2,.1))

def run_motion(io,plan,permit,progress):
    touched=success=False;errors=[];last_poll=None;max_gap=0.;delays=0;sent=0
    def guard():
        nonlocal last_poll,max_gap,delays
        io.pump()
        if io.interrupted or not permit():raise RuntimeError('operator stopped / window closed / interrupted')
        io.check_publishers(owned=touched)
        now=io.now()
        if last_poll is not None:
            gap=now-last_poll;max_gap=max(max_gap,gap)
            if gap>MAX_GAP:raise RuntimeError(f'复位监控中断 {gap:.4f}s，已停止')
            if gap>.15:
                delays+=1;io.event('reset_scheduler_delay',gap_s=gap,waypoints_skipped=0)
        last_poll=now
        return io.snapshot()
    def send(q,index):
        nonlocal sent
        actual,_=guard();limits(q);io.publish(q,actual,index);sent+=1
        progress(index+1,len(plan['joints']))
    try:
        actual,_=guard()
        if np.max(np.abs(actual-plan['start']))>.01:raise RuntimeError('start pose changed since preview')
        touched=True;io.prepare_publishers()
        for service,value in [('control_enable',False),('enable_agx_arm',True),('control_enable',True)]:
            last_poll=None;guard();io.call_pair(service,value,permit)
        last_poll=None;actual,widths=guard()
        if np.max(np.abs(actual-plan['start']))>.01:raise RuntimeError('pose moved during enabling; abort')
        end=plan.get('opening_last_index',0)
        send(plan['start'],0)
        if end:send(plan['start'],end)
        marks=np.asarray(widths).copy();advanced=np.full(2,io.now())
        while True:
            actual,widths=guard();widths=np.asarray(widths)
            if np.max(np.abs(actual-plan['start']))>np.deg2rad(.5):
                raise RuntimeError('开爪期间关节偏移超过 0.5°，停止')
            pending=np.abs(widths-.1)>.002
            if not pending.any():break
            moved=np.abs(widths-marks)>=.0001
            advanced[moved|~pending]=io.now();marks[moved]=widths[moved]
            if np.any(pending & (io.now()-advanced>STALL_TIMEOUT)):
                raise RuntimeError('夹爪张开无进展，保持关节并停止复位')
            io.idle()
        io.event('reset_grippers_open_verified',widths_m=widths.tolist())
        start_index=end
        for end_index in plan.get('segment_end_indices',[]):
            goal=plan['joints'][end_index];actual,_=guard()
            marks=actual.copy();advanced=np.full((2,7),io.now())
            send(goal,end_index)
            while True:
                actual,_=guard();pending=np.abs(actual-goal)>FINAL_TOL
                if not pending.any():
                    try:
                        settled=io.snapshot(stationary=True)[0]
                        if np.max(np.abs(settled-goal))<=FINAL_TOL:break
                    except RuntimeError as exc:
                        if 'stationary' not in str(exc):raise
                moved=np.abs(actual-marks)>=.0005
                advanced[moved|~pending]=io.now();marks[moved]=actual[moved]
                if np.any(pending & (io.now()-advanced>STALL_TIMEOUT)):
                    raise RuntimeError(f'reset feedback made no progress for {STALL_TIMEOUT:g} s; holding')
                io.idle()
            io.event('reset_segment_completed',start_index=start_index,end_index=end_index,
                controller='firmware_move_j',commands=1,feedback_verified=True)
            start_index=end_index
        final_index=plan.get('final_gripper_index');contact_evidence=None;close_started=None
        if final_index is not None:
            actual,widths=guard()
            if np.max(np.abs(actual-plan['goal']))>FINAL_TOL:
                raise RuntimeError('双臂尚未到达起手位，停止恢复任务夹爪状态')
            marks=np.asarray(widths).copy();advanced=np.full(2,io.now())
            io.event('reset_task_grippers_started',target_widths_m=plan['gripper_goal_m'])
            close_started=io.now()
            send(plan['goal'],final_index)
            while True:
                actual,widths=guard();widths=np.asarray(widths)
                if np.max(np.abs(actual-plan['goal']))>np.deg2rad(.5):
                    raise RuntimeError('恢复任务夹爪状态期间关节偏移超过 0.5°，停止')
                pending=np.abs(widths-np.asarray(plan['gripper_goal_m']))>.002
                if plan.get('gripper_completion')=='right_closed_or_contact' and pending[0]:
                    contact_evidence=io.gripper_contact(close_started,widths)
                    if contact_evidence:pending[0]=False
                if not pending.any():break
                moved=np.abs(widths-marks)>=.0001
                advanced[moved|~pending]=io.now();marks[moved]=widths[moved]
                if np.any(pending & (io.now()-advanced>STALL_TIMEOUT)):
                    raise RuntimeError('任务夹爪收拢无进展且未确认稳定接触，起手复位未完成；请检查夹爪反馈和持物')
                io.idle()
            io.event('reset_task_grippers_verified',target_widths_m=plan['gripper_goal_m'],widths_m=widths.tolist(),
                completion=contact_evidence or dict(mode='width_reached'),object_presence_verified=False)
        actual,widths=guard()
        if np.max(np.abs(actual-plan['goal']))>FINAL_TOL:raise RuntimeError('最终关节位置偏离目标，复位未完成')
        pending=np.abs(np.asarray(widths)-np.asarray(plan['gripper_goal_m']))>.002
        if contact_evidence:
            contact_evidence=io.gripper_contact(close_started,widths)
            if contact_evidence:pending[0]=False
        if pending.any():
            raise RuntimeError('最终夹爪反馈未到目标开度，复位未完成')
        success=True
        return dict(ok=True,final_joints_deg=np.rad2deg(actual).tolist(),
            final_widths_m=list(widths),gripper_goal_m=plan['gripper_goal_m'],
            gripper_completion=contact_evidence or dict(mode='width_reached'),object_presence_verified=False,
            max_error_rad=float(np.max(np.abs(actual-plan['goal']))),
            max_command_gap_s=max_gap,scheduler_delays=delays,reset_profile=PROFILE.copy(),
            feedback_driven=True,commands_sent=sent)
    finally:
        if touched:
            for service,value in [('control_enable',False)]+([] if success else [('emergency_stop',None)]):
                try:io.call_pair(service,value,None)
                except Exception as exc:errors.append(f'{service}: {exc}')
            io.event('cleanup',reached_target=success,errors=errors,motors_left_enabled=True,
                software_stop_is_hardware_estop=False,max_command_gap_s=max_gap,scheduler_delays=delays)
            if errors:raise RuntimeError('CLEANUP FAILED; use hardware emergency stop: '+str(errors))
