#!/usr/bin/env python3
"""One approved, recorded, bounded closed-loop model attempt.

prepare is read-only. execute consumes one approval, never retries or resets.
Controller completion is not task success; video/onsite adjudication is required.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import shutil
import time

import cv2
import numpy as np

import sys
from .common import LEGACY, LOWER, UPPER, source_identity
from .recording import VideoRecorder
from .driver_control import DriverTargets, feedback_snapshot, driver_preflight
sys.path.insert(0,str(LEGACY))
import plate_startpose as pos
import plate_approved_step as diag
import plate_can_observer
from .policy_protocol import validate_metadata
from . import library
from nero_pi05_bridge.observation import decode_compressed_image
from nero_pi05_bridge.policy_client import PolicyClient

PROTOCOL = 'nero_desktop.driver_targets.v7'
TASK = 'plate'
MODEL = 'mpi-base'
POLICY_SHA = ''
ENCODER_SHA = ''
SERVER_PORT = 8026
DT = .05
HORIZON = 16
MODEL_IDENTITY = {}
CAMERAS = ['main']
BOUNDS = dict(duration_s=180., steps_per_replan=4, max_command_gap_s=.15,
              relative_limits_enabled=False,
              scheduler_gap_max_s=.35,
              joint_velocity_rad_s=.15, joint_acceleration_rad_s2=.30,
              joint_command_lead_rad=.03, joint_excursion_rad=float(np.deg2rad(45)),
              gripper_velocity_m_s=.03, gripper_acceleration_m_s2=.08,
              gripper_effort_n=1., tcp_displacement_m=.20, frame_displacement_m=.25,
              tcp_step_m=.008, source_age_at_inference_s=.20,
              action_max_age_s=.50, image_max_age_s=.50,
              inference_timeout_s=.15, start_tolerance_rad=float(np.deg2rad(.2)),
              startup_hold_s=1., tracking_target_lead_rad=.02, tracking_stall_s=2.)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')


def configure(settings, metadata, port):
    global TASK, MODEL, POLICY_SHA, ENCODER_SHA, SERVER_PORT, DT, HORIZON, MODEL_IDENTITY, CAMERAS
    settings.checked()
    TASK, MODEL = settings.task, settings.model
    POLICY_SHA, ENCODER_SHA = metadata['policy_sha256'], metadata['encoder_state_sha256']
    SERVER_PORT = int(port)
    MODEL_IDENTITY=metadata
    cfg=settings.protocol()['model_runtime'];DT=cfg['dt_s'];HORIZON=cfg['action_horizon'];CAMERAS=cfg['cameras']
    timeout=cfg['inference_timeout_s'];custom=bool(metadata.get('adapter'))
    BOUNDS.update(steps_per_replan=cfg['steps_per_replan'],inference_timeout_s=timeout,
        source_age_at_inference_s=timeout+.25 if custom else .20,
        action_max_age_s=timeout+cfg['steps_per_replan']*DT+.3 if custom else .5,
        image_max_age_s=timeout+cfg['steps_per_replan']*DT+.3 if custom else .5,
        max_command_gap_s=max(.15,3*DT),scheduler_gap_max_s=max(.35,4*DT))
    if not 1024 <= SERVER_PORT <= 65535: raise ValueError('invalid server port')
    BOUNDS.update(duration_s=settings.duration_s,relative_limits_enabled=settings.relative_limits_enabled,
                  joint_excursion_rad=float(np.deg2rad(settings.excursion_deg)),
                  tcp_displacement_m=settings.tcp_cm/100, frame_displacement_m=settings.tcp_cm/100+.05)


def make_client():
    if not POLICY_SHA: raise RuntimeError('controller not configured')
    return PolicyClient('127.0.0.1',SERVER_PORT,inference_timeout_sec=BOUNDS['inference_timeout_s'])


def identity():
    return dict(source_identity(), **{str(pos.URDF):hashlib.sha256(pos.URDF.read_bytes()).hexdigest()})


class PolicyJointLimitError(ValueError):
    def __init__(self, violations):
        self.violations=violations
        row=violations[0]
        super().__init__('joint soft limit exceeded: 模型预测第'
            f'{row["step"]}步，{row["arm_label"]} joint{row["joint"]} '
            f'{row["value_deg"]:.3f}°，允许范围 '
            f'[{row["lower_deg"]:.3f}, {row["upper_deg"]:.3f}]°')


def joint_limit_violations(q):
    rows=[]
    for step,arm,joint in np.argwhere((q<LOWER)|(q>UPPER)):
        value=float(q[step,arm,joint])
        rows.append(dict(step=int(step+1),arm=pos.ARMS[arm],arm_label=('右臂','左臂')[arm],
            joint=int(joint+1),value_rad=value,value_deg=float(np.rad2deg(value)),
            lower_deg=float(np.rad2deg(LOWER[joint])),upper_deg=float(np.rad2deg(UPPER[joint]))))
    return rows


def unpack(actions, joint_limits=True, limit_steps=None):
    a = np.asarray(actions, dtype=float)
    if a.shape != (HORIZON, 16) or not np.isfinite(a).all():
        raise ValueError(f'模型应返回有限的绝对动作 [{HORIZON},16]')
    q = np.stack((a[:, :7], a[:, 8:15]), axis=1)
    if limit_steps is not None and (isinstance(limit_steps,bool) or not isinstance(limit_steps,int) or not 1<=limit_steps<=HORIZON):
        raise ValueError(f'limit_steps must be an integer in [1,{HORIZON}]')
    if joint_limits:
        violations=joint_limit_violations(q if limit_steps is None else q[:limit_steps])
        if violations:raise PolicyJointLimitError(violations)
    w = .1*(1-np.clip(a[:, [7,15]], 0, 1))
    return q, w


def braking_speed_limit(distance, acceleration, vmax, elapsed):
    """Largest speed that can reach this boundary and stop without crossing it.

    Include this dispatch interval and subsequent nominal-period braking steps.
    Unlike proportional easing, this does not create an exponential slow tail.
    """
    distance = np.maximum(0., np.asarray(distance, float))
    low = np.zeros_like(distance)
    high = np.broadcast_to(vmax, distance.shape).copy()
    decrement = acceleration*DT
    for _ in range(40):
        speed = (low+high)*.5
        count = np.maximum(0., np.ceil(speed/decrement)-1)
        stopping = speed*elapsed+DT*(count*speed-decrement*count*(count+1)*.5)
        feasible = stopping <= distance
        low = np.where(feasible, speed, low)
        high = np.where(feasible, high, speed)
    return low


class Slew:
    """Persistent velocity history across replans, joints rad / grippers metres."""
    def __init__(self, q, widths):
        self.x = np.r_[np.asarray(q).reshape(-1), np.clip(widths, 0, .1)]
        self.v = np.zeros(16)
        self.vmax = np.r_[np.full(14, BOUNDS['joint_velocity_rad_s']), np.full(2, BOUNDS['gripper_velocity_m_s'])]
        self.amax = np.r_[np.full(14, BOUNDS['joint_acceleration_rad_s2']), np.full(2, BOUNDS['gripper_acceleration_m_s2'])]
        self.lower = np.r_[np.tile(LOWER, 2), [0., 0.]]
        self.upper = np.r_[np.tile(UPPER, 2), [.1, .1]]

    def next(self, target_q, target_w, elapsed, actual_q=None):
        if not DT-1e-6 <= elapsed <= BOUNDS['max_command_gap_s']:
            raise RuntimeError('control timing outside [0.05,0.15] s')
        if actual_q is not None:
            # Advance toward the live measured pose, rather than integrating
            # a growing command error while the hardware is still starting.
            actual_q=pos.limits(actual_q)
            lead=BOUNDS['tracking_target_lead_rad']
            target_q=np.clip(target_q,actual_q-lead,actual_q+lead)
        target = np.r_[np.asarray(target_q).reshape(-1), target_w]
        if target.shape != (16,) or not np.isfinite(target).all():
            raise ValueError('invalid target')
        err = target-self.x
        desired = np.sign(err)*braking_speed_limit(np.abs(err), self.amax, self.vmax, elapsed)
        # Actual intervals preserve speed/acceleration bounds across replans.
        # Each interval executes exactly one waypoint; none are skipped.
        # A replan may reverse its target before the previous velocity has
        # stopped. Keep enough braking distance to absolute hardware bounds.
        lo = np.maximum(-braking_speed_limit(self.x-self.lower, self.amax, self.vmax, elapsed),
                        self.v-self.amax*elapsed)
        hi = np.minimum(braking_speed_limit(self.upper-self.x, self.amax, self.vmax, elapsed),
                        self.v+self.amax*elapsed)
        if np.any(lo > hi+1e-10):
            raise RuntimeError('timing jitter cannot satisfy speed/acceleration bounds')
        lo = np.minimum(lo, hi)  # Numerical contact at a braking boundary.
        velocity = np.clip(desired, lo, hi)
        candidate = self.x+velocity*elapsed
        q, widths = candidate[:14].reshape(2,7), candidate[14:]
        pos.limits(q)
        if np.any(widths < 0) or np.any(widths > .1):
            raise RuntimeError('gripper command outside [0,0.1] m')
        self.x, self.v = candidate, velocity
        return q.copy(), widths.copy()


def advance_after_delay(slew,target_q,target_w,actual_q,actual_w,elapsed):
    """A brief scheduling gap restarts the filter from fresh feedback at rest."""
    if not DT-1e-6<=elapsed<=BOUNDS['scheduler_gap_max_s']:
        raise RuntimeError(f'control interval {elapsed:.4f}s outside [{DT:g},{BOUNDS["scheduler_gap_max_s"]:g}] s')
    rebased=elapsed>BOUNDS['max_command_gap_s']
    if isinstance(slew,DriverTargets):
        if rebased:
            # Hold live feedback after a missed interval; replan next, never
            # catch up commands from an old chunk.
            slew=DriverTargets(actual_q,actual_w)
            return slew,np.array(actual_q),np.array(actual_w),True
        q,w=slew.next(target_q,target_w,elapsed,actual_q)
        return slew,q,w,False
    if rebased:
        slew=Slew(actual_q,actual_w)
        elapsed=DT  # Never integrate the missed wall-clock time as a large step.
    q,w=slew.next(target_q,target_w,elapsed,actual_q)
    return slew,q,w,rebased

class Geometry:
    def __init__(self, start, widths):
        from nero_pi05_bridge.nero_fk import NeroFK
        if hashlib.sha256(pos.URDF.read_bytes()).hexdigest() != pos.URDF_SHA:
            raise RuntimeError('URDF changed')
        self.fk = NeroFK(pos.URDF)
        box = json.loads((pos.TOOLS/'plate_shadow_audit_v2.json').read_text())['workspace_aabb']
        self.lo = np.array([box[k+'_min'] for k in 'xyz'])
        self.hi = np.array([box[k+'_max'] for k in 'xyz'])
        self.start_q = np.array(start)
        self.origin, self.frames = self.positions(start, widths)
        self.previous = self.origin.copy()

    def positions(self, q, widths):
        tcp, frames = [], []
        for i in range(2):
            tcp.append(self.fk.tcp_xyz(q[i], widths[i]).copy())
            frames.append(np.array([x.translation.copy() for x in self.fk.data.oMf]))
        return np.array(tcp), np.array(frames)

    def check(self, q, widths, commit=False, check_step=True):
        pos.limits(q)
        if BOUNDS['relative_limits_enabled'] and np.max(np.abs(q-self.start_q)) > BOUNDS['joint_excursion_rad']:
            
            delta=np.rad2deg(q-self.start_q);i,j=np.unravel_index(np.abs(delta).argmax(),delta.shape)
            raise RuntimeError(f'{pos.ARMS[i]} joint{j+1} target offset {delta[i,j]:.3f} deg exceeds {np.rad2deg(BOUNDS["joint_excursion_rad"]):g} deg')
        tcp, frames = self.positions(q, widths)
        if not np.isfinite(tcp).all() or np.any(tcp < self.lo) or np.any(tcp > self.hi):
            raise RuntimeError('TCP outside workspace AABB')
        displacement = float(np.linalg.norm(tcp-self.origin, axis=-1).max())
        if BOUNDS['relative_limits_enabled'] and displacement > BOUNDS['tcp_displacement_m']:
            raise RuntimeError(f'TCP displacement {displacement*100:.2f} cm exceeds {BOUNDS["tcp_displacement_m"]*100:g} cm')
        if BOUNDS['relative_limits_enabled'] and np.linalg.norm(frames-self.frames, axis=-1).max() > BOUNDS['frame_displacement_m']:
            distance=float(np.linalg.norm(frames-self.frames, axis=-1).max())
            raise RuntimeError(f'link-frame displacement {distance*100:.2f} cm exceeds {BOUNDS["frame_displacement_m"]*100:g} cm')
        if commit:
            if check_step and np.linalg.norm(tcp-self.previous, axis=-1).max() > BOUNDS['tcp_step_m']:
                raise RuntimeError('TCP command step exceeds 8 mm')
            self.previous = tcp.copy()
        return displacement

    def check_path(self,start_q,start_w,goal_q,goal_w):
        # Spatial path samples, not timed control points or speed restrictions.
        n=max(1,int(np.ceil(np.max(np.abs(goal_q-start_q))/.02)),
              int(np.ceil(np.max(np.abs(np.asarray(goal_w)-start_w))/.005)))
        for u in np.linspace(0,1,n+1):
            self.check(start_q+u*(goal_q-start_q),np.asarray(start_w)+u*(np.asarray(goal_w)-start_w))


class FormalIO(diag.CameraIO):
    def call_pair(self,service,value,permit=None):
        result=super().call_pair(service,value,permit)
        if service=='enable_agx_arm' and value:
            super().call_pair('emergency_stop',None,permit)
        return result

    def __init__(self, output, gui=None):
        super().__init__(output, gui)
        self.output=Path(output)
        self.feedback = pos.Feedback(require_open=False)
        self.physical_started = False
        self.gripper_pubs = {}
        self.can_observer = None
        self.wrist_frames={}
        if any(k!='main' for k in CAMERAS):
            from sensor_msgs.msg import Image
            from rclpy.qos import QoSProfile,ReliabilityPolicy
            from .camera_health import WRIST_TOPICS
            def receive(msg,key):self.wrist_frames[key]=(msg,self.now())
            for key in CAMERAS:
                if key=='main':continue
                self.subs.append(self.node.create_subscription(Image,WRIST_TOPICS[key.split('_')[0]],lambda msg,k=key:receive(msg,k),QoSProfile(depth=1,reliability=ReliabilityPolicy.BEST_EFFORT)))

    def extra_images(self,main_stamp):
        from .model_images import wrist_images
        return wrist_images(CAMERAS,self.wrist_frames,self.now(),self.node.get_clock().now().nanoseconds/1e9,main_stamp)

    def snapshot(self,stationary=False):
        return feedback_snapshot(self.feedback,self.now(),self.node.get_clock().now().nanoseconds/1e9,stationary)

    def preflight(self):
        return driver_preflight(self)

    def check_publishers(self, owned=False):
        topics = {n for n,_ in self.node.get_topic_names_and_types()
                  if n.startswith(('/right_arm/control/', '/left_arm/control/'))}
        ours = ([f'/{a}_arm/control/move_j' for a in self.pubs]+
                [f'/{a}_arm/control/joint_states' for a in self.gripper_pubs])
        topics.update(f'/{a}_arm/control/joint_states' for a in pos.ARMS)
        for topic in topics:
            if self.node.count_publishers(topic) != int(owned and topic in ours):
                raise RuntimeError('unexpected control publisher: '+topic)
        for arm in pos.ARMS:
            if self.node.count_publishers(f'/{arm}_arm/feedback/joint_states') != 1:
                raise RuntimeError('feedback publisher not unique: '+arm)
        if self.recorder and self.recorder.error:
            raise RuntimeError('recording failed: '+self.recorder.error)
        if self.monitor_camera:
            self.camera()
        if owned and self.can_observer:
            self.can_observer.check()

    def prepare_publishers(self):
        for arm in pos.ARMS:
            # Reuse the joint-only route that actually moved the arms in D001.
            self.pubs[arm] = self.node.create_publisher(self.JointState, f'/{arm}_arm/control/move_j', 1)
            self.gripper_pubs[arm] = self.node.create_publisher(self.JointState, f'/{arm}_arm/control/joint_states', 1)
            for service in ('control_enable', 'enable_agx_arm', 'emergency_stop'):
                cls = self.Empty if service == 'emergency_stop' else self.SetBool
                self.clients[arm,service] = self.node.create_client(cls, f'/{arm}_arm/{service}')
        deadline = self.now()+3.
        while any(self.node.count_subscribers(f'/{a}_arm/control/{topic}') != 1
                  for a in pos.ARMS for topic in ('move_j','joint_states')) or any(
                not self.clients[a,s].service_is_ready() for a in pos.ARMS
                for s in ('control_enable', 'enable_agx_arm', 'emergency_stop')):
            self.pump()
            if self.interrupted or (self.gui and (self.gui.stopped or self.gui.closed)) or self.now() > deadline:
                raise RuntimeError('formal control interface unavailable/cancelled')
            self.idle()

    def publish_full(self, q, widths, actual, measured_widths, index, chunk, step, preserve_gripper=False):
        # move_j destinations are not servo deltas. Monitor progress separately
        # instead of rejecting valid distant targets as a tracking error.
        pos.limits(q)
        widths=np.asarray(widths,float)
        if widths.shape!=(2,) or not np.isfinite(widths).all() or np.any((widths<0)|(widths>.1)):
            raise ValueError('gripper command outside [0,0.1] m')
        # Log dispatch intent before the first irreversible publish. A partial
        # left/right publish still counts as a started attempt.
        self.event('command_dispatch', index=index, chunk=chunk, step=step,
                   joints_rad=q.tolist(), gripper_width_m=widths.tolist(),
                   gripper_command_sent=not preserve_gripper,
                   feedback_rad=actual.tolist(), feedback_gripper_width_m=list(measured_widths),
                   joint_topic='control/move_j',gripper_topic='control/joint_states (gripper only)',
                   feedback_source_unix_s=[self.feedback.rows[a][-1]['stamp'] for a in pos.ARMS])
        self.physical_started = True
        for i, arm in enumerate(pos.ARMS):
            msg = self.JointState()
            msg.header.stamp = self.node.get_clock().now().to_msg()
            msg.name = pos.NAMES.copy()
            msg.position = q[i].tolist()
            self.pubs[arm].publish(msg)
            if not preserve_gripper:
                grip = self.JointState()
                grip.header.stamp = msg.header.stamp
                grip.name = ['gripper'];grip.position = [float(widths[i])]
                grip.effort = [BOUNDS['gripper_effort_n']]
                self.gripper_pubs[arm].publish(grip)
        self.command_count += 1
        self.max_tracking_error = max(self.max_tracking_error, float(np.max(np.abs(q-actual))))
        self.event('command_sent', index=index)


async def infer(io, client, guard, stationary=False, enforce_limits=True):
    guard()
    q, widths = io.snapshot(stationary=stationary)
    row = io.camera()
    stamps = [io.feedback.rows[a][-1]['stamp'] for a in pos.ARMS]
    if max([*stamps,row['stamp']])-min([*stamps,row['stamp']]) > .25:
        raise RuntimeError('image/arms source skew >0.25 s')
    # JPEG decoding and websocket serialization occur while ROS still pumps.
    rgb = await asyncio.to_thread(decode_compressed_image, row['data'])
    state = np.r_[q[0], np.clip(1-widths[0]/.1,0,1), q[1], np.clip(1-widths[1]/.1,0,1)].astype(np.float32)
    sent = time.monotonic()
    extra=io.extra_images(row['stamp']) if hasattr(io,'extra_images') else {}
    observation=library.observation(MODEL_IDENTITY or dict(task=TASK),rgb,state,extra)
    pending=asyncio.create_task(client.infer(observation))
    try:
        deadline=time.monotonic()+BOUNDS['inference_timeout_s']
        while not pending.done():
            guard()
            if getattr(io,'policy_deadline',None) is not None and io.now()>=io.policy_deadline:raise AttemptDurationReached()
            if time.monotonic()>=deadline:raise TimeoutError('TimeoutError: 模型推理超过配置超时')
            await asyncio.sleep(.005)
        response=pending.result()
    finally:
        if not pending.done():pending.cancel()
        await asyncio.gather(pending,return_exceptions=True)
    guard()
    now = io.node.get_clock().now().nanoseconds/1e9
    if max(now-s for s in stamps) > BOUNDS['source_age_at_inference_s'] or now-row['stamp'] > BOUNDS['image_max_age_s']:
        raise RuntimeError('observation expired during inference')
    actions = np.asarray(response['actions'])
    capture=dict(state=state.tolist(), arm_source_unix_s=stamps,
        image_source_unix_s=row['stamp'], returned_monotonic_s=time.monotonic(),
        inference_ms=(time.monotonic()-sent)*1000, server_timing=response.get('server_timing'),prompt=observation['prompt'],cameras=CAMERAS)
    execution_steps=BOUNDS['steps_per_replan']
    try:
        qs,_=unpack(actions,joint_limits=enforce_limits,limit_steps=execution_steps)
    except PolicyJointLimitError as exc:
        # The rejected chunk used to vanish before policy_chunk was logged.
        # Preserve its exact JPEG input and float32 state for reproducible diagnosis.
        io.event('policy_joint_limit_rejected',error=str(exc),violations=exc.violations,
                 execution_steps=execution_steps,actions=actions.tolist(),capture=capture)
        if getattr(io,'output',None) is not None:
            try:
                (io.output/'rejected_policy_input.jpg').write_bytes(row['data'])
                write_json(io.output/'rejected_policy_input.json',dict(capture=capture,actions=actions.tolist(),
                    violations=exc.violations,execution_steps=execution_steps,
                    image_file='rejected_policy_input.jpg',image_sha256=hashlib.sha256(row['data']).hexdigest()))
            except OSError as save_error:
                io.event('rejected_input_save_error',error=str(save_error))
        raise
    if enforce_limits:
        future=[v for v in joint_limit_violations(qs) if v['step']>execution_steps]
        if future:
            io.event('policy_future_limit_warning',execution_steps=execution_steps,violations=future,
                     disposition='unused forecast only; replan before these steps; never dispatch them',capture=capture)
    return actions,rgb,capture


async def pump_while(io):
    while True:
        io.pump()
        await asyncio.sleep(.002)

class AttemptDurationReached(Exception):pass


async def connected_prediction(io):
    spinner = asyncio.create_task(pump_while(io))
    client = make_client()
    try:
        await client.connect()
        validate_metadata(client.metadata, TASK, MODEL, POLICY_SHA)
        if client.metadata.get('encoder_state_sha256') != ENCODER_SHA:
            raise ValueError('selected encoder identity mismatch')
        def guard():
            if spinner.done():
                spinner.result()
            if io.interrupted:
                raise RuntimeError('interrupted')
            io.check_publishers()
            io.snapshot(stationary=True)
        actions, image, capture = await infer(io,client,guard,True)
        return actions, image, dict(capture, metadata=client.metadata)
    finally:
        spinner.cancel()
        await asyncio.gather(spinner,return_exceptions=True)
        await client.close()


async def run_attempt(io, body, permit, client=None):
    """Shared runtime, tested against isolated fake drivers before hardware use."""
    own_client = client is None
    client = client or make_client()
    spinner = asyncio.create_task(pump_while(io))
    touched = False
    began = None
    chunks = 0
    delay_rebases = 0
    max_control_interval = 0.
    reason = 'duration_reached_pending_adjudication'
    def guard():
        if spinner.done():
            spinner.result()
        io.pump()
        if not permit() or io.interrupted:
            raise RuntimeError('operator stop/window loss/interruption')
        io.check_publishers(owned=touched)
        actual, widths = io.snapshot()
        return actual, widths
    try:
        await client.connect()
        validate_metadata(client.metadata,TASK,MODEL,POLICY_SHA)
        if client.metadata.get('encoder_state_sha256') != ENCODER_SHA:
            raise ValueError('selected encoder identity mismatch')
        initial, widths = guard()
        if np.max(np.abs(initial-np.asarray(body['start_joints_rad']))) > BOUNDS['start_tolerance_rad'] or np.max(np.abs(np.array(widths)-body['start_widths_m'])) > .001:
            raise RuntimeError('start differs from approved preview')
        geometry = Geometry(initial,widths)
        slew = DriverTargets(initial,widths)
        # Prove the live server response before enabling any control interface.
        await infer(io,client,guard,True)
        touched = True
        io.prepare_publishers()
        for service,value in [('control_enable',False),('enable_agx_arm',True),('control_enable',True)]:
            guard()
            io.call_pair(service,value,permit)
        actual,widths = guard()
        if np.max(np.abs(actual-initial)) > BOUNDS['start_tolerance_rad']:
            raise RuntimeError('arms moved while enabling')
        began = io.now()
        io.policy_deadline=began+BOUNDS['duration_s']
        last = began
        # Establish control at the exact current joint pose. This is part of the
        # same approved timed attempt, never a separate diagnostic motion.
        while io.now()-began < BOUNDS['startup_hold_s']:
            while io.now()-last < DT:
                guard();await asyncio.sleep(.002)
            actual,measured_widths=guard()
            if np.max(np.abs(actual-initial)) > BOUNDS['start_tolerance_rad']:
                raise RuntimeError('pose changed during initial hold')
            stamp=io.now()
            if stamp-last>BOUNDS['scheduler_gap_max_s']:raise RuntimeError(f'startup hold control interval {stamp-last:.4f}s exceeds 0.35 s')
            io.publish_full(initial,np.clip(widths,0,.1),actual,measured_widths,
                            io.command_count,-1,io.command_count,preserve_gripper=True)
            last=stamp
        progress_q=actual.copy();last_progress=np.full((2,7),io.now())
        while io.now()-began < BOUNDS['duration_s']:
            try:actions, _, capture = await infer(io,client,guard)
            except AttemptDurationReached:break
            if MODEL_IDENTITY.get('adapter'):
                # Inference latency is budgeted separately from a stalled dispatch
                # loop. Slow VLA calls must not rebase away every fresh chunk.
                last=io.now()-DT
            qs, ws = unpack(actions,limit_steps=BOUNDS['steps_per_replan'])
            io.event('policy_chunk', chunk=chunks, actions=actions.tolist(), capture=capture)
            for step in range(BOUNDS['steps_per_replan']):
                while io.now()-last < DT:
                    guard()
                    await asyncio.sleep(.002)
                if io.now()-began >= BOUNDS['duration_s']:
                    break
                actual,widths = guard()
                progressed=np.abs(actual-progress_q)>.002
                tracking=np.abs(slew.x[:14].reshape(2,7)-actual)<.008
                last_progress[progressed|tracking]=io.now()
                progress_q[progressed]=actual[progressed]
                if np.any(io.now()-last_progress > BOUNDS['tracking_stall_s']):
                    raise RuntimeError('hardware joint tracking stalled for >2 s')
                now = io.node.get_clock().now().nanoseconds/1e9
                if max(now-s for s in capture['arm_source_unix_s']) > BOUNDS['action_max_age_s'] or now-capture['image_source_unix_s'] > BOUNDS['image_max_age_s']:
                    raise RuntimeError('policy action observation expired')
                geometry.check(actual,widths)
                # Reject any unbounded raw waypoint as well as the next command.
                geometry.check(qs[step],ws[step])
                tick = io.now()
                interval=tick-last;max_control_interval=max(max_control_interval,interval)
                slew,q,w,rebased=advance_after_delay(slew,qs[step],ws[step],actual,widths,interval)
                if rebased:
                    delay_rebases+=1
                    io.event('policy_scheduler_delay',elapsed_s=interval,chunk=chunks,step=step,
                             recovery='hold fresh measured joints, preserve existing gripper command, discard remaining chunk; replan next')
                geometry.check_path(actual,np.asarray(widths),q,w)
                geometry.check(q,w,commit=True,check_step=False)
                io.publish_full(q,w,actual,widths,io.command_count,chunks,step,preserve_gripper=rebased)
                last = tick
                if rebased:break
                if io.gui:
                    io.gui.text.set(f'{body["trial_id"]}: {io.now()-began:.1f}/{BOUNDS["duration_s"]:g} s\n模型与夹爪控制中。STOP / Esc 停止。\n指令数：{io.command_count}；成功与否请根据录像判定。')
            chunks += 1
        return dict(runtime_completed=True, termination=reason, chunks=chunks,
                    active_duration_s=io.now()-began, plate_lift_success=None,
                    scheduler_delay_rebases=delay_rebases,max_control_interval_s=max_control_interval)
    finally:
        io.policy_deadline=None
        errors = []
        if touched:
            # Even normal time expiry must stop an unfinished SDK target.
            for service,value in [('control_enable',False),('emergency_stop',None)]:
                try:
                    io.call_pair(service,value,None)
                except Exception as exc:
                    errors.append(f'{service}: {exc}')
            io.event('cleanup',errors=errors,motors_left_enabled=True,
                     scheduler_delay_rebases=delay_rebases,max_control_interval_s=max_control_interval,
                     gripper_target_unchanged=True,software_stop_is_hardware_estop=False)
        spinner.cancel()
        await asyncio.gather(spinner,return_exceptions=True)
        if own_client:
            try:
                await client.close()
            except Exception as exc:
                io.event('policy_close_error',error=str(exc))
        if errors:
            raise RuntimeError('CLEANUP FAILED; operator hardware stop required: '+str(errors))


