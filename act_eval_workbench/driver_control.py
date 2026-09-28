"""Targets for the existing firmware move_j planner, without a second rate limiter.

This module creates no ROS or CAN interfaces. Hardware owns interpolation;
feedback validation, workspace checks and stop supervision remain in the worker.
"""
import numpy as np


def feedback_snapshot(feedback, now, ros_now, stationary=False):
    """Legacy freshness/name checks, without its diagnostic 0.2 rad/s ceiling."""
    from . import controller as c
    if feedback.error:
        raise RuntimeError(feedback.error)
    latest=[]
    for arm in c.pos.ARMS:
        rows=feedback.rows[arm]
        if not rows:raise RuntimeError(f'{arm}: missing feedback')
        last=rows[-1]
        if not 0<=now-last['received']<=.2 or not -.05<=ros_now-last['stamp']<=.2:
            raise RuntimeError(f'{arm}: stale/future feedback (>0.2 s)')
        if feedback.require_open and not .095<=last['width']<=.101:
            raise RuntimeError(f'{arm}: gripper must already be open')
        if stationary:
            recent=[r for r in rows if now-r['received']<=.8]
            if len(recent)<10 or recent[-1]['stamp']-recent[0]['stamp']<.5:
                raise RuntimeError(f'{arm}: need 0.5 s of unique stationary feedback')
            if np.ptp(np.array([r['q'] for r in recent]),axis=0).max()>.005:
                raise RuntimeError(f'{arm}: arm not stationary within 0.005 rad')
        latest.append(last)
    if abs(latest[0]['stamp']-latest[1]['stamp'])>.05:
        raise RuntimeError('left/right source skew >0.05 s')
    return c.pos.limits(np.array([r['q'] for r in latest])),[r['width'] for r in latest]


class DriverTargets:
    """Pass absolute targets to move_j; never generate an application ramp."""
    def __init__(self,q,widths):
        self.x=np.r_[np.asarray(q).reshape(-1),widths].astype(float)

    def next(self,q,widths,elapsed,actual_q=None):
        from .common import limits
        if not .05-1e-6<=elapsed<=.35:
            raise RuntimeError('control interval outside [0.05,0.35] s')
        q=limits(q).copy();widths=np.asarray(widths,float)
        if widths.shape!=(2,) or not np.isfinite(widths).all() or np.any((widths<0)|(widths>.1)):
            raise ValueError('gripper target outside [0,0.1] m')
        self.x=np.r_[q.reshape(-1),widths]
        return q,widths.copy()


def driver_preflight(io):
    """Read-only readiness/configuration audit; accepts the driver 1-100% range."""
    from . import controller as c
    # The sealed legacy preflight always sleeps three seconds. For reset,
    # proceed as soon as fresh stationary feedback / ROS discovery is ready.
    from rcl_interfaces.srv import GetParameters
    began=io.now();deadline=began+9.;last_error='feedback not ready'
    while True:
        io.pump()
        if io.interrupted or (io.gui and (io.gui.closed or io.gui.stopped)):
            raise RuntimeError('cancelled before motion execution')
        try:
            io.snapshot(stationary=True);io.camera()
            if any(not io.clients[a,'params'].service_is_ready() for a in c.pos.ARMS):
                raise RuntimeError('parameter services not ready')
            services=dict(io.node.get_service_names_and_types())
            for arm in c.pos.ARMS:
                for service in ('control_enable','enable_agx_arm','emergency_stop'):
                    expected='std_srvs/srv/Empty' if service=='emergency_stop' else 'std_srvs/srv/SetBool'
                    if expected not in services.get(f'/{arm}_arm/{service}',[]):
                        raise RuntimeError(f'{arm}: missing {service} service')
            break
        except RuntimeError as exc:last_error=str(exc)
        if io.now()>=deadline:raise RuntimeError('motion preflight not ready: '+last_error)
        io.idle()
    io.check_publishers();futures={}
    for arm in c.pos.ARMS:
        request=GetParameters.Request();request.names=['arm_type','speed_percent','auto_enable']
        futures[arm]=io.clients[arm,'params'].call_async(request)
    params={}
    for arm,response in io.wait_futures(futures).items():
        if response is None or len(response.values)!=3:raise RuntimeError('invalid parameter response')
        kind,speed,auto=response.values
        if kind.type!=4 or 'nero' not in kind.string_value or speed.type!=2 or not 1<=speed.integer_value<=100:
            raise RuntimeError(f'{arm}: require nero and actual startup speed_percent in [1,100]')
        if auto.type!=1 or auto.bool_value:raise RuntimeError(f'{arm}: require auto_enable=false startup configuration')
        params[arm]=dict(arm_type=kind.string_value,speed_percent=speed.integer_value,auto_enable=auto.bool_value)
    io.driver_parameters=params
    io.event('read_only_preflight',driver_parameters=params,readiness_driven=True,elapsed_s=io.now()-began)
    io.check_publishers()
    return io.snapshot(stationary=True)

