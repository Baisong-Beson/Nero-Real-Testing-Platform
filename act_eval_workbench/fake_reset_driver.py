"""Reset/disable test driver. Isolated ROS only; no SDK, CAN or hardware."""
import os
import time
import numpy as np
from .test_integration import isolated
from .common import ready_pose

def main():
    isolated()
    if os.environ.get('ACT_EVAL_FAKE_DRIVER')!='1':raise RuntimeError('fake flag required')
    import rclpy
    from rclpy.node import Node
    from rclpy.executors import SingleThreadedExecutor
    from sensor_msgs.msg import JointState
    from agx_arm_msgs.msg import GripperStatus
    from std_srvs.srv import SetBool,Empty
    rclpy.init()
    class Driver(Node):
        def __init__(self,arm,index):
            super().__init__('agx_arm_ctrl_single_node',namespace=f'/{arm}_arm')
            for k,v in [('arm_type','nero'),('speed_percent',35),('auto_enable',False)]:self.declare_parameter(k,v)
            self.q=np.array(ready_pose(os.environ.get('FAKE_TASK','plate'))['joints_rad'][index])
            self.q[0]+=float(os.environ.get('FAKE_START_J1_OFFSET','0'));self.target=self.q.copy()
            self.width=float(os.environ.get('FAKE_START_WIDTH_'+arm.upper(),os.environ.get('FAKE_START_WIDTH','.029')));self.target_w=self.width
            self.contact_width=float(os.environ.get('FAKE_RIGHT_CONTACT_WIDTH','0')) if arm=='right' else 0.
            self.enabled=self.grip_enabled=os.environ.get('FAKE_START_ENABLED','0')=='1';self.gate=False
            self.pub=self.create_publisher(JointState,'feedback/joint_states',1)
            self.status=self.create_publisher(GripperStatus,'feedback/gripper_status',1)
            self.create_subscription(JointState,'control/move_j',self.command,1)
            self.create_subscription(JointState,'control/joint_states',self.gripper,1)
            self.create_service(SetBool,'control_enable',self.control)
            self.create_service(SetBool,'enable_agx_arm',self.enable)
            self.create_service(Empty,'emergency_stop',self.hold)
            self.create_timer(.005,self.tick)
        def control(self,req,res):self.gate=req.data;res.success=True;return res
        def enable(self,req,res):
            self.enabled=req.data
            if not req.data:self.grip_enabled=False;self.gate=False
            res.success=True;res.message='Arm/gripper disabled' if not req.data else 'Arm enabled';return res
        def hold(self,req,res):self.target=self.q.copy();return res
        def command(self,msg):
            assert list(msg.name)==['joint'+str(i) for i in range(1,8)] and len(msg.position)==7
            if self.gate and self.enabled:self.target=np.array(msg.position)
        def gripper(self,msg):
            assert list(msg.name)==['gripper'] and list(msg.effort)==[1.] and 0<=msg.position[0]<=.1
            if self.gate and self.enabled:self.target_w=msg.position[0];self.grip_enabled=True
        def tick(self):
            if self.enabled:self.q+=np.clip(self.target-self.q,-.00085,.00085)
            if self.grip_enabled and os.environ.get('FAKE_GRIP_STUCK')!='1':self.width+=np.clip(max(self.target_w,self.contact_width)-self.width,-.0002,.0002)
            stamp=self.get_clock().now().to_msg()
            msg=JointState();msg.header.stamp=stamp
            msg.name=['joint'+str(i) for i in range(1,8)]+['gripper'];msg.position=self.q.tolist()+[float(self.width)];self.pub.publish(msg)
            status=GripperStatus();status.header.stamp=stamp;status.width=float(self.width)
            if self.grip_enabled and self.target_w<self.contact_width and abs(self.width-self.contact_width)<1e-6:status.force=1.
            status.driver_enable_status=self.grip_enabled;self.status.publish(status)
    nodes=[Driver(a,i) for i,a in enumerate(('right','left'))];executor=SingleThreadedExecutor()
    for node in nodes:executor.add_node(node)
    try:executor.spin()
    finally:
        for node in nodes:node.destroy_node()
        rclpy.shutdown()

if __name__=='__main__':main()
