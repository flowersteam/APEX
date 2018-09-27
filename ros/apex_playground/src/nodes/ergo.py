#!/usr/bin/env python

import rospy
import rosnode
import json
from std_msgs.msg import Bool
from apex_playground.srv import *
from apex_playground.msg import *
from poppy_msgs.srv import ReachTarget, ReachTargetRequest, SetCompliant, SetCompliantRequest
from sensor_msgs.msg import Joy, JointState
from std_msgs.msg import Bool
from rospkg import RosPack
from os.path import join
from apex_playground.ergo.button import Button


class Ergo(object):
    def __init__(self):
        self.rospack = RosPack()
        with open(join(self.rospack.get_path('apex_playground'), 'config', 'ergo.json')) as f:
            self.params = json.load(f)
        self.button = Button(self.params)
        self.rate = rospy.Rate(self.params['publish_rate'])
        self.button.switch_led(False)

        # Service callers
        self.robot_reach_srv_name = '{}/reach'.format(self.params['robot_name'])
        self.robot_compliant_srv_name = '{}/set_compliant'.format(self.params['robot_name'])
        rospy.loginfo("Ergo node is waiting for poppy controllers...")
        rospy.wait_for_service(self.robot_reach_srv_name)
        rospy.wait_for_service(self.robot_compliant_srv_name)
        self.reach_proxy = rospy.ServiceProxy(self.robot_reach_srv_name, ReachTarget)
        self.compliant_proxy = rospy.ServiceProxy(self.robot_compliant_srv_name, SetCompliant)
        rospy.loginfo("Controllers connected!")

        self.state_pub = rospy.Publisher('ergo/state', CircularState, queue_size=1)
        self.button_pub = rospy.Publisher('sensors/buttons/help', Bool, queue_size=1)

        self.goals = []
        self.goal = 0.
        self.joy_x = 0.
        self.joy_y = 0.
        self.motion_started_joy = 0.
        self.js = JointState()
        rospy.Subscriber('sensors/joystick/{}'.format(self.params["control_joystick_id"]), Joy, self.cb_joy)
        rospy.Subscriber('{}/joint_state'.format(self.params['robot_name']), JointState, self.cb_js)
        rospy.Subscriber('sensors/button_leds/pause', Bool, self.cb_bt_led)

        self.t = rospy.Time.now()
        self.srv_reset = None
        self.extended = False
        self.standby = False
        self.last_activity = rospy.Time.now()
        self.delta_t = rospy.Time.now()

    def cb_bt_led(self, msg):
        self.button.switch_led(msg.data)

    def cb_js(self, msg):
        self.js = msg

    def reach(self, target, duration):
        js = JointState()
        js.name = target.keys()
        js.position = target.values()
        self.reach_proxy(ReachTargetRequest(target=js,
                                            duration=rospy.Duration(duration)))

    def set_compliant(self, compliant):
        self.compliant_proxy(SetCompliantRequest(compliant=compliant))

    def cb_joy(self, msg):
        self.joy_x = msg.axes[0]
        self.joy_y = msg.axes[1]

    def go_to_start(self, slow=True):
        self.go_to([0.0, -15.4, 35.34, 0.0, -15.69, 71.99], 4 if slow else 1)

    def go_to_extended(self):
        extended = {'m2': 60, 'm3': -37, 'm5': -50, 'm6': 96}
        self.reach(extended, 0.5)
        self.extended = True

    def go_to_rest(self):
        if self.extended:
            rest = {'m2': -26, 'm3': 59, 'm5': -30, 'm6': 78}
            self.reach(rest, 0.5)
            self.extended = False

    def is_controller_running(self):
        return len([node for node in rosnode.get_node_names() if rospy.get_namespace() + 'controller' in node or '/manager' == node]) > 1

    def go_or_resume_standby(self):
        recent_activity = rospy.Time.now() - self.last_activity < rospy.Duration(self.params['auto_standby_duration'])
        if recent_activity and self.standby:
            rospy.loginfo("Ergo is resuming from standby")
            self.set_compliant(False)
            self.standby = False
        elif not self.standby and not recent_activity:
            rospy.loginfo("Ergo is entering standby mode")
            self.standby = True
            self.set_compliant(True)

        if self.is_controller_running():
            self.last_activity = rospy.Time.now()

    def go_to(self, motors, duration):
        self.goals = motors
        self.goal = self.goals[0] - self.goals[3]
        self.reach(dict(zip(['m1', 'm2', 'm3', 'm4', 'm5', 'm6'], motors)), duration)
        rospy.sleep(duration)

    def run(self):
        self.go_to_start()
        self.last_activity = rospy.Time.now()
        self.srv_reset = rospy.Service('ergo/reset', Reset, self._cb_reset)
        rospy.loginfo('Ergo is ready and starts joystick servoing...')
        self.t = rospy.Time.now()

        while not rospy.is_shutdown():
            now = rospy.Time.now()
            self.delta_t = (now - self.t).to_sec()
            self.t = now

            self.go_or_resume_standby()
            self.servo_robot(self.joy_y, self.joy_x)
            self.publish_state()
            self.publish_button()

            # Update the last activity
            if abs(self.joy_x) > self.params['min_joy_activity'] or abs(self.joy_y) > self.params['min_joy_activity']:
                self.last_activity = rospy.Time.now()

            self.rate.sleep()

    def servo_axis_rotation(self, x):
        if abs(x) > self.params['sensitivity_joy']:
            x = x if abs(x) > self.params['sensitivity_joy'] else 0
            min_x = self.params['bounds'][0][0] + self.params['bounds'][3][0]
            max_x = self.params['bounds'][0][1] + self.params['bounds'][3][1]
            self.goal = min(max(min_x, self.goal + self.params['speed']*x*self.delta_t), max_x)

            if self.goal > self.params['bounds'][0][1]:
                new_x_m3 = self.goal - self.params['bounds'][0][1]
                new_x = self.params['bounds'][0][1]
            elif self.goal < self.params['bounds'][0][0]:
                new_x_m3 = self.goal - self.params['bounds'][0][0]
                new_x = self.params['bounds'][0][0]
            else:
                new_x = self.goal
                new_x_m3 = 0

            new_x_m3 = max(min(new_x_m3, self.params['bounds'][3][1]), self.params['bounds'][3][0])
            self.reach({'m1': new_x, 'm4': new_x_m3}, 0)  # Duration = 0 means joint teleportation

    def servo_axis_elongation(self, x):
        if -x > self.params['min_joy_elongation']:
            self.go_to_extended()
        else:
            self.go_to_rest()

    def servo_robot(self, x, y):
        now = rospy.Time.now().to_sec()
        max_abs = max(abs(y), abs(x))
        if max_abs > self.params['sensitivity_joy'] and self.motion_started_joy == 0.:
            self.motion_started_joy = now

        elif max_abs < self.params['sensitivity_joy'] and self.motion_started_joy > 0.:
            self.motion_started_joy = 0.
            self.servo_axis_elongation(0)

        elif self.motion_started_joy > 0. and now - self.motion_started_joy > self.params['delay_joy']:
            self.servo_axis_rotation(y)
            self.servo_axis_elongation(x)

    def publish_button(self):
        self.button_pub.publish(Bool(data=self.button.pressed))

    def publish_state(self):
        # TODO We might want a better state here, get the arena center, get EEF and do the maths as in environment/get_state
        if 'm1' in self.js.name and 'm4' in self.js.name:
            angle = self.js.position[0] + self.js.position[3]
            self.state_pub.publish(CircularState(angle=angle, extended=self.extended))

    def _cb_reset(self, request):
        rospy.loginfo("Resetting Ergo...")
        self.go_to_start(request.slow)
        return ResetResponse()


if __name__ == '__main__':
    rospy.init_node('ergo')
    Ergo().run()
