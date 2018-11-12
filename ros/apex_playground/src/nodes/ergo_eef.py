#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from apex_playground.srv import ErgoPose, ErgoPoseResponse


class ErgoEefPos(object):
    def __init__(self):
        self.sub = rospy.Subscriber("poppy_ergo_jr/end_effector_pose", PoseStamped, self.cb_eef)

    def cb_eef(self, msg):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.pose.position)
        self.cur_eef_pos = msg

    @property
    def eef_pos(self):
        return self.cur_eef_pos


class ErgoEefService(object):
    def __init__(self):
        self.ergo_pos = ErgoEefPos()

    def run(self):
        rospy.Service("ergoeff", ErgoPose, self.get_pos)
        rospy.loginfo("Done, end effector position perception is up!")
        rospy.spin()

    def get_pos(self, req):
        return self.ergo_pos.eef_pos


if __name__ == '__main__':
    rospy.init_node('ergoeff')
    ergo = ErgoEefService()
    ergo.run()
