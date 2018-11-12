#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
# from apex_playground.srv import ErgoPose, ErgoPoseResponse


class ErgoEefPos(object):
    def __init__(self):
        self.apex_name = "apex_1"
        # self.rospack = RosPack()
        # print("Ego Eff Pose on {}".format(n_apex))
        # rospy.Subscriber(self.topics["torso_l_eef"]["topic"], self.topics["torso_l_eef"]["type"], self.cb_eef)
        # self.sub = rospy.Subscriber("/apex_1/poppy_ergo_jr/end_effector_pose", PoseStamped, self.cb_eef)

    def cb_eef(self, msg):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg)
        self.eef_pose = msg

    def run(self):
        # rospy.wait_for_service(service)
        # self.set_torso_compliant_srv = rospy.ServiceProxy(self.service_name_set_compliant, SetTorsoCompliant)
        # rospy.Service(self.service_name_get, GetSensorialState, self.cb_get)
        rospy.loginfo("Done, perception is up!")
        rospy.Subscriber("/apex_1/poppy_ergo_jr/end_effector_pose", PoseStamped, self.cb_eef)
        rospy.loginfo("Done, perception is down!")
        rospy.spin()

    # def get(self):
    #    return self.eef_pose

    @property
    def ergo(self):
        return self.eef_pose


class ErgoEefService(object):
    def __init__(self):
        self.pos = ErgoEefPos()

    def run(self):
        rospy.loginfo("Perception is down!")
        rospy.Service("ergoeff", ErgoPose, self.get_pos)
        rospy.loginfo("Done, perception is up!")
        rospy.spin()

    def get_pos(self):
        return self.pos.ergo


if __name__ == '__main__':
    rospy.init_node('ergoeff')
    ErgoEefPos().run()
    # ErgoEefService().run()
