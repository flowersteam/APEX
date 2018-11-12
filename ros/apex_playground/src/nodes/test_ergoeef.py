import rospy
from rospkg import RosPack
from geometry_msgs.msg import PoseStamped


class ErgoEefPose(object):
    def __init__(self, n_apex):
        self.apex_name = "apex_{}".format(n_apex)
        # self.rospack = RosPack()
        print("Ego Eff Pose on {}".format(n_apex))
        # rospy.Subscriber(self.topics["torso_l_eef"]["topic"], self.topics["torso_l_eef"]["type"], self.cb_eef)
        self.sub = rospy.Subscriber("{}/end_effector_pose".format(n_apex), PoseStamped, self.cb_eef)

    def cb_eef(self, msg):
        self.eef_pose = msg

    @property
    def ergo(self):
        return self.eef_pose

    # def get_eff_pose(self):


if __name__ == '__main__':
    ergo_eff_pose = ErgoEefPose(1)
    print(ergo_eff_pose.ergo)
