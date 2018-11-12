import numpy as np
import rospy
from apex_playground.srv import ErgoPose, ErgoPoseRequest


class ErgoPos(object):
    def __init__(self, n_apex):
        self.apex_name = "apex_{}".format(n_apex)
        rospy.wait_for_service('/{}/ergoeff'.format(self.apex_name))
        self.get_msg = rospy.ServiceProxy('/{}/ergoeff'.format(self.apex_name), ErgoPose)
        print("ErgoPose on ", self.apex_name)

    def get_position(self):
        # msg = self.get_msg()
        msg = self.get_msg(ErgoPoseRequest())
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        print(pos)


if __name__ == "__main__":
    ergo_pos = ErgoPos(1)
    ergo_pos.get_position()
