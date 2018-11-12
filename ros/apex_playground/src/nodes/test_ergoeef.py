import rospy
from apex_playground.srv import ErgoPose, ErgoPoseRequest


class ErgoPos(object):
    def __init__(self, n_apex):
        self.apex_name = "apex_{}".format(n_apex)
        rospy.wait_for_service('/{}/ergoeff'.format(self.apex_name))
        self.get_pos = rospy.ServiceProxy('/{}/ergoeff'.format(self.apex_name), ErgoPose)
        print("ErgoPose on ", self.apex_name)

    def get_pos(self):
        msg = self.get_pos(ErgoPoseRequest())
        pos = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z])
        print(pos)


if __name__ == "__main__":
    camera = ErgoPos(1)
    camera.get_pos()
