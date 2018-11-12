import rospy
from geometry_msgs.msg import PoseStamped
from apex_playground.srv import ErgoPose, ErgoPoseRequest


class ErgoPos(object):
    def __init__(self, n_apex):
        self.apex_name = "apex_{}".format(n_apex)
        print("ErgoPose on ", self.apex_name)

    def get_pose(self):
        rospy.wait_for_service('/{}/ergoeff'.format(self.apex_name))


        read = rospy.ServiceProxy('/{}/ergoeff'.format(self.apex_name), Camera)
        image = [x.data for x in read(CameraRequest()).image]
        image = np.array(image).reshape(144, 176, 3)
        plt.imshow(image)
        plt.show()

if __name__ == "__main__":
    camera = CameraViewer()
    camera.show_image()


if __name__ == '__main__':
    rospy.init_node('ergoeff')
    ErgoEefPose().run()