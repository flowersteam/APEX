#!/usr/bin/python

import rospy
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from rospkg import RosPack
from apex_playground.srv import Camera, CameraRequest
import json
import os
from os.path import join
import numpy as np


class CameraViewer(object):
    def __init__(self):
        self.rospack = RosPack()
        self.apex_name = os.environ.get("ROS_HOSTNAME").replace("-ergo.local", "").replace("-", "_");
        print("CameraViewer on {}", self.apex_name)
        
    def show_image(self):
        rospy.wait_for_service('/{}/camera'.format(self.apex_name))
        read = rospy.ServiceProxy('/{}/camera'.format(self.apex_name), Camera)
        image = [x.data for x in read(CameraRequest()).image]
        image = np.array(image).reshape(144,176,3)
        plt.imshow(image)
        plt.show()

if __name__ == "__main__":
    camera = CameraViewer()
    camera.show_image()
