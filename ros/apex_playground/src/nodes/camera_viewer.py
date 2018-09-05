#!/usr/bin/python

import rospy
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
from rospkg import RosPack
from apex_playground.srv import Camera, CameraRequest
import json
from os.path import join
import numpy as np


class CameraViewer(object):
    def __init__(self):
        self.rospack = RosPack()
        with open(join(self.rospack.get_path('apex_playground'), 'config', 'ergo.json')) as f:
            self.params = json.load(f)
        print(self.params)
        
    def show_image(self):
        rospy.wait_for_service('/apex_1/camera')
        read = rospy.ServiceProxy('/apex_1/camera', Camera)
        image = [x.data for x in read(CameraRequest()).image]
        image = np.array(image).reshape(144,176,3)
	plt.imshow(image)
        plt.show()

if __name__ == "__main__":
    camera = CameraViewer()
    camera.show_image()
