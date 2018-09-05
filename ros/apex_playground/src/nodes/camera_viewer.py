#!/usr/bin/python

import rospy
import matplotlib.pyplot as plt
from rospkg import RosPack
from apex_playground import Camera, CameraRequest
import json
from os.path import join



class CameraViewer(object):
    def __init__(self):
        self.rospack = RosPack()
        with open(join(self.rospack.get_path('apex_playground'), 'config', 'ergo.json')) as f:
            self.params = json.load(f)
        print(self.params)
        
    def show_image(self):
        rospy.wait_for_service('/{}/camera'.format(self.params['robot_name']))
        read = rospy.ServiceProxy('/{}/camera'.format(self.params['robot_name']), Camera)
        image = read(CameraRequest())
        plt.imshow(image)
        plt.show()

if __name__ == "__main__":
    camera = CameraViewer()
    camera.show_image()