#!/usr/bin/env python
from collections import deque
import numpy as np
import cv2
import rospy
from apex_playground.srv import Camera

class CameraService(object):

    def __init__(self, height, width):
        self.camera = cv2.VideoCapture(0)
        self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
        self.camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        rospy.Service("camera", Camera, self.read)

    def read(self, req):
        success, image = self.camera.read()
        if not success:
            raise Exception("Failed to read camera...")
        return image

    def close(self):
        # cleanup the camera and close any open windows
        self.camera.release()
        cv2.destroyAllWindows()

if __name__=="__main__":
    rospy.init_node("camera")
    camera = CameraService(352, 288)
    rospy.spin() 