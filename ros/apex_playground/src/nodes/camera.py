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

    def read(self):
        success, image = self.camera.read()
        if not success:
            raise Execption("Failed to read camera...")
        return image

    def close(self):
        # cleanup the camera and close any open windows
        self.camera.release()
        cv2.destroyAllWindows()

if __name__="__main__":
    camera = CameraService(352, 288)
    rospy.Service("camera", Camera, camera.read)
    rospy.spin() 