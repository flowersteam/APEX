#!/usr/bin/env python
from collections import deque
import numpy as np
import cv2
import rospy
from apex_playground.srv import Camera, CameraResponse
from std_msgs.msg import Float32
from threading import Thread, Lock
import time


class CameraService(object):

    def __init__(self, height, width):
        camera = cv2.VideoCapture(0)
        camera.set(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT, height)
        camera.set(cv2.cv.CV_CAP_PROP_FRAME_WIDTH, width)
        camera.set(cv2.cv.CV_CAP_PROP_FPS, 30)
        self.frame = np.zeros(1)
        self.frame_lock = Lock()
        self.exit = False
        self.exit_lock = Lock()
        self.thread = Thread(target=self.__in_thread_loop, args=(camera,))
        self.thread.start()
        rospy.Service("camera", Camera, self.read)

    def __in_thread_loop(self, camera):
        while True:
            with self.exit_lock:
                do_exit = self.exit
            success, frame = camera.read()
            if do_exit:
                camera.release()
                break
            if not success:
                rospy.logerr("Frame acquire failed")
                continue
            else:
                with self.frame_lock:
                    self.frame = frame
                time.sleep(1/20)

    def read(self, req):
        with self.frame_lock:
            image = self.frame.copy()
        resp = CameraResponse()
        resp.image = [Float32(p) for p in image.astype(np.float32).flatten()]
        return resp

    def close(self):
        with self.exit_lock:
            self.exit = True
        self.thread.join()


if __name__=="__main__":
    rospy.init_node("camera")
    camera = CameraService(352, 288)
    rospy.spin() 
    camera.close()
