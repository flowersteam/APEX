#!/usr/bin/python

import rospy
from poppy_msgs.srv import ReachTarget, ReachTargetRequest, SetCompliant, SetCompliantRequest
from apex_playground.srv import Camera, CameraRequest
from sensor_msgs.msg import JointState
import numpy as np
import cv2
from numpy import arctan2, sqrt
from collections import deque


class BallTracking(object):
    def __init__(self, parameters):
        self.params = parameters

        # initialize the lists of tracked points in a map and the coordinate deltas
        self.pts = {}
        self.dX, self.dY = (0, 0)

    def get_images(self, frame):
        frame = frame[16:-16, 48:-48]
        # resize the frame, blur it, and convert it to the HSV color space
        #frame = imutils.resize(frame, width=600)
        # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        mask_ball = cv2.inRange(hsv, self.params['tracking']['ball']['lower'], self.params['tracking']['ball']['upper'])
        mask_ball = cv2.erode(mask_ball, None, iterations=4)
        mask_ball = cv2.dilate(mask_ball, None, iterations=10)

        mask_arena = cv2.inRange(hsv, self.params['tracking']['arena']['lower'], self.params['tracking']['arena']['upper'])
        mask_arena = cv2.erode(mask_arena, None, iterations=4)
        mask_arena = cv2.dilate(mask_arena, None, iterations=10)

        return hsv, mask_ball, mask_arena

    def find_center(self, name, frame, mask, min_radius):
        if name not in self.pts:
            self.pts[name] = deque(maxlen=self.params['tracking']['buffer_size'])

        # find contours in the mask and initialize the current (x, y) center of the ball
        cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        # only proceed if at least one contour was found
        if len(cnts) > 0:
            # find the largest contour in the mask, then use it to compute the minimum enclosing circle and centroid
            c = max(cnts, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            center = (int(x), int(y))

            # only proceed if the radius meets a minimum size
            if radius > min_radius:
                # draw the circle and centroid on the frame, then update the list of tracked points
                cv2.circle(frame, center, int(radius), (0, 255, 255), 2)
                cv2.circle(frame, center, 5, (0, 0, 255), -1)
                self.pts[name].appendleft(center)
                smooth_points = 8
                return (int(np.mean([self.pts[name][i][0] for i in range(min(smooth_points, len(self.pts[name])))])),
                        int(np.mean([self.pts[name][i][1] for i in range(min(smooth_points, len(self.pts[name])))]))), radius
        return None, None

    def draw_images(self, frame, hsv, mask_ball, mask_arena, arena_center, arena_ring_radius=None):
        self.draw_history(frame, 'ball')
        self.draw_history(frame, 'arena')
        if arena_ring_radius is not None:
            cv2.circle(frame, arena_center, arena_ring_radius, (0, 128, 255), 2)
        return frame

    def draw_history(self, frame, name):
        # loop over the set of tracked points
        if name in self.pts:
            for i in np.arange(1, len(self.pts[name])):
                # if either of the tracked points are None, ignore
                # them
                if self.pts[name][i - 1] is None or self.pts[name][i] is None:
                    continue

                # check to see if enough points have been accumulated in
                # the buffer
                if len(self.pts[name]) >= 10 and i == 1 and self.pts[name][-10] is not None:
                    # compute the difference between the x and y
                    # coordinates and re-initialize the direction
                    # text variables
                    self.dX = self.pts[name][-10][0] - self.pts[name][i][0]
                    self.dY = self.pts[name][-10][1] - self.pts[name][i][1]

                # otherwise, compute the thickness of the line and
                # draw the connecting lines
                thickness = int(np.sqrt(self.params['tracking']['buffer_size'] / float(i + 1)) * 2.5)
                cv2.line(frame, self.pts[name][i - 1], self.pts[name][i], (0, 0, 255), thickness)
            if len(self.pts[name]) > 1:
                # show the movement deltas of movement
                cv2.putText(frame, "dx: {}, dy: {}".format(self.dX, self.dY),
                            (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.35, (0, 0, 255), 1)

    def get_state(self, ball_center, arena_center):
        """
        Reduce the current joint configuration of the ergo in (angle, theta)
        :return: a CircularState
        """
        x, y = (ball_center[0] - arena_center[0], ball_center[1] - arena_center[1])
        elongation = sqrt(x*x + y*y)
        theta = arctan2(y, x)
        return elongation, theta


class CameraRecorder(object):
    def __init__(self, n_apex):
        self.apex_name = "apex_{}".format(n_apex)
        print("CameraRecorder on ", self.apex_name)

    def get_image(self):
        rospy.wait_for_service('/{}/camera'.format(self.apex_name))
        read = rospy.ServiceProxy('/{}/camera'.format(self.apex_name), Camera)
        image = [x.data for x in read(CameraRequest()).image]
        image = np.array(image).reshape(288, 352, 3)
        return np.flip(image, axis=2)


class ErgoMover(object):
    def __init__(self, n_apex):
        self._apex_name = "apex_{}".format(n_apex)
        self._reach_service_name = '/{}/poppy_ergo_jr/reach'.format(self._apex_name)
        rospy.wait_for_service(self._reach_service_name)
        self._reach_service_prox = rospy.ServiceProxy(self._reach_service_name, ReachTarget)
        self._compliant_service_name = '/{}/poppy_ergo_jr/set_compliant'.format(self._apex_name)
        rospy.wait_for_service(self._compliant_service_name)
        self._compliant_service_prox = rospy.ServiceProxy(self._compliant_service_name, SetCompliant)

    def set_compliant(self, compliant):
        self._compliant_service_prox(SetCompliantRequest(compliant=compliant))

    def move_to(self, point, duration=0.2):
        reach_jointstate = JointState(position=point, name=["m{}".format(i) for i in range(1, 7)])
        reach_request = ReachTargetRequest(target=reach_jointstate,
                                           duration=rospy.Duration(duration))
        self._reach_service_prox(reach_request)
        rospy.sleep(duration - 0.05)


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("tkagg")
    import matplotlib.pyplot as plt

    camera = CameraRecorder(1)
    image = camera.get_image()
    plt.imshow(image)
    plt.show()
