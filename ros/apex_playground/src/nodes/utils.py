#!/usr/bin/python

import rospy
from poppy_msgs.srv import ReachTarget, ReachTargetRequest, SetCompliant, SetCompliantRequest
from apex_playground.srv import Camera, CameraRequest, ErgoPose, ErgoPoseRequest
from sensor_msgs.msg import JointState
import numpy as np
import cv2
from numpy import arctan2, sqrt
from collections import deque


class BallTracker(object):
    def __init__(self, parameters):
        self.params = parameters

        # initialize the lists of tracked points in a map and the coordinate deltas
        self.pts = {}
        self.dX, self.dY = (0, 0)

    def get_images(self, frame):
        # resize the frame, blur it, and convert it to the HSV color space
        #frame = imutils.resize(frame, width=600)
        # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

        mask_ball = cv2.inRange(hsv, self.params['tracking']['ball']['lower'], self.params['tracking']['ball']['upper'])
        mask_ball = cv2.erode(mask_ball, None, iterations=2)
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
                return center, radius
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
        Reduce the current configuration of the ball in (angle, theta)
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
        image = np.array(image).reshape(144, 176, 3)
        image = cv2.cvtColor(image.astype('uint8'), cv2.COLOR_BGR2RGB)
        return image


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


class ErgoTracker(object):
    def __init__(self, n_apex):
        self.apex_name = "apex_{}".format(n_apex)
        rospy.wait_for_service('/{}/ergoeff'.format(self.apex_name))
        self.get_msg = rospy.ServiceProxy('/{}/ergoeff'.format(self.apex_name), ErgoPose)
        print("ErgoTracker on ", self.apex_name)

    def get_position(self):
        msg = self.get_msg(ErgoPoseRequest())
        pos = np.array([msg.pos.pose.position.x, msg.pos.pose.position.y, msg.pos.pose.position.z])
        return pos


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("tkagg")
    import matplotlib.pyplot as plt

    from os.path import join
    import json

    from rospkg import RosPack

    camera = CameraRecorder(1)
    img = camera.get_image()
    print(img.shape)
    plt.imshow(img)
    plt.show()

    rospack = RosPack()
    with open(join(rospack.get_path('apex_playground'), 'config', 'environment.json')) as f:
        params = json.load(f)
    params['tracking']['ball']['lower'] = tuple(params['tracking']['ball']['lower'])
    params['tracking']['ball']['upper'] = tuple(params['tracking']['ball']['upper'])
    params['tracking']['arena']['lower'] = tuple(params['tracking']['arena']['lower'])
    params['tracking']['arena']['upper'] = tuple(params['tracking']['arena']['upper'])

    tracking = BallTracker(params)

    frame = img
    hsv, mask_ball, mask_arena = tracking.get_images(frame)

    min_radius_ball = params['tracking']['resolution'][0] * params['tracking']['resolution'][1] / 20000.
    ball_center, _ = tracking.find_center('ball', frame, mask_ball, min_radius_ball)

    min_radius_arena = params['tracking']['resolution'][0] * params['tracking']['resolution'][1] / 2000.
    arena_center, arena_radius = tracking.find_center('arena', frame, mask_arena, min_radius_arena)

    ring_radius = int(arena_radius / params['tracking']['ring_divider']) if arena_radius is not None else None

    if ball_center is not None and arena_center is not None:
        elongation, theta = tracking.get_state(ball_center, arena_center)

    frame = tracking.draw_images(frame, hsv, mask_ball, mask_arena, arena_center, ring_radius)
    plt.imshow(frame)
    plt.show()
