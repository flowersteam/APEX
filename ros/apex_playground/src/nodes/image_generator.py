#!/usr/bin/python

import random
import string
import argparse
import rospy
from poppy_msgs.srv import ReachTarget, ReachTargetRequest, SetCompliant, SetCompliantRequest
from apex_playground.srv import Camera, CameraRequest
from sensor_msgs.msg import JointState
import os
import numpy as np
from explauto.utils import bounds_min_max
import scipy.misc
import datetime

from apex_playground.learning.dmp.mydmp import MyDMP


class CameraRecorder(object):
    def __init__(self, n_apex):
        self.apex_name = "apex_{}".format(n_apex)
        print("CameraRecorder on ", self.apex_name)

    def get_image(self):
        rospy.wait_for_service('/{}/camera'.format(self.apex_name))
        read = rospy.ServiceProxy('/{}/camera'.format(self.apex_name), Camera)
        image = [x.data for x in read(CameraRequest()).image]
        image = np.array(image).reshape(144, 176, 3)
        return image

class ErgoDMP(object):
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
        rospy.sleep(duration-0.05)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save images of the arena.')
    parser.add_argument('--path', metavar='-p', type=str, help='path to save images')
    parser.add_argument('--apex', metavar='-a', type=int, help='ergo number')
    parser.add_argument('--n-iter', metavar='-n', type=int, help='number of images to take')
    args = parser.parse_args()
    camera = CameraRecorder(args.apex)
    mover = ErgoDMP(args.apex)
    mover.set_compliant(False)

    n_dmps = 6
    n_bfs = 7
    timesteps = 30
    max_params = np.array([300.] * n_bfs * n_dmps + [1.] * n_dmps)
    bounds_motors_max = np.array([180, 10, 20, 10, 30, 30])
    bounds_motors_min = np.array([-180, -20, -20, -15, -20, -20])
    dmp = MyDMP(n_dmps=n_dmps, n_bfs=n_bfs, timesteps=timesteps, max_params=max_params)

    for _ in range(args.n_iter):
        point = [0, 0, 0, 0, 0, 0]
        mover.move_to(list(point), duration=1)
        m = np.random.randn(dmp.n_dmps * dmp.n_bfs + n_dmps) * max_params
        normalized_traj = dmp.trajectory(m)
        normalized_traj = bounds_min_max(normalized_traj, n_dmps * [-1.], n_dmps * [1.])
        traj = ((normalized_traj - np.array([-1.] * n_dmps)) / 2.) * (bounds_motors_max - bounds_motors_min) + bounds_motors_min
        for m in traj:
            mover.move_to(list(m))
        image = np.flip(camera.get_image(), axis=2)
        filename = '{}-{}'.format(args.apex, datetime.datetime.now())
        scipy.misc.imsave(os.path.join(args.path, filename) + '.jpeg', image)
