#!/usr/bin/python

import argparse
import rospy
from poppy_msgs.srv import ReachTarget, ReachTargetRequest, SetCompliant, SetCompliantRequest
from sensor_msgs.msg import JointState
import os
import numpy as np
from explauto.utils import bounds_min_max
import scipy.misc
import datetime
import json
import pickle

from apex_playground.learning.dmp.mydmp import MyDMP
from utils import BallTracker, CameraRecorder, ErgoTracker


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


class PosExtractor(object):
    def __init__(self, apex):
        with open(os.path.join(self.rospack.get_path('apex_playground'), 'config', 'environment.json')) as f:
            self.params = json.load(f)
        self.params['tracking']['ball']['lower'] = tuple(self.params['tracking']['ball']['lower'])
        self.params['tracking']['ball']['upper'] = tuple(self.params['tracking']['ball']['upper'])
        self.params['tracking']['arena']['lower'] = tuple(self.params['tracking']['arena']['lower'])
        self.params['tracking']['arena']['upper'] = tuple(self.params['tracking']['arena']['upper'])

        self.camera = CameraRecorder(apex)
        self.ball_tracking = BallTracker(self.params)
        self.ergo_tracker = ErgoTracker(apex)

        self.ball_center = None
        self.arena_center = None
        self.get_context()
        if self.ball_center is None:
            print("Could not find ball center, exiting.")
            import sys
            sys.exit(0)

    def get_context(self):
        frame = self.camera.get_image()
        img = frame.copy()

        hsv, mask_ball, mask_arena = self.ball_tracking.get_images(frame)

        min_radius_ball = self.params['tracking']['resolution'][0] * self.params['tracking']['resolution'][1] / 20000.
        ball_center, _ = self.ball_tracking.find_center('ball', frame, mask_ball, min_radius_ball)

        min_radius_arena = self.params['tracking']['resolution'][0] * self.params['tracking']['resolution'][1] / 2000.
        arena_center, arena_radius = self.ball_tracking.find_center('arena', frame, mask_arena, min_radius_arena)

        if ball_center is not None:
            self.ball_center = np.array(ball_center)
            self.extracted = True
        else:
            self.extracted = False
        if arena_center is not None:
            self.arena_center = np.array(arena_center)

        ergo_pos = self.ergo_tracker.get_position()

        return img, self.ball_center, self.arena_center, ergo_pos, self.extracted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Save images of the arena.')
    parser.add_argument('--path', metavar='-p', type=str, help='path to save images')
    parser.add_argument('--apex', metavar='-a', type=int, help='ergo number')
    parser.add_argument('--n-iter', metavar='-n', type=int, help='number of images to take')
    parser.add_argument('--save-pos', type=int, help='whether to save positional data also')
    args = parser.parse_args()

    position_extractor = PosExtractor(args.apex)
    mover = ErgoDMP(args.apex)
    mover.set_compliant(False)

    n_dmps = 6
    n_bfs = 7
    timesteps = 30
    max_params = np.array([300.] * n_bfs * n_dmps + [1.] * n_dmps)
    bounds_motors_max = np.array([180, 0, 30, 70, 20, 70])
    bounds_motors_min = np.array([-180, 0, -20, -70, 0, 0])
    dmp = MyDMP(n_dmps=n_dmps, n_bfs=n_bfs, timesteps=timesteps, max_params=max_params)

    for i in range(args.n_iter):
        point = [0, 0, 0, 0, 0, 0]
        mover.move_to(list(point), duration=1.)
        m = np.random.randn(dmp.n_dmps * dmp.n_bfs + n_dmps) * max_params
        normalized_traj = dmp.trajectory(m)
        normalized_traj = bounds_min_max(normalized_traj, n_dmps * [-1.], n_dmps * [1.])
        traj = ((normalized_traj - np.array([-1.] * n_dmps)) / 2.) * (bounds_motors_max - bounds_motors_min) + bounds_motors_min
        for m in traj:
            mover.move_to(list(m))
        image, ball_center, arena_center, ergo_position, extracted = position_extractor.get_context()
        if extracted:
            # filename = '{}-{}'.format(args.apex, datetime.datetime.now())
            filename = '{}-{}'.format(args.apex, i)
            scipy.misc.imsave(os.path.join(args.path, filename) + '.jpeg', image)
            if args.save_data:
                data = {"m": np.array(traj, dtype=np.float16),
                         "ball": np.array(ball_center, dtype=np.float16),
                         "ergo": np.array(ergo_position)}
                with open(os.path.join(args.path, filename), 'wb') as f:
                    pickle.dump(data, f)

