import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

import scipy.misc
import numpy as np

from explauto.utils import bounds_min_max
from rospkg import RosPack
from os.path import join
import json
from apex_playground.learning.dmp.mydmp import MyDMP

from utils import BallTracker, CameraRecorder, ErgoMover, ErgoTracker


class ArenaEnvironment(object):
    def __init__(self, apex, debug=False):
        self.debug = debug
        self.rospack = RosPack()
        with open(join(self.rospack.get_path('apex_playground'), 'config', 'environment.json')) as f:
            self.params = json.load(f)
        self.params['tracking']['ball']['lower'] = tuple(self.params['tracking']['ball']['lower'])
        self.params['tracking']['ball']['upper'] = tuple(self.params['tracking']['ball']['upper'])
        self.params['tracking']['arena']['lower'] = tuple(self.params['tracking']['arena']['lower'])
        self.params['tracking']['arena']['upper'] = tuple(self.params['tracking']['arena']['upper'])

        self.ball_tracking = BallTracker(self.params)
        self.ergo_tracker = ErgoTracker(apex)
        self.camera = CameraRecorder(apex)

        self.ergo_mover = ErgoMover(apex)
        self.ergo_mover.set_compliant(False)

        self.n_dmps = 6
        self.n_bfs = 7
        self.m_ndims = self.n_bfs * self.n_dmps + self.n_dmps
        timesteps = 50
        max_params = np.array([300.] * self.n_bfs * self.n_dmps + [1.] * self.n_dmps)
        self.bounds_motors_max = np.array([180, 0, 30, 70, 20, 70])
        self.bounds_motors_min = np.array([-180, -40, -50, -70, -50, -50])
        self.dmp = MyDMP(n_dmps=self.n_dmps, n_bfs=self.n_bfs, timesteps=timesteps, max_params=max_params)

        self.ball_center = None
        self.arena_center = None
        self.get_ball_position()
        if not self.ball_center:
            print("Could not find ball center, exiting.")
            import sys
            sys.exit(0)


    def get_ball_position(self):
        frame = self.camera.get_image()
        img = frame.copy()

        hsv, mask_ball, mask_arena = self.ball_tracking.get_images(frame)

        min_radius_ball = self.params['tracking']['resolution'][0] * self.params['tracking']['resolution'][1] / 20000.
        ball_center, _ = self.ball_tracking.find_center('ball', frame, mask_ball, min_radius_ball)

        min_radius_arena = self.params['tracking']['resolution'][0] * self.params['tracking']['resolution'][1] / 2000.
        arena_center, arena_radius = self.ball_tracking.find_center('arena', frame, mask_arena, min_radius_arena)

        ring_radius = int(arena_radius / self.params['tracking']['ring_divider']) if arena_radius is not None else None

        if self.debug:
            frame = self.ball_tracking.draw_images(frame, hsv, mask_ball, mask_arena, arena_center, ring_radius)
            scipy.misc.imsave('/home/flowers/Documents/tests/frame.jpeg', frame)
            scipy.misc.imsave('/home/flowers/Documents/tests/img.jpeg', img)
            plt.imshow(frame)
            plt.show()
            import time
            time.sleep(1)

            # image = Float32MultiArray()
            # for dim in range(len(frame.shape)):
            #     image.layout.dim.append(MultiArrayDimension(size=frame.shape[dim], label=str(frame.dtype)))
            # length = reduce(int.__mul__, frame.shape)
            # image.data = list(frame.reshape(length))
            # self.image_pub.publish(image)

        if ball_center is not None:
            self.ball_center = np.array(ball_center)
        if arena_center is not None:
            self.arena_center = np.array(arena_center)

        return img

    def get_current_context(self):
        img = self.get_ball_position()
        ergo_pos = self.ergo_tracker.get_position()

        return img, self.ball_center, self.arena_center, ergo_pos

    def reset(self):
        point = [0, 0, 0, 0, 0, 0]
        self.ergo_mover.move_to(list(point), duration=3)

    def update(self, m):
        normalized_traj = self.dmp.trajectory(m)
        normalized_traj = bounds_min_max(normalized_traj, self.n_dmps * [-1.], self.n_dmps * [1.])
        traj = ((normalized_traj - np.array([-1.] * self.n_dmps)) / 2.) * (
                    self.bounds_motors_max - self.bounds_motors_min) + self.bounds_motors_min
        for m in traj:
            self.ergo_mover.move_to(list(m))
        return self.get_current_context()


class DummyEnvironment(object):
    def __init__(self):
        self.params = {"tracking": {
                        "resolution": [176, 144],
                        "ball": {
                          "lower": [27, 45, 70],
                          "upper": [38, 255, 255]
                        },
                        "arena": {
                          "lower": [95, 40, 25],
                          "upper": [130, 255, 255],
                          "radius": 0.225
                        },
                        "buffer_size": 32,
                        "ring_divider": 1.4 * 2
                      },
                      "rate": 15,
                      "sound": {
                        "freq": [100, 2000]
                      }
                    }
        self.params['tracking']['ball']['lower'] = tuple(self.params['tracking']['ball']['lower'])
        self.params['tracking']['ball']['upper'] = tuple(self.params['tracking']['ball']['upper'])
        self.params['tracking']['arena']['lower'] = tuple(self.params['tracking']['arena']['lower'])
        self.params['tracking']['arena']['upper'] = tuple(self.params['tracking']['arena']['upper'])

        self.tracking = BallTracker(self.params)

    def get_current_context(self, debug=False):
        frame = scipy.misc.imread('/Users/adrien/Documents/post-doc/expe_poppy/imgs/8.jpeg')
        img = frame.copy().reshape(144, 176, 3)

        hsv, mask_ball, mask_arena = self.tracking.get_images(frame)

        min_radius_ball = self.params['tracking']['resolution'][0] * self.params['tracking']['resolution'][1] / 20000.
        ball_center, _ = self.tracking.find_center('ball', frame, mask_ball, min_radius_ball)

        min_radius_arena = self.params['tracking']['resolution'][0] * self.params['tracking']['resolution'][1] / 2000.
        arena_center, arena_radius = self.tracking.find_center('arena', frame, mask_arena, min_radius_arena)
        ring_radius = int(arena_radius / self.params['tracking']['ring_divider']) if arena_radius is not None else None

        if ball_center is not None and arena_center is not None:
            elongation, theta = self.tracking.get_state(ball_center, arena_center)
        else:
            elongation, theta = None, None

        if debug:
            frame = self.tracking.draw_images(frame, hsv, mask_ball, mask_arena, arena_center, ring_radius)
            image = Float32MultiArray()
            for dim in range(len(frame.shape)):
                image.layout.dim.append(MultiArrayDimension(size=frame.shape[dim], label=str(frame.dtype)))
            length = reduce(int.__mul__, frame.shape)
            image.data = list(frame.reshape(length))
            self.image_pub.publish(image)

        return img, elongation, theta

    def reset(self):
        pass

    def update(self, m):
        return self.get_current_context()
