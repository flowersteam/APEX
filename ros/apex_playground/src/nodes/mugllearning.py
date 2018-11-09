#!/usr/bin/python

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
from rospkg import RosPack
from os.path import join
import json
import pickle
import matplotlib.pyplot as plt

from explauto.utils import prop_choice
from explauto.utils.config import make_configuration

from apex_playground.learning.core.learning_module import LearningModule
from apex_playground.learning.core.representation_pytorch import ArmBallsVAE
from apex_playground.learning.dmp.mydmp import MyDMP

from utils import BallTracking, CameraRecorder, ErgoMover


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

        self.tracking = BallTracking(self.params)
        self.camera = CameraRecorder(apex)

        self.ergo_mover = ErgoMover(apex)
        self.ergo_mover.set_compliant(False)

        self.n_dmps = 6
        n_bfs = 7
        timesteps = 30
        max_params = np.array([300.] * n_bfs * n_dmps + [1.] * n_dmps)
        self.bounds_motors_max = np.array([180, 0, 40, 70, 20, 70])
        self.bounds_motors_min = np.array([-180, 0, 0, -70, 0, 0])
        self.dmp = MyDMP(n_dmps=n_dmps, n_bfs=n_bfs, timesteps=timesteps, max_params=max_params)

    def get_current_context(self, debug=False):
        frame = camera.get_image()
        img = frame.copy()

        hsv, mask_ball, mask_arena = self.tracking.get_images(frame)

        min_radius_ball = self.params['tracking']['resolution'][0] * self.params['tracking']['resolution'][1] / 20000.
        ball_center, _ = self.tracking.find_center('ball', frame, mask_ball, min_radius_ball)

        min_radius_arena = self.params['tracking']['resolution'][0] * self.params['tracking']['resolution'][1] / 2000.
        arena_center, arena_radius = self.tracking.find_center('arena', frame, mask_arena, min_radius_arena)

        ring_radius = int(arena_radius / self.params['tracking']['ring_divider']) if arena_radius is not None else None

        if self.debug:
            frame = self.tracking.draw_images(frame, hsv, mask_ball, mask_arena, arena_center, ring_radius)
            plt.imshow(frame)
            plt.show()

            # image = Float32MultiArray()
            # for dim in range(len(frame.shape)):
            #     image.layout.dim.append(MultiArrayDimension(size=frame.shape[dim], label=str(frame.dtype)))
            # length = reduce(int.__mul__, frame.shape)
            # image.data = list(frame.reshape(length))
            # self.image_pub.publish(image)

        return img, ball_center, arena_center

    def reset(self):
        point = [0, 0, 0, 0, 0, 0]
        self.ergo.move_to(list(point), duration=1)

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

        self.tracking = BallTracking(self.params)

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


class Learner(object):
    def __init__(self, config, babbling_mode="MGEVAE", n_motor_babbling=0.1, explo_noise=0.05, choice_eps=0.1):
        self.babbling_mode = babbling_mode
        self.n_motor_babbling = n_motor_babbling
        self.explo_noise = explo_noise
        self.choice_eps = choice_eps

        self.conf = make_configuration(**config)

        self.t = 0
        self.modules = {}
        self.chosen_modules = []
        self.goals = []
        self.contexts = []
        self.outcomes = []
        self.progresses_evolution = {}
        self.interests_evolution = {}

        # Define motor and sensory spaces:
        m_ndims = self.conf.m_ndims  # number of motor parameters
        latents_ndims = 10  # Number of latent variables in representation

        self.m_space = list(range(m_ndims))
        self.c_dims = list(range(m_ndims, m_ndims + latents_ndims))
        self.s_latents = list(range(m_ndims + latents_ndims, m_ndims + 2 * latents_ndims))

        self.s_spaces = dict(s_latents=self.s_latents)

        self.ms = None
        self.mid_control = ''
        self.measure_interest = False

        # print()
        # print("Initialize agent with spaces:")
        # print("Motor", self.m_space)
        # print("Ergo", self.s_ergo)
        # print("Ball", self.s_ball)

        # Create the learning modules:
        if self.babbling_mode == "MGEVAE":
            self.representation = ArmBallsVAE
            self.representation.sorted_latents = np.array(range(10))
            # Create one module per two latents
            n_modules = 5
            for i in range(n_modules):
                module_id = "mod" + str(i)
                c_mod = self.representation.sorted_latents[
                        i * self.representation.n_latents // n_modules:(i + 1) * self.representation.n_latents // n_modules]
                s_mod = self.representation.sorted_latents[
                        i * self.representation.n_latents // n_modules:(i + 1) * self.representation.n_latents // n_modules] + m_ndims + self.representation.n_latents
                module = LearningModule(module_id, self.m_space, list(c_mod + m_ndims) + list(s_mod), self.conf,
                                        interest_model='normal',
                                        context_mode=dict(mode='mcs',
                                                          context_n_dims=self.representation.n_latents // n_modules,
                                                          context_dims=list(c_mod),
                                                          context_sensory_bounds=[
                                                              [-2.5] * (self.representation.n_latents // n_modules),
                                                              [2.5] * (self.representation.n_latents // n_modules)]),
                                        explo_noise=self.explo_noise)
                self.modules[module_id] = module

        for mid in self.modules.keys():
            self.progresses_evolution[mid] = []
            self.interests_evolution[mid] = []

    def motor_babbling(self):
        # TODO: check this
        self.m = self.modules["mod1"].motor_babbling()
        return self.m

    def choose_babbling_module(self):
        interests = {}
        for mid in self.modules.keys():
            interests[mid] = self.modules[mid].interest()

        idx = prop_choice(list(interests.values()), eps=self.choice_eps)
        mid = list(interests.keys())[idx]
        self.chosen_modules.append(mid)
        return mid

    def produce(self, context):
        if np.random.random() < self.n_motor_babbling:
            self.mid_control = None
            self.chosen_modules.append("motor_babbling")
            return self.motor_babbling()
        else:
            mid = self.choose_babbling_module()

            explore = True
            self.measure_interest = False
            # print("babbling_mode", self.babbling_mode)
            # print("interest", mid, self.modules[mid].interest())
            if self.modules[mid].interest() == 0.:
                # print("interest 0: exploit")
                # In condition AMB, in 20% of iterations we do not explore but measure interest
                explore = False
                self.measure_interest = True
            if np.random.random() < 0.2:
                # print("random chosen to exploit")
                # In condition AMB, in 20% of iterations we do not explore but measure interest
                explore = False
                self.measure_interest = True

            if self.modules[mid].context_mode is None:
                self.m = self.modules[mid].produce(explore=explore)
            else:
                self.representation.act(X_pred=np.array(context))
                context = self.representation.representation.ravel()
                self.m = self.modules[mid].produce(context=context[self.modules[mid].context_mode["context_dims"]],
                                                   explore=explore)
            return self.m

    def perceive(self, context, outcome):
        # print("perceive len(s)", len(s), s[92:112])
        # TODO: Check if necessary
        # if self.ball_moves(s[92:112]):
        #     rospy.sleep(5)

        context_sensori = np.stack([context, outcome])
        self.representation.act(X_pred=context_sensori)
        context_sensori_latents = self.representation.representation.ravel()

        ms = self.set_ms(self.m, context_sensori_latents)
        self.ms = ms
        self.update_sensorimotor_models(ms)
        if self.mid_control is not None and self.measure_interest:
            self.modules[self.mid_control].update_im(self.modules[self.mid_control].get_m(ms),
                                                     self.modules[self.mid_control].get_s(ms))
        if self.mid_control is not None and self.measure_interest and self.modules[self.mid_control].t >= \
                self.modules[self.mid_control].motor_babbling_n_iter:
            self.goals.append(self.modules[self.mid_control].s)
        else:
            self.goals.append(None)
        self.t = self.t + 1

        for mid in self.modules.keys():
            self.progresses_evolution[mid].append(self.modules[mid].progress())
            self.interests_evolution[mid].append(self.modules[mid].interest())

        return True

    def set_ms(self, m, s):
        return np.array(list(m) + list(s))

    def update_sensorimotor_models(self, ms):
        for mid in self.modules.keys():
            m = self.modules[mid].get_m(ms)
            s = self.modules[mid].get_s(ms)
            self.modules[mid].update_sm(m, s)

    def save_iteration(self, i):
        interests = {}
        for mid in self.modules.keys():
            interests[mid] = np.float16(self.interests_evolution[mid][i])
        return {"ms": np.array(self.ms, dtype=np.float16),
                "chosen_module": self.chosen_modules[i],
                "goal": self.goals[i],
                "interests": interests}

    def save(self, experiment_name, task, trial, folder="/media/usb/"):
        folder_trial = os.path.join(folder, experiment_name, "task_" + str(task),
                                    "condition_" + str(self.babbling_mode), "trial_" + str(trial))
        if not os.path.isdir(folder_trial):
            os.makedirs(folder_trial)
        iteration = self.t - 1
        filename = "iteration_" + str(iteration) + ".pickle"
        with open(os.path.join(folder_trial, filename), 'wb') as f:
            pickle.dump(self.save_iteration(iteration), f)

        # Check saved file
        try:
            with open(os.path.join(folder_trial, filename), 'r') as f:
                saved_data = pickle.load(f)
            return (len(saved_data["ms"]) == 204) and (saved_data["goal"] is None or len(saved_data["goal"]) == len(
                self.modules[saved_data["chosen_module"]].s_space))
        except:
            return False

    def record(self, context, outcome):
        self.contexts.append(context)
        self.outcomes.append(outcome)


class Exploration(object):
    def __init__(self, learner, environment):
        # TODO
        self.environment = environment
        self.learner = learner

    def explore(self, n_iter):
        for _ in range(n_iter):
            self.environment.reset()
            context_img, context_ball_center, context_arena_center = self.environment.get_current_context()

            left = int((context_img.shape[0] - 128) / 2)
            top = int((context_img.shape[1] - 128) / 2)
            width = 128
            height = 128
            context_img = context_img[left:left + width, top:top + height]
            context_img = scipy.misc.imresize(context_img, (64, 64, 3))

            m = self.learner.produce(context_img)
            outcome_img, outcome_ball_center, outcome_arena_center = self.environment.update(m)

            left = int((outcome_img.shape[0] - 128) / 2)
            top = int((outcome_img.shape[1] - 128) / 2)
            width = 128
            height = 128
            outcome_img = outcome_img[left:left + width, top:top + height]
            outcome_img = scipy.misc.imresize(outcome_img, (64, 64, 3))

            self.learner.record((context_ball_center, context_arena_center),
                                (outcome_ball_center, outcome_arena_center))
            self.learner.perceive(context_img, outcome_img)
            self.learner.save(experiment_name="test", task="mge_fi", trial=0, folder="../../../../../data/test")


if __name__ == "__main__":
    config = dict(m_mins=[-1.] * 6,
                  m_maxs=[1.] * 6,
                  s_mins=[-2.5] * 20,
                  s_maxs=[2.5] * 20)
    learner = Learner(config)
    environment = ArenaEnvironment(1)
    exploration = Exploration(learner, environment)
    exploration.explore(1)





    parser = argparse.ArgumentParser(description='Save images of the arena.')
    parser.add_argument('--path', metavar='-p', type=str, help='path to save images')
    parser.add_argument('--apex', metavar='-a', type=int, help='ergo number')
    parser.add_argument('--n-iter', metavar='-n', type=int, help='number of images to take')
    args = parser.parse_args()
    camera = CameraRecorder(args.apex)
    mover = ErgoMover(args.apex)
    mover.set_compliant(False)

    n_dmps = 6
    n_bfs = 7
    timesteps = 30
    max_params = np.array([300.] * n_bfs * n_dmps + [1.] * n_dmps)
    bounds_motors_max = np.array([180, 0, 40, 70, 20, 70])
    bounds_motors_min = np.array([-180, 0, 0, -70, 0, 0])
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
