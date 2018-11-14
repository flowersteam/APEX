#!/usr/bin/python

import argparse
import os
import numpy as np
import scipy.misc
import pickle

from explauto.utils import prop_choice
from explauto.utils.config import make_configuration

from apex_playground.learning.core.learning_module import LearningModule
from apex_playground.learning.core.representation_pytorch import ArmBallsVAE, ArmBallsBetaVAE

from environments import ArenaEnvironment, DummyEnvironment


class MUGLLearner(object):
    def __init__(self, config, environment, babbling_mode, n_modules, experiment_name, trial, n_motor_babbling=0.1,
                 explo_noise=0.05, choice_eps=0.1, debug=False):
        self.debug = debug

        self.experiment_name = experiment_name
        self.trial = trial

        self.environment = environment
        self.babbling_mode = babbling_mode
        self.n_motor_babbling = n_motor_babbling
        self.explo_noise = explo_noise
        self.choice_eps = choice_eps

        self.conf = make_configuration(**config)

        self.t = 0
        self.n_modules = n_modules
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

        self.ms = None
        self.mid_control = None
        self.measure_interest = False

        # Create the learning modules:
        if self.babbling_mode == "MGEVAE":
            self.representation = ArmBallsVAE
            self.representation.sorted_latents = np.array(range(10))
            # Create one module per two latents
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
        elif self.babbling_mode == "MGEBetaVAE":
            self.representation = ArmBallsBetaVAE
            self.representation.sorted_latents = np.array(range(10))
            # Create one module per two latents
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
        else:
            raise NotImplementedError

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
        if self.debug:
            # We check the reconstruction by the reprensentation
            self.representation.act(X_pred=np.array(context))
            reconstruction = self.representation.prediction[0]
            scipy.misc.imsave('/home/flowers/Documents/tests/context' + str(self.t) + '.jpeg', context)
            scipy.misc.imsave('/home/flowers/Documents/tests/reconstruction' + str(self.t) + '.jpeg', reconstruction)
        if np.random.random() < self.n_motor_babbling:
            self.mid_control = None
            self.chosen_modules.append("motor_babbling")
            return self.motor_babbling()
        else:
            mid = self.choose_babbling_module()
            self.mid_control = mid

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
                "context": self.contexts[i],
                "outcome": self.outcomes[i],
                "interests": interests}

    def save(self, experiment_name, task, trial, folder="/media/usb/"):
        print('saving')
        folder_trial = os.path.join(folder, experiment_name, "task_" + str(task),
                                    "condition_" + str(self.babbling_mode), "trial_" + str(trial))
        if not os.path.isdir(folder_trial):
            os.makedirs(folder_trial)
        iteration = self.t - 1
        filename = "iteration_" + str(iteration) + ".pickle"
        with open(os.path.join(folder_trial, filename), 'wb') as f:
            pickle.dump(self.save_iteration(iteration), f)

    def record(self, context, outcome):
        self.contexts.append(context)
        self.outcomes.append(outcome)

    def explore(self, n_iter):
        for _ in range(n_iter):
            self.environment.reset()
            context_img, context_ball_center, context_arena_center, context_ergo_pos = self.environment.get_current_context()

            left = int((context_img.shape[0] - 128) / 2)
            top = int((context_img.shape[1] - 128) / 2)
            width = 128
            height = 128

            context_img = context_img[left:left + width, top:top + height]
            context_img = scipy.misc.imresize(context_img, (64, 64, 3))

            m = self.produce(context_img)

            outcome_img, outcome_ball_center, outcome_arena_center, outcome_ergo_pos = self.environment.update(m)
            outcome_img = outcome_img[left:left + width, top:top + height]
            outcome_img = scipy.misc.imresize(outcome_img, (64, 64, 3))

            self.record((context_ball_center, context_arena_center, context_ergo_pos),
                        (outcome_ball_center, outcome_arena_center, outcome_ergo_pos))
            self.perceive(context_img, outcome_img)
            self.save(experiment_name=self.experiment_name, task=self.babbling_mode,
                      trial=self.trial, folder="~/Documents/expe_poppimage")


class FILearner(object):
    def __init__(self, config, environment, babbling_mode, n_modules, experiment_name, trial, n_motor_babbling=0.1,
                 explo_noise=0.05, choice_eps=0.1, debug=False):
        self.debug = debug

        self.experiment_name = experiment_name
        self.trial = trial

        self.environment = environment
        self.babbling_mode = babbling_mode
        self.n_motor_babbling = n_motor_babbling
        self.explo_noise = explo_noise
        self.choice_eps = choice_eps

        self.conf = make_configuration(**config)

        self.t = 0
        self.n_modules = n_modules
        self.modules = {}
        self.chosen_modules = []
        self.goals = []
        self.contexts = []
        self.outcomes = []
        self.progresses_evolution = {}
        self.interests_evolution = {}

        # Define motor and sensory spaces:
        m_ndims = self.conf.m_ndims  # number of motor parameters

        self.m_space = list(range(m_ndims))
        self.c_dims = range(m_ndims, m_ndims + 2)
        self.s_ball = range(m_ndims + 2, m_ndims + 4)
        self.s_ergo = range(m_ndims + 4, m_ndims + 7)

        self.ms = None
        self.mid_control = None
        self.measure_interest = False

        # Create the learning modules:
        if self.babbling_mode == "FlatFI":
            self.modules["mod1"] = LearningModule("mod1", self.m_space, self.c_dims + self.s_ergo + self.s_ball,
                                                  self.conf,
                                                  context_mode=dict(mode='mcs', context_n_dims=2,
                                                                    context_sensory_bounds=[[-1., -1.],
                                                                                            [1., 1.]],
                                                                    context_dims=range(2)),
                                                  explo_noise=self.explo_noise)
        elif self.babbling_mode == "ModularFI":
            self.modules["mod1"] = LearningModule("mod1", self.m_space, self.c_dims + self.s_ergo, self.conf,
                                                  context_mode=dict(mode='mcs', context_n_dims=2,
                                                                    context_sensory_bounds=[[-1., -1.],
                                                                                            [1., 1.]],
                                                                    context_dims=range(2)),
                                                  explo_noise=self.explo_noise)
            self.modules["mod1"] = LearningModule("mod2", self.m_space, self.c_dims + self.ball, self.conf,
                                                  context_mode=dict(mode='mcs', context_n_dims=2,
                                                                    context_sensory_bounds=[[-1., -1.],
                                                                                            [1., 1.]],
                                                                    context_dims=range(2)),
                                                  explo_noise=self.explo_noise)
        else:
            raise NotImplementedError

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
            self.mid_control = mid
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
                self.m = self.modules[mid].produce(context=context[self.modules[mid].context_mode["context_dims"]],
                                                   explore=explore)
            return self.m

    def perceive(self, context, outcome):
        # print("perceive len(s)", len(s), s[92:112])
        # TODO: Check if necessary
        # if self.ball_moves(s[92:112]):
        #     rospy.sleep(5)

        context_sensori = np.concatenate([context, outcome])

        ms = self.set_ms(self.m, context_sensori)
        self.ms = ms

        print(ms.shape)
        print(self.m_space)
        print(self.c_dims)
        print(self.s_ergo)
        print(self.s_ball)

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
                "context": self.contexts[i],
                "outcome": self.outcomes[i],
                "interests": interests}

    def save(self, experiment_name, task, trial, folder="/media/usb/"):
        print('saving')
        folder_trial = os.path.join(folder, experiment_name, "task_" + str(task),
                                    "condition_" + str(self.babbling_mode), "trial_" + str(trial))
        if not os.path.isdir(folder_trial):
            os.makedirs(folder_trial)
        iteration = self.t - 1
        filename = "iteration_" + str(iteration) + ".pickle"
        with open(os.path.join(folder_trial, filename), 'wb') as f:
            pickle.dump(self.save_iteration(iteration), f)

    def record(self, context, outcome):
        self.contexts.append(context)
        self.outcomes.append(outcome)

    def explore(self, n_iter):
        for _ in range(n_iter):
            self.environment.reset()
            _, context_ball_center, context_arena_center, context_ergo_pos = self.environment.get_current_context()

            m = self.produce(context_ball_center)

            _, outcome_ball_center, outcome_arena_center, outcome_ergo_pos = self.environment.update(m)

            self.record((context_ball_center, context_arena_center, context_ergo_pos),
                        (outcome_ball_center, outcome_arena_center, outcome_ergo_pos))
            outcome = np.concatenate([outcome_ball_center, context_ergo_pos])
            self.perceive(context_ball_center, outcome)
            self.save(experiment_name=self.experiment_name, task=self.babbling_mode,
                      trial=self.trial, folder="~/Documents/expe_poppimage")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Perform mugl learning.')
    parser.add_argument('--exp_name', type=str, help='Experiment name (part of path to save data)')
    parser.add_argument('--babbling', type=str, help='Babbling mode')
    parser.add_argument('--apex', metavar='-a', type=int, help='Ergo arena where to perform experience')
    parser.add_argument('--trial', type=int, help='Trial number')
    parser.add_argument('--n_iter', metavar='-n', type=int, help='Number of exploration iterations')
    parser.add_argument('--n_modules', type=int, default=5, help='Number of modules for MUGL learning')
    args = parser.parse_args()

    environment = ArenaEnvironment(args.apex, debug=False)

    if args.babbling == "RandomMotor":
        config = dict(m_mins=[-1.] * environment.m_ndims,
                      m_maxs=[1.] * environment.m_ndims,
                      s_mins=[-2.5] * 20,
                      s_maxs=[2.5] * 20)
        learner = FILearner(config, environment, babbling_mode=args.babbling, n_modules=args.n_modules,
                              experiment_name=args.exp_name, trial=args.trial, n_motor_babbling=1., debug=False)
    elif "VAE" in args.babbling:
        config = dict(m_mins=[-1.] * environment.m_ndims,
                      m_maxs=[1.] * environment.m_ndims,
                      s_mins=[-2.5] * 20,
                      s_maxs=[2.5] * 20)
        learner = MUGLLearner(config, environment, babbling_mode=args.babbling, n_modules=args.n_modules,
                              experiment_name=args.exp_name, trial=args.trial, debug=False)
    elif "FI" in args.babbling:
        config = dict(m_mins=[-1.] * environment.m_ndims,
                      m_maxs=[1.] * environment.m_ndims,
                      s_mins=[-1] * 7,
                      s_maxs=[1] * 7)
        learner = FILearner(config, environment, babbling_mode=args.babbling, n_modules=args.n_modules,
                            experiment_name=args.exp_name, trial=args.trial, n_motor_babbling=1., debug=False)
    else:
        raise NotImplementedError

    learner.explore(args.n_iter)

    import sys
    sys.exit(0)
