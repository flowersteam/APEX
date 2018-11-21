#!/usr/bin/python

import os
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc
import pickle
from tqdm import tqdm

from explauto.utils import prop_choice
from explauto.utils.config import make_configuration

from apex_playground.learning.core.learning_module import LearningModule
from apex_playground.learning.core.representation_pytorch import ArmBallsVAE, ArmBallsBetaVAE

from apex_playground.learning.environment_explauto.armballs import TestArmBallsEnv, TestArmBallsObsEnv


class Learner(object):
    def __init__(self):
        self.t = 0
        self.modules = {}
        self.chosen_modules = []
        self.goals = []
        self.contexts = []
        self.outcomes = []
        self.progresses_evolution = {}
        self.interests_evolution = {}

        self.babbling_mode = None
        self.choice_eps = None

        self.ms = None
        self.mid_control = None
        self.measure_interest = False

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

    def set_ms(self, m, s):
        return np.array(list(m) + list(s))

    def update_sensorimotor_models(self, ms):
        for mid in self.modules.keys():
            m = self.modules[mid].get_m(ms)
            s = self.modules[mid].get_s(ms)
            self.modules[mid].update_sm(m, s)

    def save_iteration(self, i):
        interests = {}
        progresses = {}
        for mid in self.modules.keys():
            interests[mid] = np.float16(self.interests_evolution[mid][i])
            progresses[mid] = np.float16(self.progresses_evolution[mid][i])
        return {"ms": np.array(self.ms, dtype=np.float16),
                "chosen_module": self.chosen_modules[i],
                "goal": self.goals[i],
                "interests": interests,
                "progresses": progresses}

    def save(self, experiment_name, trial, folder):
        folder_trial = os.path.join(folder, experiment_name, "condition_" + str(self.babbling_mode),
                                    "trial_" + str(trial))
        if not os.path.isdir(folder_trial):
            os.makedirs(folder_trial)
        iteration = self.t - 1
        filename = "iteration_" + str(iteration) + ".pickle"
        with open(os.path.join(folder_trial, filename), 'wb') as f:
            pickle.dump(self.save_iteration(iteration), f)

    def record(self, context, outcome):
        self.contexts.append(context)
        self.outcomes.append(outcome)

    def plot_interest_evolution(self):
        fig, ax = plt.subplots()
        data = np.transpose(
            np.array([self.interests_evolution[mid] for mid in ["mod0", "mod1", "mod2", "mod3", "mod4"]]))
        ax.plot(data, lw=2)
        if "VAE" in self.babbling_mode:
            ax.legend(["mod0", "mod1", "mod2", "mod3", "mod4"], ncol=3)
        else:
            ax.legend(["Arm", "Ball"], ncol=3)
        ax.set_xlabel('Time steps', fontsize=20)
        ax.set_ylabel('Interest', fontsize=20)
        plt.show(block=True)

    def plot_progress_evolution(self):
        fig, ax = plt.subplots()
        data = np.transpose(
            np.array([self.progresses_evolution[mid] for mid in ["mod0", "mod1", "mod2", "mod3", "mod4"]]))
        ax.plot(data, lw=2)
        if "VAE" in self.babbling_mode:
            ax.legend(["mod0", "mod1", "mod2", "mod3", "mod4"], ncol=3)
        else:
            ax.legend(["Arm", "Ball"], ncol=3)
        ax.set_xlabel('Time steps', fontsize=20)
        ax.set_ylabel('Progress', fontsize=20)
        plt.show(block=True)


class MUGLLearner(Learner):
    def __init__(self, config, environment, babbling_mode, n_modules, experiment_name, trial, n_motor_babbling=0.1,
                 explo_noise=0.05, choice_eps=0.1, debug=False):
        super(MUGLLearner, self).__init__()
        self.debug = debug

        self.experiment_name = experiment_name
        self.trial = trial

        self.environment = environment
        self.babbling_mode = babbling_mode
        self.n_motor_babbling = n_motor_babbling
        self.explo_noise = explo_noise
        self.choice_eps = choice_eps

        self.conf = make_configuration(**config)

        self.n_modules = n_modules

        # Define motor and sensory spaces:
        m_ndims = self.conf.m_ndims  # number of motor parameters
        latents_ndims = 10  # Number of latent variables in representation

        self.m_space = list(range(m_ndims))
        self.c_dims = list(range(m_ndims, m_ndims + latents_ndims))
        self.s_latents = list(range(m_ndims + latents_ndims, m_ndims + 2 * latents_ndims))

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
            self.representation.sorted_latents = np.array([9, 4, 1, 7, 6, 8, 2, 3, 0, 5])
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

        context = np.array(context)
        outcome = np.array(outcome)

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

    def explore(self, n_iter):
        for _ in tqdm(range(n_iter)):
            self.environment.reset()
            context_img = self.environment.get_current_context()

            m = self.produce(context_img)

            self.environment.update(m, reset=False)
            outcome_img = self.environment.get_current_context()

            self.perceive(context_img, outcome_img)
            self.save(experiment_name=self.experiment_name, trial=self.trial,
                      folder="/Users/adrien/Documents/post-doc/expe_poppy/data")


class FILearner(Learner):
    def __init__(self, config, environment, babbling_mode, experiment_name, trial, n_motor_babbling=0.1,
                 explo_noise=0.05, choice_eps=0.1, debug=False):
        super(FILearner, self).__init__()
        self.debug = debug

        self.experiment_name = experiment_name
        self.trial = trial

        self.environment = environment
        self.babbling_mode = babbling_mode
        self.n_motor_babbling = n_motor_babbling
        self.explo_noise = explo_noise
        self.choice_eps = choice_eps

        self.conf = make_configuration(**config)

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
        if self.babbling_mode == "FlatFI" or self.babbling_mode == "RandomMotor":
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
            self.modules["mod2"] = LearningModule("mod2", self.m_space, self.c_dims + self.ball, self.conf,
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

    def explore(self, n_iter):
        for _ in tqdm(range(n_iter)):
            self.environment.reset()
            _, context_ball_center, context_arena_center, context_ergo_pos, context_extracted = self.environment.get_current_context()

            m = self.produce(context_ball_center)

            _, outcome_ball_center, outcome_arena_center, outcome_ergo_pos, outcome_extracted = self.environment.update(m)

            self.record((context_ball_center, context_arena_center, context_ergo_pos, context_extracted),
                        (outcome_ball_center, outcome_arena_center, outcome_ergo_pos, outcome_extracted))
            outcome = np.concatenate([outcome_ball_center, context_ergo_pos])
            self.perceive(context_ball_center, outcome)
            self.save(experiment_name=self.experiment_name, trial=self.trial,
                      folder="/Users/adrien/Documents/post-doc/expe_poppy/data")


if __name__ == "__main__":
    # np.random.seed(0)

    render = False
    print("Create environment")
    environment = TestArmBallsObsEnv(render=render)

    config = dict(m_mins=environment.conf.m_mins,
                  m_maxs=environment.conf.m_maxs,
                  s_mins=[-2.5] * 20,
                  s_maxs=[2.5] * 20)
    learner = MUGLLearner(config, environment, babbling_mode="MGEBetaVAE", n_modules=5, explo_noise=0.01,
                          choice_eps=0.1, experiment_name="test", trial=0, debug=False)

    learner.explore(10000)
    learner.plot_interest_evolution()
    # learner.plot_progress_evolution()
