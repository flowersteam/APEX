import argparse
import os
import datetime
import json
import pickle
import numpy as np

from baselines import logger

from explauto.utils import prop_choice
from explauto.utils.config import make_configuration

from apex_playground.learning.core.learning_module import LearningModule
from apex_playground.learning.core.representation_pytorch import ArmBallsVAE, ArmBallsBetaVAE


class SupervisorRep(object):
    def __init__(self, config, babbling_mode="MGEVAE", n_motor_babbling=0.1, explo_noise=0.05, choice_eps=0.1):
        self.config = config
        self.babbling_mode = babbling_mode
        self.n_motor_babbling = n_motor_babbling
        self.explo_noise = explo_noise
        self.choice_eps = choice_eps

        self.conf = make_configuration(**config)

        self.t = 0
        self.modules = {}
        self.chosen_modules = []
        self.goals = []
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
        self.have_to_replay_arm_demo = None
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
            self.representation.sorted_latents = np.array([9, 4, 0, 3, 7, 6, 2, 1, 5, 8])
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

        elif self.babbling_mode == "MGEBVAE":
            self.representation = ArmBallsBetaVAE
            self.representation.sorted_latents = np.array([9, 4, 1, 7, 6, 8, 2, 3, 0, 5])
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

    def mid_to_space(self, mid):
        return self.mid2space[mid]

    def space_to_mid(self, space):
        return self.space2mid[space]

    def get_space_names(self):
        # TODO: modify this
        if self.babbling_mode == "MGEFI":
            return ["s_ergo", "s_ball"]
        elif self.babbling_mode == "RGEFI":
            return ["s_ergoball"]

    def choose_babbling_module(self):
        interests = {}
        for mid in self.modules.keys():
            interests[mid] = self.modules[mid].interest()

        idx = prop_choice(list(interests.values()), eps=self.choice_eps)
        mid = list(interests.keys())[idx]
        self.chosen_modules.append(mid)
        return mid

    def produce(self, context, space=None):
        if np.random.random() < self.n_motor_babbling:
            self.mid_control = None
            self.chosen_modules.append("motor_babbling")
            # TODO: install rospy
            # rospy.loginfo("Random Motor Babbling")
            return self.motor_babbling()
        else:
            mid = self.choose_babbling_module()
            self.mid_control = mid
            # TODO: install rospy
            # rospy.loginfo("Chosen module: {}".format(mid))

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

    def get_last_focus(self):
        return self.mid_to_space(self.mid_control) if self.mid_control else ""

    def learning_mode(self):
        for mod in self.modules.values():
            mod.sensorimotor_model.mode = self.sm_modes[mod.mid]

    def check_bounds_dmp(self, m_ag):
        return bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)

    def get_m(self, ms):
        return ms[self.conf.m_dims]

    def get_s(self, ms):
        return ms[self.conf.s_dims]

    def motor_babbling(self):
        # TODO: check this
        self.m = self.modules["mod1"].motor_babbling()
        return self.m

    def set_ms(self, m, s):
        return np.array(list(m) + list(s))

    def update_sensorimotor_models(self, ms):
        for mid in self.modules.keys():
            m = self.modules[mid].get_m(ms)
            s = self.modules[mid].get_s(ms)
            self.modules[mid].update_sm(m, s)

    def increase_interest(self, mid):
        self.modules[mid].interest_model.current_progress = self.modules[mid].interest_model.current_progress * 1.1
        self.modules[mid].interest_model.current_interest = abs(self.modules[mid].interest_model.current_progress)

    def inverse(self, mid, s, context):
        if self.modules[mid].context_mode is not None:
            s = np.array(list(context[self.modules[mid].context_mode["context_dims"]]) + list(s))
        else:
            s = np.array(s)
        self.mid_control = None
        self.chosen_modules.append("inverse_" + mid)
        self.m = self.modules[mid].inverse(s)
        return self.m

    def dist_angle(self, a1, a2):
        return min(abs(a1 - a2), 2 - abs(a1 - a2))

    def ball_moves(self, s):
        a1 = s[16]
        a2 = s[18]
        # print("ball end angular speed", self.dist_angle(a1, a2))
        return self.dist_angle(a1, a2) > 0.1

    def perceive(self, s):
        # print("perceive len(s)", len(s), s[92:112])
        # TODO: Check if necessary
        # if self.ball_moves(s[92:112]):
        #     rospy.sleep(5)
        if not hasattr(self, "m"):
            return False

        context, sensori = np.hsplit(s, 2)
        context_sensori = np.stack([context, sensori])
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

    def get_normalized_interests_evolution(self):
        if self.babbling_mode in ["MGEVAE", "MGEBVAE"]:
            data = np.transpose(np.array([self.interests_evolution[mid] for mid in ["mod0", "mod1", "mod2", "mod3", "mod4"]]))
        if self.babbling_mode in ["RGEVAE", "RGEBVAE"]:
            data = np.transpose(np.array([self.interests_evolution[mid] for mid in ["mod0"]]))
        data_sum = data.sum(axis=1)
        data_sum[data_sum == 0.] = 1.
        return data / data_sum.reshape(data.shape[0], 1)

    def get_unnormalized_interests_evolution(self):
        if self.babbling_mode in ["MGEVAE", "MGEBVAE"]:
            data = np.transpose(np.array([self.interests_evolution[mid] for mid in ["mod0", "mod1", "mod2", "mod3", "mod4"]]))
        if self.babbling_mode in ["RGEVAE", "RGEBVAE"]:
            data = np.transpose(np.array([self.interests_evolution[mid] for mid in ["mod0"]]))
        return data

    def get_normalized_interests(self):
        interests = {}
        for mid in self.modules.keys():
            interests[mid] = self.modules[mid].interest()

        s = sum(interests.values())
        if s > 0:
            for mid in self.modules.keys():
                interests[mid] = interests[mid] / s
        return interests

    def save_iteration(self, i):
        interests = {}
        for mid in self.modules.keys():
            interests[mid] = np.float16(self.interests_evolution[mid][i])
        return {"ms": np.array(self.ms, dtype=np.float16),
                "chosen_module": self.chosen_modules[i],
                "goal": self.goals[i],
                "interests": interests}

    def forward_iteration(self, data_iteration):
        ms = data_iteration["ms"]
        m = self.get_m(ms)
        chosen_mid = data_iteration["chosen_module"]
        sg = data_iteration["goal"]
        interests = data_iteration["interests"]

        for mid in self.modules.keys():
            smid = self.modules[mid].get_s(ms)
            if self.babbling_mode == "MGEFI":
                if mid == "mod1":
                    if min(abs(smid[0] - smid[-2]), 2 - abs(smid[0] - smid[-2])) > 0.02:
                        self.modules[mid].update_sm(m, smid)
                elif mid == "mod2":
                    if min(abs(smid[1] - smid[-2]), 2 - abs(smid[1] - smid[-2])) > 0.02:
                        self.modules[mid].update_sm(m, smid)
                else:
                    self.modules[mid].update_sm(m, smid)
            if self.babbling_mode == "RGEFI":
                self.modules[mid].update_sm(m, smid)

        if sg is not None:
            self.modules[chosen_mid].s = sg
            self.modules[chosen_mid].update_im(m, self.modules[chosen_mid].get_s(ms))
        for mid in self.modules.keys():
            self.interests_evolution[mid].append(interests[mid])

        self.goals.append(sg)
        self.chosen_modules.append(chosen_mid)

        self.t += 1

    def fast_forward(self, log, forward_im=False):
        # ms_list = []
        for m, s in zip(log.logs['motor'], log.logs['sensori']):
            ms = np.append(m, s)
            self.update_sensorimotor_models(ms)
            # ms_list += [ms]
        for mid, mod in self.modules.iteritems():
            mod.fast_forward_models(log, ms_list=None, from_log_mod=mid, forward_im=forward_im)

    def eval_mode(self):
        self.sm_modes = {}
        for mod in self.modules.values():
            self.sm_modes[mod.mid] = mod.sensorimotor_model.mode
            mod.sensorimotor_model.mode = 'exploit'
