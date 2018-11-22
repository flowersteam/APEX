#!/usr/bin/python

import argparse
import os
import numpy as np
import scipy.misc
import pickle
import json
from tqdm import tqdm

from explauto.utils import prop_choice
from explauto.utils.config import make_configuration

from apex_playground.learning.core.learning_module import LearningModule
from apex_playground.learning.core.representation_pytorch import PoppimageVAE10, Poppimage10_B10_C25_D800, Poppimage20_B15_C30_D300, Poppimage20_B15_C50_D300

from environments import ArenaEnvironment, DummyEnvironment


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

        self.save_folder = "/home/flowers/Documents/expe_poppimage"

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
                "context": self.contexts[i],
                "outcome": self.outcomes[i],
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


class MUGLLearner(Learner):
    def __init__(self, config, environment, babbling_mode, n_modules, experiment_name, trial, n_latents,
                 n_motor_babbling=0.1, explo_noise=0.05, choice_eps=0.1, debug=False):
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
        latents_ndims = n_latents  # Number of latent variables in representation

        self.m_space = list(range(m_ndims))
        self.c_dims = list(range(m_ndims, m_ndims + latents_ndims))
        self.s_latents = list(range(m_ndims + latents_ndims, m_ndims + 2 * latents_ndims))

        # Create the learning modules:
        if self.babbling_mode == "MGEVAE10":
            self.representation = PoppimageVAE10
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
        elif self.babbling_mode == "MGEVAE20":
            self.representation = PoppimageVAE20
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
        elif self.babbling_mode == "MGEBetaVAE10":
            self.representation = Poppimage10_B10_C25_D800
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
        elif self.babbling_mode == "MGEBetaVAE20C30":
            self.representation = Poppimage20_B15_C30_D300
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
        elif self.babbling_mode == "MGEBetaVAE20C50":
            self.representation = Poppimage20_B15_C50_D300
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
        # Normalize data
        context = np.array(context) / 249.0

        if self.debug:
            # We check the reconstruction by the representation
            self.representation.act(X_pred=context)
            reconstruction = self.representation.prediction[0].transpose(1, 2, 0)
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
                self.representation.act(X_pred=context)
                context = self.representation.representation.ravel()
                self.m = self.modules[mid].produce(context=context[self.modules[mid].context_mode["context_dims"]],
                                                   explore=explore)
            return self.m

    def perceive(self, context, outcome):
        # Normalize data
        context = np.array(context) / 249.0
        outcome = np.array(context) / 249.0
        context_sensori = np.stack([context, outcome])

        if self.debug:
            # We check the reconstruction by the representation
            self.representation.act(X_pred=context_sensori)
            reconstruction = self.representation.prediction[0].transpose(1, 2, 0)
            scipy.misc.imsave('/home/flowers/Documents/tests/context_sensori' + str(self.t) + '.jpeg', reconstruction)

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
            c_img, c_ball_center, c_arena_center, c_ergo_pos, _, c_extracted = self.environment.get_current_context()

            c_img = self.preprocess_image(c_img)
            m = self.produce(c_img)

            o_img, o_ball_center, o_arena_center, o_ergo_pos, _, o_extracted = self.environment.update(m)
            o_img = self.preprocess_image(o_img)

            self.record((c_ball_center, c_arena_center, c_ergo_pos, c_extracted),
                        (o_ball_center, o_arena_center, o_ergo_pos, o_extracted))
            self.perceive(c_img, o_img)
            self.save(experiment_name=self.experiment_name, trial=self.trial, folder=self.save_folder)

    def preprocess_image(self, image):
        left = int((image.shape[0] - 128) / 2)
        top = int((image.shape[1] - 128) / 2)
        width = 128
        height = 128

        image = image[left:left + width, top:top + height]
        image = scipy.misc.imresize(image, (64, 64, 3))
        return image


class FILearner(Learner):
    def __init__(self, config, environment, babbling_mode, n_modules, experiment_name, trial, n_motor_babbling=0.1,
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

        self.n_modules = n_modules

        # Define motor and sensory spaces:
        m_ndims = self.conf.m_ndims  # number of motor parameters

        self.m_space = list(range(m_ndims))
        self.c_dims = range(m_ndims, m_ndims + 2)
        self.s_ball = range(m_ndims + 2, m_ndims + 4)
        self.s_ergo = range(m_ndims + 4, m_ndims + 7)

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
        for _ in range(n_iter):
            self.environment.reset()
            _, c_ball_center, c_arena_center, c_ergo_pos, c_ball_state, c_extracted = self.environment.get_current_context()

            m = self.produce(c_ball_state)

            _, o_ball_center, o_arena_center, o_ergo_pos, o_ball_state, o_extracted = self.environment.update(m)

            self.record((c_ball_center, c_arena_center, c_ergo_pos, c_extracted),
                        (o_ball_center, o_arena_center, o_ergo_pos, o_extracted))
            outcome = np.concatenate([o_ball_state, c_ergo_pos])
            self.perceive(c_ball_state, outcome)
            self.save(experiment_name=self.experiment_name, trial=self.trial, folder=self.save_folder)


if __name__ == "__main__":
    # Possible babbling modes: MGEVAE10, MGEVAE20, MGEBetaVAE10, MGEBetaVAE20C30, MGEBetaVAE20C50
    parser = argparse.ArgumentParser(description='Perform mugl learning.')
    parser.add_argument('--exp_name', type=str, help='Experiment name (part of path to save data)')
    parser.add_argument('--babbling', type=str, help='Babbling mode')
    parser.add_argument('--apex', metavar='-a', type=int, help='Ergo arena where to perform experience')
    parser.add_argument('--trial', type=int, help='Trial number')
    parser.add_argument('--n_iter', metavar='-n', type=int, help='Number of exploration iterations')
    parser.add_argument('--n_modules', type=int, default=5, help='Number of modules for MUGL learning')
    parser.add_argument('--debug', type=int, default=0, help='Enable debug mode or not')
    args = parser.parse_args()

    folder_trial = os.path.join("/home/flowers/Documents/expe_poppimage",
                                args.exp_name, "condition_" + args.babbling,
                                "trial_" + str(args.trial))
    if not os.path.isdir(folder_trial):
        os.makedirs(folder_trial)
    with open(os.path.join(folder_trial, 'config.json'), 'w') as f:
        json.dump(vars(args), f, separators=(',\n', ': '))

    environment = ArenaEnvironment(args.apex, debug=args.debug)

    if args.babbling == "RandomMotor":
        config = dict(m_mins=[-1.] * environment.m_ndims,
                      m_maxs=[1.] * environment.m_ndims,
                      s_mins=[-2.5] * 20,
                      s_maxs=[2.5] * 20)
        learner = FILearner(config, environment, babbling_mode=args.babbling, n_modules=args.n_modules,
                            experiment_name=args.exp_name, trial=args.trial, n_motor_babbling=1., debug=args.debug)
    elif "VAE10" in args.babbling:
        config = dict(m_mins=[-1.] * environment.m_ndims,
                      m_maxs=[1.] * environment.m_ndims,
                      s_mins=[-2.5] * 20,
                      s_maxs=[2.5] * 20)
        learner = MUGLLearner(config, environment, babbling_mode=args.babbling, n_modules=args.n_modules,
                              n_latents=10, experiment_name=args.exp_name, trial=args.trial, debug=args.debug)
    elif "VAE20" in args.babbling:
        config = dict(m_mins=[-1.] * environment.m_ndims,
                      m_maxs=[1.] * environment.m_ndims,
                      s_mins=[-2.5] * 40,
                      s_maxs=[2.5] * 40)
        learner = MUGLLearner(config, environment, babbling_mode=args.babbling, n_modules=args.n_modules,
                              n_latents=20, experiment_name=args.exp_name, trial=args.trial, debug=args.debug)
    elif "FI" in args.babbling:
        config = dict(m_mins=[-1.] * environment.m_ndims,
                      m_maxs=[1.] * environment.m_ndims,
                      s_mins=[-1] * 7,
                      s_maxs=[1] * 7)
        learner = FILearner(config, environment, babbling_mode=args.babbling, n_modules=args.n_modules,
                            experiment_name=args.exp_name, trial=args.trial, n_motor_babbling=1., debug=args.debug)
    else:
        raise NotImplementedError

    learner.explore(args.n_iter)
