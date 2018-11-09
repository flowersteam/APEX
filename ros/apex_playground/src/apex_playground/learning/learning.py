
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
import datetime

from core.supervisor import Supervisor
from core.flat_goal_babbling import FGB




class Learning(object):
    def __init__(self, config, condition="AMB", n_motor_babbling=0.1, explo_noise=0.05, choice_eps=0.2, enable_hand=True, normalize_interests=True):
        self.config = config
        if not condition in ["AMB", "AMBPROP", "AMBMIX", "RMB", "RmB", "FC", "OS"]:
            raise NotImplementedError
        self.condition = condition
        self.n_motor_babbling = n_motor_babbling
        self.explo_noise = explo_noise
        self.choice_eps = choice_eps
        self.enable_hand = enable_hand
        self.normalize_interests = normalize_interests
        self.agent = None
        
    def produce(self, context, space=None, goal=None):
        # context is the rotation of the ergo and the ball: "context = environment.get_current_context()"
        if space is None:
            if goal is None:
                # Autonomous step
                return self.agent.produce(context)
            else:
                assert goal in ["hand_up", "hand_forward", "hand_right", "hand_left", 
                                "joystick_1_forward", "joystick_1_right", "joystick_1_left",
                                "joystick_2_forward", "joystick_2_right", "joystick_2_left",
                                "ergo_right", "ergo_left",
                                "ball_right", "ball_left",
                                "light", "sound"]
                return self.agent.produce_goal(context, goal=goal)
        else:
            # Force space
            assert space in ["s_hand", "s_joystick_1", "s_joystick_2", 's_ergo', "s_ball", "s_light", "s_sound"]
            return self.agent.produce(context, space=space)
            
    def perceive(self, s, m_demo=None, j_demo=False):
        if m_demo is not None:
            assert len(m_demo) == 32, len(m_demo)
            assert len(s) == 312, len(s)
            # Demonstration of a torso arm trajectory converted to weights with "m_demo = environment.torsodemo2m(m_traj)"
            return self.agent.perceive(s, m_demo=m_demo)
        elif j_demo:
            assert len(s) == 312, len(s)
            return self.agent.perceive(list(s[:2]) + list(s[32:]), j_demo=True)
        else:
            # Perception of environment when m was produced
            assert len(s) == 312, len(s)
            return self.agent.perceive(s)
            
    def get_iterations(self): return self.agent.t
    def get_normalized_interests(self): return self.agent.get_normalized_interests()    
    def get_normalized_interests_evolution(self): return self.agent.get_normalized_interests_evolution()
    def get_last_focus(self): return self.agent.get_last_focus()
    def get_space_names(self): return self.agent.get_space_names()
    def motor_babbling(self): return self.agent.motor_babbling()
    
    def get_data_from_file(self, file_path):
        with open(file_path, 'r') as f:
            data = pickle.load(f)
        return data
                
    def save(self, experiment_name, task, trial, folder="/home/pi/apex_results/"):
        if self.agent is not None:
            folder_trial = os.path.join(folder, experiment_name, "task_" + str(task), "condition_" + str(self.condition), "trial_" + str(trial))
            if not os.path.isdir(folder_trial):
                os.makedirs(folder_trial)
            iteration = self.get_iterations() - 1
            filename = "iteration_" + str(iteration) + ".pickle"
            with open(os.path.join(folder_trial, filename), 'w') as f:
                pickle.dump(self.agent.save_iteration(iteration), f)
                
            # Check saved file
            try:
                with open(os.path.join(folder_trial, filename), 'r') as f:
                    saved_data = pickle.load(f)
                return (len(saved_data["ms"]) == 204) and (saved_data["goal"] is None or len(saved_data["goal"]) == len(self.agent.modules[saved_data["chosen_module"]].s_space))
            except:
                return False
        else:
            return False
            
    
    def start(self):
        if self.condition == "AMB":
            self.agent = Supervisor(self.config, 
                                    babbling_mode="active",
                                    n_motor_babbling=self.n_motor_babbling, 
                                    explo_noise=self.explo_noise, 
                                    choice_eps=self.choice_eps,
                                    normalize_interests=self.normalize_interests)
        elif self.condition == "AMBPROP":
            self.agent = Supervisor(self.config, 
                                    babbling_mode="prop",
                                    n_motor_babbling=self.n_motor_babbling, 
                                    explo_noise=self.explo_noise, 
                                    choice_eps=self.choice_eps,
                                    normalize_interests=self.normalize_interests)
        elif self.condition == "AMBMIX":
            self.agent = Supervisor(self.config, 
                                    babbling_mode="activemix",
                                    n_motor_babbling=self.n_motor_babbling, 
                                    explo_noise=self.explo_noise, 
                                    choice_eps=self.choice_eps,
                                    normalize_interests=self.normalize_interests)
        elif self.condition == "RMB":
            self.agent = Supervisor(self.config, 
                                    babbling_mode="random",
                                    n_motor_babbling=self.n_motor_babbling, 
                                    explo_noise=self.explo_noise, 
                                    choice_eps=self.choice_eps,
                                    normalize_interests=self.normalize_interests)
        elif self.condition == "RmB":
            self.agent = Supervisor(self.config, 
                                    n_motor_babbling=1.)
        elif self.condition == "FGB":
            self.agent = FGB(self.config,
                                n_motor_babbling=self.n_motor_babbling, 
                                explo_noise=self.explo_noise)
        elif self.condition == "FC":
            self.agent = Supervisor(self.config, 
                                    babbling_mode="FC",
                                    n_motor_babbling=self.n_motor_babbling, 
                                    explo_noise=self.explo_noise, 
                                    choice_eps=self.choice_eps,
                                    normalize_interests=self.normalize_interests)
        elif self.condition == "OS":
            self.agent = Supervisor(self.config, 
                                    babbling_mode="OS",
                                    n_motor_babbling=self.n_motor_babbling, 
                                    explo_noise=self.explo_noise, 
                                    choice_eps=self.choice_eps,
                                    normalize_interests=self.normalize_interests)
        else:
            raise NotImplementedError
    
    def restart_from_files(self, experiment_name, task, trial, iteration, folder="/home/pi/apex_results/"):
        self.start()
        folder_trial = os.path.join(folder, experiment_name, "task_" + str(task), "condition_" + str(self.condition), "trial_" + str(trial))
        for it in range(iteration):
            filename = "iteration_" + str(it) + ".pickle"
            data = self.get_data_from_file(os.path.join(folder_trial, filename))
            self.agent.forward_iteration(data)

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.get_normalized_interests_evolution(), lw=2)
        ax.legend(["Hand", "Joystick_1", "Joystick_2", "Ergo", "Ball", "Light", "Sound"], ncol=3)
        ax.set_xlabel('Time steps', fontsize=20)
        ax.set_ylabel('Learning progress', fontsize=20)
        plt.show(block=True)
        
