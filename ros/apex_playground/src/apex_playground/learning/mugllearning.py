import os
import pickle
import matplotlib.pyplot as plt

from core.mge_fi import SupervisorFI
from core.mge_rep import SupervisorRep


class Learning(object):
    def __init__(self, config, condition="MGEFI", n_motor_babbling=0.1, explo_noise=0.05, choice_eps=0.2):
        self.config = config
        if not condition in ["MGEFI", "RGEFI", "MGEVAE", "RGEVAE", "MGEBVAE", "RGEBVAE", "RmB"]:
            raise NotImplementedError
        self.condition = condition
        self.n_motor_babbling = n_motor_babbling
        self.explo_noise = explo_noise
        self.choice_eps = choice_eps
        self.agent = None
        
    def produce(self, context):
        # context is the image of the scene before action: "context = environment.get_current_context()"
        return self.agent.produce(context)
            
    def perceive(self, s):
        # Perception of environment when m was produced
        return self.agent.perceive(s)
            
    def get_iterations(self): return self.agent.t

    def get_normalized_interests(self): return self.agent.get_normalized_interests()

    def get_normalized_interests_evolution(self): return self.agent.get_normalized_interests_evolution()

    def get_unnormalized_interests_evolution(self):
        return self.agent.get_unnormalized_interests_evolution()

    def get_last_focus(self): return self.agent.get_last_focus()

    def get_space_names(self): return self.agent.get_space_names()

    def motor_babbling(self): return self.agent.motor_babbling()
    
    def get_data_from_file(self, file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        return data
                
    def save(self, experiment_name, trial, folder="/media/usb/"):
        if self.agent is not None:
            folder_trial = os.path.join(folder, experiment_name, "condition_" + str(self.condition), "trial_" + str(trial))
            if not os.path.isdir(folder_trial):
                os.makedirs(folder_trial)
            iteration = self.get_iterations() - 1
            filename = "iteration_" + str(iteration) + ".pickle"
            with open(os.path.join(folder_trial, filename), 'wb') as f:
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
        if self.condition in ["MGEFI", "RGEFI"]:
            self.agent = SupervisorFI(self.config,
                                      babbling_mode=self.condition,
                                      n_motor_babbling=self.n_motor_babbling,
                                      explo_noise=self.explo_noise,
                                      choice_eps=self.choice_eps)
        elif self.condition in ["MGEVAE", "RGEVAE", "MGEBVAE", "RGEBVAE"]:
            self.agent = SupervisorRep(self.config,
                                       babbling_mode=self.condition,
                                       n_motor_babbling=self.n_motor_babbling,
                                       explo_noise=self.explo_noise,
                                       choice_eps=self.choice_eps)
        elif self.condition == "RmB":
            self.agent = SupervisorFI(self.config,
                                      n_motor_babbling=1.)
        else:
            raise NotImplementedError
    
    def restart_from_files(self, experiment_name, trial, iteration, folder="/media/usb/"):
        self.start()
        folder_trial = os.path.join(folder, experiment_name, "condition_" + str(self.condition), "trial_" + str(trial))
        for it in range(iteration):
            filename = "iteration_" + str(it) + ".pickle"
            data = self.get_data_from_file(os.path.join(folder_trial, filename))
            self.agent.forward_iteration(data)

    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.get_normalized_interests_evolution(), lw=2)
        if self.condition in ["MGEFI", "RGEFI"]:
            ax.legend(["Arm", "Distractor", "Ball"], ncol=3)
        if self.condition in ["MGEVAE", "MGEBVAE"]:
            ax.legend(["mod0", "mod1", "mod2", "mod3", "mod4"], ncol=3)
        ax.set_xlabel('Time steps', fontsize=20)
        ax.set_ylabel('Interest', fontsize=20)
        plt.show(block=True)
