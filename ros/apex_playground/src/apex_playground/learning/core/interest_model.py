
import numpy as np

from explauto.interest_model.random import RandomInterest

from dataset import Dataset


class MiscRandomInterest(RandomInterest):
    def __init__(self, conf, expl_dims, n_sdims, win_size=100):
        
        RandomInterest.__init__(self, conf, expl_dims)
        
        self.win_size = win_size
        self.n_sdims = n_sdims
        
        self.current_progress = 0.
        self.current_interest = 0.
        self.data = Dataset(len(expl_dims),
                            len(expl_dims),
                            max_size=1000)
    
    def competence_dist(self, target, reached):
        return - np.linalg.norm(target - reached) / self.n_sdims
        
    def update(self, sg, s, log=False):
        if len(self.data) > 0:
            # Current competence: from distance between current goal and reached s 
            c = self.competence_dist(sg, s)
            # Get NN of goal sg
            idx_sg_NN = self.data.nn_y(sg)[1][0]
            # Get corresponding reached sensory state s
            s_NN = self.data.get_x(idx_sg_NN)
            # old competence! from distance between current goal and old reached s for NN goal sg_NN 
            c_old = self.competence_dist(sg, s_NN)
            # Progress is the difference between current and old competence
            progress = c - c_old
            if log:
                print("sg", sg)
                print("s", s)
                print("s_NN", s_NN)
                print("c", c)
                print("c_old", c_old)
                print("progress", progress)
                print("data", self.data.data)
        else:
            progress = 0.
        # Update progress and interest
        self.current_progress += (1. / self.win_size) * (progress - self.current_progress)
        self.current_interest = abs(self.current_progress)
        # Log reached point and goal
        self.data.add_xy(s, sg)
        return progress
    
        

class ContextRandomInterest(MiscRandomInterest):
    def __init__(self, 
                 conf, 
                 expl_dims,
                 n_sdims,
                 win_size,
                 context_mode):
        
        self.context_mode = context_mode
        
        MiscRandomInterest.__init__(self,
                                    conf, 
                                    expl_dims,
                                    n_sdims,
                                    win_size)        

              
    def competence_dist(self, csg, cs):
        s = cs[self.context_mode["context_n_dims"]:]
        sg = csg[self.context_mode["context_n_dims"]:]
        return MiscRandomInterest.competence_dist(self, sg, s)
        
