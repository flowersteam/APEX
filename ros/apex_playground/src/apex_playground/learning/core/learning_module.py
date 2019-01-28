import numpy as np

from numpy import array, hstack

from explauto.agent import Agent
from explauto.utils import rand_bounds
from explauto.utils.config import make_configuration
from explauto.exceptions import ExplautoBootstrapError

from sensorimotor_model import DemonstrableNN
from interest_model import MiscRandomInterest, ContextRandomInterest

from dataset import BufferedDataset

import rospy


class LearningModule(Agent):
    def __init__(self, mid, m_space, s_space, env_conf, explo_noise=0., normalize_interests=True, context_mode=None):


        self.conf = make_configuration(env_conf.m_mins[m_space], 
                                       env_conf.m_maxs[m_space], 
                                       array(list(env_conf.m_mins[m_space]) + list(env_conf.s_mins))[s_space],
                                       array(list(env_conf.m_maxs[m_space]) + list(env_conf.s_maxs))[s_space])
        
        self.im_dims = self.conf.s_dims
        
        self.mid = mid
        self.m_space = m_space
        self.context_mode = context_mode
        self.s_space = s_space
        self.motor_babbling_n_iter = 0
        self.n_mdims = 4
        self.n_sdims = len(s_space) // 10
        self.explo_noise = 0.2

        self.s = None
        self.last_interest = 0
        
        if context_mode is None:
            self.im = MiscRandomInterest(self.conf, 
                                         self.conf.s_dims, 
                                         self.n_sdims, 
                                         win_size=100)
        else:
            self.im = ContextRandomInterest(self.conf, 
                                            self.conf.s_dims, 
                                            self.n_sdims, 
                                            100,
                                            context_mode)
            
        
        #self.im = im_cls(self.conf, self.im_dims, **kwargs)
        
        self.sm = BufferedDataset(self.conf.m_ndims, 
                                  self.conf.s_ndims,
                                  buffer_size=10000,
                                  lateness=10)
        #sm_cls, kwargs = (DemonstrableNN, {'fwd': 'NN', 'inv': 'NN', 'sigma_explo_ratio':explo_noise})
        #self.sm = sm_cls(self.conf, **kwargs)
        
        Agent.__init__(self, self.conf, self.sm, self.im, context_mode=self.context_mode)
        

        
    def motor_babbling(self, n=1): 
        if n == 1:
            return rand_bounds(self.conf.m_bounds)[0]
        else:
            return rand_bounds(self.conf.m_bounds, n)
        
    def goal_babbling(self):
        s = rand_bounds(self.conf.s_bounds)[0]
        m = self.sm.infer(self.conf.s_dims, self.conf.m_dims, s)
        return m
            
    def get_m(self, ms): return array(ms[self.m_space])
    def get_s(self, ms): return array(ms[self.s_space])
        
    def set_one_m(self, ms, m):
        """ Set motor dimensions used by module
        """
        ms = array(ms)
        ms[self.mconf['m']] = m
        
    def set_m(self, ms, m):
        """ Set motor dimensions used by module on one ms
        """
        self.set_one_m(ms, m)
        if self.mconf['operator'] == "seq":
            return [array(ms), array(ms)]
        elif self.mconf['operator'] == "par":
            return ms
        else:
            raise NotImplementedError
    
    def set_s(self, ms, s):
        """ Set sensory dimensions used by module
        """
        ms = array(ms)
        ms[self.mconf['s']] = s
        return ms          
    
    def inverse(self, sg, explore=True):
        # Get nearest neighbor
        if len(self.sm):
            _, idx = self.sm.nn_y(sg)
            m = np.array(self.sm.get_x(idx[0]))
            snn = self.sm.get_y(idx[0])
        else:
            return self.motor_babbling()
        # Add Exploration Noise
        if explore:
            # Detect Movement
            snn_steps = 10
            move_step = snn_steps
            for i in range(1, snn_steps):
                if abs(snn[self.n_sdims * i] - snn[self.n_sdims * (i-1)]) > 0.01:
                    #Move at step i
                    move_step = i
                    break
            # Explore after Movement detection
            if move_step == 1 or move_step == snn_steps:
                start_explo = 0
            else:
                start_explo = (move_step) // 2
            explo_vect = [0.] * start_explo * self.n_mdims + [self.explo_noise]*(snn_steps//2-start_explo) * self.n_mdims
            
            rospy.loginfo("Explonoise: " + str(snn_steps) + str(move_step) + str(snn) + str(explo_vect) + str(m))
            m = np.random.normal(m, explo_vect).clip(-1.,1.)
            rospy.loginfo("New m:" + str(m))
        return m
            
    def produce(self, context=None, explore=True):
        if self.t < self.motor_babbling_n_iter:
            self.m = self.motor_babbling()
            self.s = np.zeros(len(self.s_space))
            self.x = np.zeros(len(self.expl_dims))
        else:
            self.x = self.choose(context)
            self.y = self.inverse(self.x, explore=explore)
            self.m, self.s = self.y, self.x         
        return self.m        
    
    def update_sm(self, m, s): 
        self.sm.add_xy(m, s)
        self.t += 1 
    
    def update_im(self, m, s):
        if self.t >= self.motor_babbling_n_iter:
            return self.interest_model.update(self.s, s)
        
    def competence(self): return self.interest_model.competence()
    def progress(self): return self.interest_model.progress()
    def interest(self): return self.interest_model.current_interest
    

    def perceive(self, m, s):
        self.update_sm(m, s)
        self.last_interest = self.update_im(m, s)
