import numpy as np
import rospy
from explauto.utils import rand_bounds, bounds_min_max, softmax_choice, prop_choice
from explauto.utils.config import make_configuration
from learning_module import LearningModule


class Supervisor(object):
    def __init__(self, config, babbling_mode="active", n_motor_babbling=0.1, explo_noise=0.05, choice_eps=0.2, normalize_interests=True):
        self.config = config
        self.babbling_mode = babbling_mode
        self.n_motor_babbling = n_motor_babbling
        self.explo_noise = explo_noise
        self.choice_eps = choice_eps,
        self.normalize_interests = normalize_interests
        
        self.conf = make_configuration(**config)
        
        self.t = 0
        self.modules = {}
        self.chosen_modules = []
        self.goals = []
        self.progresses_evolution = {}
        self.interests_evolution = {}
        
        self.ms = None
        
        self.have_to_replay_arm_demo = None
            
        self.mid_control = ''
        self.measure_interest = False
        
            
        # Define motor and sensory spaces:
        m_ndims = self.conf.m_ndims # number of motor parameters
        
        self.m_space = range(m_ndims)
        self.c_dims = range(m_ndims, m_ndims+2)
        self.s_hand = range(m_ndims+2, m_ndims+32)
        self.s_joystick_1 = range(m_ndims+32, m_ndims+52)
        self.s_joystick_2 = range(m_ndims+52, m_ndims+72)
        self.s_ergo = range(m_ndims+72, m_ndims+92)
        self.s_ball = range(m_ndims+92, m_ndims+112)
        self.s_light = range(m_ndims+112, m_ndims+122)
        self.s_sound = range(m_ndims+122, m_ndims+132)
        
        self.s_hand_right = range(m_ndims+132, m_ndims+162)
        self.s_base = range(m_ndims+162, m_ndims+192)
        self.s_arena = range(m_ndims+192, m_ndims+212)
        self.s_obj1 = range(m_ndims+212, m_ndims+232)
        self.s_obj2 = range(m_ndims+232, m_ndims+252)
        self.s_obj3 = range(m_ndims+252, m_ndims+272)
        self.s_rdm1 = range(m_ndims+272, m_ndims+292)
        self.s_rdm2 = range(m_ndims+292, m_ndims+312)
        
        self.s_spaces = dict(s_hand=self.s_hand, 
                             s_joystick_1=self.s_joystick_1, 
                             s_joystick_2=self.s_joystick_2, 
                             s_ergo=self.s_ergo, 
                             s_ball=self.s_ball, 
                             s_light=self.s_light, 
                             s_sound=self.s_sound,
                             s_hand_right=self.s_hand_right,
                             s_base=self.s_base,
                             s_arena=self.s_arena,
                             s_obj1=self.s_obj1,
                             s_obj2=self.s_obj2,
                             s_obj3=self.s_obj3,
                             s_rdm1=self.s_rdm1,
                             s_rdm2=self.s_rdm2)
        
        #print
        #print "Initialize agent with spaces:"
        #print "Motor", self.m_space
        #print "Context", self.c_dims
        #print "Hand", self.s_hand
        #print "Joystick1", self.s_joystick_1
        #print "Joystick2", self.s_joystick_2
        #print "Ergo", self.s_ergo
        #print "Ball", self.s_ball
        #print "Light", self.s_light
        #print "Sound", self.s_sound
        

        # Create the 6 learning modules:
        self.modules['mod1'] = LearningModule("mod1", self.m_space, self.s_hand, self.conf, explo_noise=self.explo_noise, normalize_interests=self.normalize_interests)
        self.modules['mod2'] = LearningModule("mod2", self.m_space, self.s_joystick_1, self.conf, explo_noise=self.explo_noise, normalize_interests=self.normalize_interests)
        self.modules['mod3'] = LearningModule("mod3", self.m_space, self.s_joystick_2, self.conf, explo_noise=self.explo_noise, normalize_interests=self.normalize_interests)
        self.modules['mod4'] = LearningModule("mod4", self.m_space, [self.c_dims[0]] + self.s_ergo, self.conf, context_mode=dict(mode='mcs', context_n_dims=1, context_sensory_bounds=[[-1.],[1.]]), explo_noise=self.explo_noise, normalize_interests=self.normalize_interests)
        self.modules['mod5'] = LearningModule("mod5", self.m_space, self.c_dims + self.s_ball, self.conf, context_mode=dict(mode='mcs', context_n_dims=2, context_sensory_bounds=[[-1., -1.],[1., 1.]]), explo_noise=self.explo_noise, normalize_interests=self.normalize_interests)
        self.modules['mod6'] = LearningModule("mod6", self.m_space, self.c_dims + self.s_light, self.conf, context_mode=dict(mode='mcs', context_n_dims=2, context_sensory_bounds=[[-1., -1.],[1., 1.]]), explo_noise=self.explo_noise, normalize_interests=self.normalize_interests)
        self.modules['mod7'] = LearningModule("mod7", self.m_space, self.c_dims + self.s_sound, self.conf, context_mode=dict(mode='mcs', context_n_dims=2, context_sensory_bounds=[[-1., -1.],[1., 1.]]), explo_noise=self.explo_noise, normalize_interests=self.normalize_interests)
    
        self.modules['mod8'] = LearningModule("mod8", self.m_space, self.s_hand_right, self.conf, explo_noise=self.explo_noise, normalize_interests=self.normalize_interests)
        self.modules['mod9'] = LearningModule("mod9", self.m_space, self.s_base, self.conf, explo_noise=self.explo_noise, normalize_interests=self.normalize_interests)
        self.modules['mod10'] = LearningModule("mod10", self.m_space, self.s_arena, self.conf, explo_noise=self.explo_noise, normalize_interests=self.normalize_interests)
        self.modules['mod11'] = LearningModule("mod11", self.m_space, self.s_obj1, self.conf, explo_noise=self.explo_noise, normalize_interests=self.normalize_interests)
        self.modules['mod12'] = LearningModule("mod12", self.m_space, self.s_obj2, self.conf, explo_noise=self.explo_noise, normalize_interests=self.normalize_interests)
        self.modules['mod13'] = LearningModule("mod13", self.m_space, self.s_obj3, self.conf, explo_noise=self.explo_noise, normalize_interests=self.normalize_interests)
        self.modules['mod14'] = LearningModule("mod14", self.m_space, self.s_rdm1, self.conf, explo_noise=self.explo_noise, normalize_interests=self.normalize_interests)
        self.modules['mod15'] = LearningModule("mod15", self.m_space, self.s_rdm2, self.conf, explo_noise=self.explo_noise, normalize_interests=self.normalize_interests)
        
        self.space2mid = dict(s_hand="mod1", 
                             s_joystick_1="mod2", 
                             s_joystick_2="mod3", 
                             s_ergo="mod4", 
                             s_ball="mod5", 
                             s_light="mod6", 
                             s_sound="mod7",
                             s_hand_right='mod8',
                             s_base='mod9',
                             s_arena='mod10',
                             s_obj1='mod11',
                             s_obj2='mod12',
                             s_obj3='mod13',
                             s_rdm1='mod14',
                             s_rdm2='mod15')   
         
        self.mid2space = dict(mod1="s_hand", 
                             mod2="s_joystick_1", 
                             mod3="s_joystick_2", 
                             mod4="s_ergo", 
                             mod5="s_ball", 
                             mod6="s_light", 
                             mod7="s_sound",
                             mod8="s_hand_right",
                             mod9="s_base",
                             mod10="s_arena",
                             mod11="s_obj1",
                             mod12="s_obj2",
                             mod13="s_obj3",
                             mod14="s_rdm1",
                             mod15="s_rdm2",)
        
        for mid in self.modules.keys():
            self.progresses_evolution[mid] = []
            self.interests_evolution[mid] = []
    
    def mid_to_space(self, mid): return self.mid2space[mid]
    def space_to_mid(self, space): return self.space2mid[space]
    def get_space_names(self): return ["s_hand", "s_joystick_1", "s_joystick_2", "s_ergo", "s_ball", "s_light", "s_sound", "s_hand_right", "s_base", "s_arena", "s_obj1", "s_obj2", "s_obj3", "s_rdm1", "s_rdm2"]
    def get_last_focus(self): return self.mid_to_space(self.mid_control) if self.mid_control else ""
    
    def save_iteration(self, i):
        interests = {}
        for mid in self.modules.keys():
            interests[mid] = np.float16(self.interests_evolution[mid][i])
        return {"ms":np.array(self.ms, dtype=np.float16)[range(self.conf.m_ndims+132) + range(self.conf.m_ndims+272, self.conf.m_ndims+self.conf.s_ndims)],
                "chosen_module":self.chosen_modules[i],
                "goal":self.goals[i],
                "interests": interests}

    def forward_iteration(self, data_iteration):
        ms = np.zeros(self.conf.m_ndims+self.conf.s_ndims)
        ms[range(self.conf.m_ndims+132) + range(self.conf.m_ndims+272, self.conf.m_ndims+self.conf.s_ndims)] = data_iteration["ms"]        
        m = self.get_m(ms)
        chosen_mid = data_iteration["chosen_module"]
        sg = data_iteration["goal"]
        interests = data_iteration["interests"]
                
        for mid in self.modules.keys():
            smid = self.modules[mid].get_s(ms)
            
            if mid == "mod4":
                if min(abs(smid[0] - smid[-2]), 2 - abs(smid[0] - smid[-2])) > 0.02:
                    self.modules[mid].update_sm(m, smid)
            elif mid == "mod5":
                if min(abs(smid[1] - smid[-2]), 2 - abs(smid[1] - smid[-2])) > 0.02:
                    self.modules[mid].update_sm(m, smid)                
            else:
                self.modules[mid].update_sm(m, smid)

        if sg is not None:
            self.modules[chosen_mid].s = sg
            self.modules[chosen_mid].update_im(m, self.modules[chosen_mid].get_s(ms))
        for mid in self.modules.keys():
            self.interests_evolution[mid].append(interests[mid])
        
        self.goals.append(sg)
        self.chosen_modules.append(chosen_mid)

        self.t += 1
    
        
    def choose_babbling_module(self, mode='active'):
        interests = {}
        for mid in self.modules.keys():
            interests[mid] = self.modules[mid].interest()
        
        if mode == 'random':
            mid = np.random.choice(interests.keys())
        elif mode == 'greedy':
            eps_rgb = 0.2
            if np.random.random() < eps_rgb:
                mid = np.random.choice(interests.keys())
            else:
                mid = max(interests, key=interests.get)
        elif mode == 'active' or mode == 'activemix':
            eps_rgb = 0.2
            n_rgb = 200
            interest_threshold = 0.0
            temperature = 0.5
            
            if self.t < n_rgb or np.random.random() < eps_rgb or sum(interests.values()) == 0.:
                mid = np.random.choice(interests.keys())
            else:
                total_interest = sum([interests[key] for key in interests.keys()])
                non_zero_interests = {key:interests[key] for key in interests.keys() if interests[key] > total_interest * interest_threshold}
                w = np.array(non_zero_interests.values())   
                w = w / np.sum(w)
                probas = np.exp(w / temperature)
                probas = probas / np.sum(probas)
                idx = np.where(np.random.multinomial(1, probas) == 1)[0][0]
                mid = non_zero_interests.keys()[idx]
                
        elif mode == 'prop':
            eps_rgb = 0.2
            n_rgb = 200
            
            if self.t < n_rgb or np.random.random() < eps_rgb or sum(interests.values()) == 0.:
                mid = np.random.choice(interests.keys())
            else:
                total_interest = sum([interests[key] for key in interests.keys()])
                w = np.array(interests.values())   
                probas = w / np.sum(w)
                idx = np.where(np.random.multinomial(1, probas) == 1)[0][0]
                mid = interests.keys()[idx]
                
        
        elif mode == 'FC':
            # Fixed Curriculum
            mids = ["mod1", "mod3", "mod2", "mod4", "mod5", "mod6", "mod7"]
            n = 20000.
            i = max(0, min(int(self.t / (n / 7.)), 6))
            mid = mids[i]
            
        elif mode == 'OS':
            # One Space: ball space
            mid = "mod5"
        
        self.chosen_modules.append(mid)
        return mid
        
        
    def fast_forward(self, log, forward_im=False):
        #ms_list = []
        for m,s in zip(log.logs['motor'], log.logs['sensori']):
            ms = np.append(m,s)
            self.update_sensorimotor_models(ms)
            #ms_list += [ms]
        for mid, mod in self.modules.iteritems():
            mod.fast_forward_models(log, ms_list=None, from_log_mod=mid, forward_im=forward_im)        
        
    def eval_mode(self): 
        self.sm_modes = {}
        for mod in self.modules.values():
            self.sm_modes[mod.mid] = mod.sensorimotor_model.mode
            mod.sensorimotor_model.mode = 'exploit'
                
    def learning_mode(self): 
        for mod in self.modules.values():
            mod.sensorimotor_model.mode = self.sm_modes[mod.mid]
                
    def check_bounds_dmp(self, m_ag):return bounds_min_max(m_ag, self.conf.m_mins, self.conf.m_maxs)
    def motor_primitive(self, m): return m
    def sensory_primitive(self, s): return s
    def get_m(self, ms): return ms[self.conf.m_dims]
    def get_s(self, ms): return ms[self.conf.s_dims]
    
    def motor_babbling(self):
        self.m = self.modules["mod1"].motor_babbling()
        return self.m
    
    def set_ms(self, m, s): return np.array(list(m) + list(s))
            
    def update_sensorimotor_models(self, ms):
        for mid in self.modules.keys():
            m = self.modules[mid].get_m(ms)
            s = self.modules[mid].get_s(ms)
            
            if mid == "mod4":
                if min(abs(s[0] - s[-2]), 2 - abs(s[0] - s[-2])) > 0.02:
                    self.modules[mid].update_sm(m, s)
            elif mid == "mod5":
                if min(abs(s[1] - s[-2]), 2 - abs(s[1] - s[-2])) > 0.02:
                    self.modules[mid].update_sm(m, s)                
            else:
                self.modules[mid].update_sm(m, s)
        
    def increase_interest(self, mid):
        self.modules[mid].interest_model.current_progress = self.modules[mid].interest_model.current_progress * 1.1
        self.modules[mid].interest_model.current_interest = abs(self.modules[mid].interest_model.current_progress)
            
            
    def produce(self, context, space=None):
        if np.random.random() < self.n_motor_babbling:
            self.mid_control = None
            self.chosen_modules.append("motor_babbling")
            rospy.loginfo("Random Motor Babbling")
            return self.motor_babbling()
        else:
            if space is None:
                if self.have_to_replay_arm_demo is not None:
                    self.m = self.have_to_replay_arm_demo
                    self.have_to_replay_arm_demo = None
                    self.chosen_modules.append("replay_arm_demo")
                    self.mid_control = None
                    return self.m
                mid = self.choose_babbling_module(self.babbling_mode)
            else:
                mid = self.space2mid[space]
                self.chosen_modules.append("forced_" + mid)
                self.increase_interest(mid)
            self.mid_control = mid
            rospy.loginfo("Chosen module: {}".format(mid))


            explore = True  
            self.measure_interest = False   
            #print "babbling_mode", self.babbling_mode           
            if self.babbling_mode == "active" or self.babbling_mode == "prop":
                #print "interest", mid, self.modules[mid].interest()
                if self.modules[mid].interest() == 0.:
                    #print "interest 0: exploit"
                    # In condition AMB, in 20% of iterations we do not explore but measure interest
                    explore = False
                    self.measure_interest = True
                if np.random.random() < 0.2:                        
                    #print "random chosen to exploit"
                    # In condition AMB, in 20% of iterations we do not explore but measure interest
                    explore = False
                    self.measure_interest = True  
                          
            elif self.babbling_mode == "activemix":
                if np.random.random() < 0.2:                        
                    #print "random chosen to exploit"
                    # In condition AMB, in 20% of iterations we do not explore but measure interest
                    explore = False
                self.measure_interest = True 

            j_sm = self.modules["mod2"].sensorimotor_model
            if self.modules[mid].context_mode is None:
                self.m = self.modules[mid].produce(j_sm=j_sm, explore=explore)
            else:                     
                self.m = self.modules[mid].produce(context=np.array(context)[range(self.modules[mid].context_mode["context_n_dims"])], j_sm=j_sm, explore=explore)
            return self.m
    
    def inverse(self, mid, s, context):
        if self.modules[mid].context_mode is not None:
            s = np.array(list(context[:self.modules[mid].context_mode["context_n_dims"]]) + list(s))
        else:
            s = np.array(s)
        self.mid_control = None
        self.chosen_modules.append("inverse_" + mid)
        self.m = self.modules[mid].inverse(s)
        return self.m
    
    def dist_angle(self, a1, a2): return min(abs(a1 - a2), 2 - abs(a1 - a2))
    
    def ball_moves(self, s):
        a1 = s[16]
        a2 = s[18]
        # print "ball end angular speed", self.dist_angle(a1, a2)
        return self.dist_angle(a1, a2) > 0.1
    
    def perceive(self, s, m_demo=None, j_demo=False):
        s = self.sensory_primitive(s)
        #print "perceive len(s)", len(s), s[92:112]
        if j_demo or self.ball_moves(s[92:112]):
            rospy.sleep(5)
        if m_demo is not None:
            ms = self.set_ms(m_demo, s)
            self.update_sensorimotor_models(ms)
            self.have_to_replay_arm_demo = m_demo
            self.chosen_modules.append("m_demo")
        elif j_demo:
            m0 = [0]*self.conf.m_ndims
            m0s = self.set_ms(m0, s[:2] + [0]*30 + s[2:])
            for mid in self.modules.keys():
                if not (mid == "mod1"): # don't update hand model
                    self.modules[mid].update_sm(self.modules[mid].get_m(m0s), self.modules[mid].get_s(m0s))
            self.chosen_modules.append("j_demo")
        else:
            if not hasattr(self, "m"):
                return False
            ms = self.set_ms(self.m, s)
            self.ms = ms
            self.update_sensorimotor_models(ms)
            if self.mid_control is not None and self.measure_interest:
                self.modules[self.mid_control].update_im(self.modules[self.mid_control].get_m(ms), self.modules[self.mid_control].get_s(ms))
            if self.mid_control is not None and self.measure_interest and self.modules[self.mid_control].t >= self.modules[self.mid_control].motor_babbling_n_iter:
                self.goals.append(self.modules[self.mid_control].s)
            else:
                self.goals.append(None)
        self.t = self.t + 1
        
        for mid in self.modules.keys():
            self.progresses_evolution[mid].append(self.modules[mid].progress())
            self.interests_evolution[mid].append(self.modules[mid].interest())
                
    
        return True
    
    
    def produce_goal(self, context, goal):
        if goal == "hand_up":
            self.m = self.move_hand(context, "up")
        elif goal == "hand_forward":
            self.m = self.move_hand(context, "forward")
        elif goal == "hand_right":
            self.m = self.move_hand(context, "right")
        elif goal == "hand_left":
            self.m = self.move_hand(context, "left")
        elif goal == "joystick_1_forward":        
            self.m = self.motor_move_joystick_1(context, "forward")
        elif goal == "joystick_1_right":
            self.m = self.motor_move_joystick_1(context, "right")
        elif goal == "joystick_1_left":
            self.m = self.motor_move_joystick_1(context, "left")
        elif goal == "joystick_2_forward":
            self.m = self.motor_move_joystick_2(context, "forward")
        elif goal == "joystick_2_right":
            self.m = self.motor_move_joystick_2(context, "right")
        elif goal == "joystick_2_left":
            self.m = self.motor_move_joystick_2(context, "left")
        elif goal == "ergo_right":
            self.m = self.motor_move_ergo(context, "right")
        elif goal == "ergo_left":
            self.m = self.motor_move_ergo(context, "left")
        elif goal == "ball_right":
            self.m = self.motor_move_ball(context, "right")
        elif goal == "ball_left":
            self.m = self.motor_move_ball(context, "left")
        elif goal == "light":
            self.m = self.motor_make_light(context)
        elif goal == "sound":
            self.m = self.motor_make_sound(context)
        return self.m
    
    def move_hand(self, context, direction="up"):
        if direction=="up":
            return self.inverse("mod1", [0., 0., 0.,
                                               0., 0., 0., 
                                               0., 0., 0.5,
                                               0., 0., 0.5,
                                               0., 0., 1.,
                                               0., 0., 1., 
                                               0., 0., 1.,
                                               0., 0., 1.,
                                               0., 0., 1., 
                                               0., 0., 1.], context)
        elif direction=="forward":
            return self.inverse("mod1", [0., 0., 0., 
                                               0., 0., 0., 
                                               0.5, 0., 0., 
                                               0.5, 0., 0., 
                                               1., 0., 0., 
                                               1., 0., 0., 
                                               1., 0., 0., 
                                               1., 0., 0., 
                                               1., 0., 0., 
                                               1., 0., 0.,], context)
        elif direction=="right":
            return self.inverse("mod1", [0., 0., 0., 
                                               0., 0., 0., 
                                               0., -0.5, 0., 
                                               0., -0.5, 0., 
                                               0., -1., 0., 
                                               0., -1., 0., 
                                               0., -1., 0., 
                                               0., -1., 0., 
                                               0., -1., 0., 
                                               0., -1., 0.,], context)
        elif direction=="left":
            return self.inverse("mod1", [0., 0., 0., 
                                               0., 0., 0., 
                                               0., 0.5, 0., 
                                               0., 0.5, 0., 
                                               0., 1., 0., 
                                               0., 1., 0., 
                                               0., 1., 0., 
                                               0., 1., 0., 
                                               0., 1., 0., 
                                               0., 1., 0.,], context)
        else:
            raise NotImplementedError
            
        
    def motor_move_joystick_1(self, context, direction="forward"):
        if direction=="forward":
            return self.inverse("mod2", [-1., 0., 
                                               -1., 0., 
                                               0., 0., 
                                               1., 0., 
                                               1., 0., 
                                               1., 0., 
                                               1., 0., 
                                               0., 0., 
                                               -1., 0., 
                                               -1., 0.], context)
        elif direction=="right":
            return self.inverse("mod2", [-1., 0., 
                                               -1., 0., 
                                               -1., 0., 
                                               -1., 1., 
                                               -1., 1., 
                                               -1., 1., 
                                               -1., 1., 
                                               -1., 0., 
                                               -1., 0., 
                                               -1., 0.], context)
        elif direction=="left":
            return self.inverse("mod2", [-1., 0., 
                                               -1., 0., 
                                               -1., 0., 
                                               -1., -1., 
                                               -1., -1., 
                                               -1., -1., 
                                               -1., -1., 
                                               -1., 0., 
                                               -1., 0., 
                                               -1., 0.], context)  
        else:
            raise NotImplementedError
              
    def motor_move_joystick_2(self, context, direction="forward"):
        if direction=="forward":
            return self.inverse("mod3", [0., -1., 
                                               0., -1., 
                                               0., 0., 
                                               0., 1., 
                                               0., 1., 
                                               0., 1., 
                                               0., 1., 
                                               0., 0., 
                                               0., -1., 
                                               0., -1.], context)
        elif direction=="right":
            return self.inverse("mod3", [0., -1.,
                                               0., -1.,
                                               0., -1., 
                                               -1., -1.,
                                               -1., -1.,
                                               -1., -1.,
                                               -1., -1.,
                                               0., -1., 
                                               0., -1., 
                                               0., -1.], context)
        elif direction=="left":
            return self.inverse("mod3", [0., -1., 
                                               0., -1., 
                                               0., -1., 
                                               1., -1.,
                                               1., -1.,
                                               1., -1.,
                                               1., -1.,
                                               0., -1., 
                                               0., -1., 
                                               0., -1.], context)
        else:
            raise NotImplementedError
    
    def motor_move_ergo(self, context, direction="right"):
        angle = context[0]
        if direction=="right":
            return self.inverse("mod4", [angle, -1.,
                                               angle, -1.,
                                               ((angle+1.) % 2.)- 1., 0.,
                                               ((angle+1.-0.1) % 2.)- 1., 1.,
                                               ((angle+1.-0.2) % 2.)- 1., 1.,
                                               ((angle+1.-0.3) % 2.)- 1., 1.,
                                               ((angle+1.-0.4) % 2.)- 1., 1.,
                                               ((angle+1.-0.5) % 2.)- 1., 0.,
                                               ((angle+1.-0.5) % 2.)- 1., -1.,
                                               ((angle+1.-0.5) % 2.)- 1., -1.], context)
        elif direction=="left":
            return self.inverse("mod4", [angle, -1.,
                                               angle, -1.,
                                               ((angle+1.) % 2.)- 1., 0.,
                                               ((angle+1.+0.1) % 2.)- 1., 1.,
                                               ((angle+1.+0.2) % 2.)- 1., 1.,
                                               ((angle+1.+0.3) % 2.)- 1., 1.,
                                               ((angle+1.+0.4) % 2.)- 1., 1.,
                                               ((angle+1.+0.5) % 2.)- 1., 0.,
                                               ((angle+1.+0.5) % 2.)- 1., -1.,
                                               ((angle+1.+0.5) % 2.)- 1., -1.], context)
        else:
            raise NotImplementedError
        
    def motor_move_ball(self, context, direction="right"):
        angle = context[1]
        if direction=="right":
            return self.inverse("mod5", [angle, -1.,
                                               angle, -1.,
                                               ((angle+1.) % 2.)- 1., 0.,
                                               ((angle+1.+0.1) % 2.)- 1., 1.,
                                               ((angle+1.+0.2) % 2.)- 1., 1.,
                                               ((angle+1.+0.3) % 2.)- 1., 1.,
                                               ((angle+1.+0.4) % 2.)- 1., 1.,
                                               ((angle+1.+0.5) % 2.)- 1., 0.,
                                               ((angle+1.+0.5) % 2.)- 1., -1.,
                                               ((angle+1.+0.5) % 2.)- 1., -1.], context)
        elif direction=="left":
            return self.inverse("mod5", [angle, -1.,
                                               angle, -1.,
                                               ((angle+1.) % 2.)- 1., 0.,
                                               ((angle+1.+0.1) % 2.)- 1., 1.,
                                               ((angle+1.+0.2) % 2.)- 1., 1.,
                                               ((angle+1.+0.3) % 2.)- 1., 1.,
                                               ((angle+1.+0.4) % 2.)- 1., 1.,
                                               ((angle+1.+0.5) % 2.)- 1., 0.,
                                               ((angle+1.+0.5) % 2.)- 1., -1.,
                                               ((angle+1.+0.5) % 2.)- 1., -1.], context)
        else:
            raise NotImplementedError
            
    
    def motor_make_light(self, context):
        current_light = context[1]
        return self.inverse("mod6", [((current_light+1.)%2.) - 1.,
                                     ((current_light+1.)%2.) - 1.,
                                     ((current_light+1.)%2.) - 1.,
                                     ((current_light+1.+0.25)%2.) - 1.,
                                     ((current_light+1.+0.5)%2.) - 1.,
                                     ((current_light+1.+0.5)%2.) - 1.,
                                     ((current_light+1.+0.5)%2.) - 1.,
                                     ((current_light+1.+0.5)%2.) - 1.,
                                     ((current_light+1.+0.5)%2.) - 1.,
                                     ((current_light+1.+0.5)%2.) - 1.], context)
    
    def motor_make_sound(self, context):
        current_sound = context[1]
        return self.inverse("mod7", [((current_sound+1.)%2.) - 1.,
                                     ((current_sound+1.)%2.) - 1.,
                                     ((current_sound+1.)%2.) - 1.,
                                     ((current_sound+1.+0.25)%2.) - 1.,
                                     ((current_sound+1.+0.5)%2.) - 1.,
                                     ((current_sound+1.+0.5)%2.) - 1.,
                                     ((current_sound+1.+0.5)%2.) - 1.,
                                     ((current_sound+1.+0.5)%2.) - 1.,
                                     ((current_sound+1.+0.5)%2.) - 1.,
                                     ((current_sound+1.+0.5)%2.) - 1.], context)
        

    def get_normalized_interests_evolution(self):
        data = np.transpose(np.array([self.interests_evolution[mid] for mid in ["mod1", "mod2", "mod3", "mod4", "mod5", "mod6", "mod7", "mod8", "mod9", "mod10", "mod11", "mod12", "mod13", "mod14", "mod15"]]))
        data_sum = data.sum(axis=1)
        data_sum[data_sum==0.] = 1.
        return data / data_sum.reshape(data.shape[0],1)
    
    def get_normalized_interests(self):
        interests = {}
        for mid in self.modules.keys():
            interests[mid] = self.modules[mid].interest()
            
        s = sum(interests.values())
        if s > 0:
            for mid in self.modules.keys():
                interests[mid] = interests[mid] / s
        return interests
        
