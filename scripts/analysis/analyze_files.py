import os
import sys
import cPickle
import numpy as np



if len(sys.argv) > 1:
    experiment_name = sys.argv[1]
else:
    experiment_name = "nips_24_juillet-3"
    
    
# PARAMS
log_dir = "/data/APEX/"
configs = dict(RMB=4)
iterations = 6000
task = "task_1"

def get_m(ms):
    return ms[:32]

def get_s(ms, mid):
    m_ndims = 32 # number of motor parameters
    s_spaces = {}
    context = range(m_ndims, m_ndims+2)
    s_spaces["mod1"] = range(m_ndims+2, m_ndims+32)
    s_spaces["mod2"] = range(m_ndims+32, m_ndims+52)
    s_spaces["mod3"] = range(m_ndims+52, m_ndims+72)
    s_spaces["mod4"] = [context[0]] + range(m_ndims+72, m_ndims+92)
    s_spaces["mod5"] = context + range(m_ndims+92, m_ndims+112)
    s_spaces["mod6"] = context + range(m_ndims+112, m_ndims+122)
    s_spaces["mod7"] = context + range(m_ndims+122, m_ndims+132)
    s_spaces["mod8"] = range(m_ndims+132, m_ndims+162)
    s_spaces["mod9"] = range(m_ndims+162, m_ndims+192)
    s_spaces["mod10"] = range(m_ndims+192, m_ndims+212)
    s_spaces["mod11"] = range(m_ndims+212, m_ndims+232)
    s_spaces["mod12"] = range(m_ndims+232, m_ndims+252)
    s_spaces["mod13"] = range(m_ndims+252, m_ndims+272)
    s_spaces["mod14"] = range(m_ndims+272, m_ndims+292)
    s_spaces["mod15"] = range(m_ndims+292, m_ndims+312)
    return ms[s_spaces[mid]]



for config in configs.keys():
    print config
    for trial in range(configs[config]):
        print trial
        log = {}
        log["sm_data"] = {}
        log["interests_evolution"] = {}
        for mid in ["mod1", "mod2", "mod3", "mod4", "mod5", "mod6", "mod7", "mod8", "mod9", "mod10", "mod11", "mod12", "mod13", "mod14", "mod15"]:
            log["sm_data"][mid] = [[], []]
            log["interests_evolution"][mid] = []
        log["chosen_modules"] = []
        
        try:
            for iteration in range(iterations):
                
                filename = log_dir + experiment_name + "/" + task + "/condition_" + config + "/trial_" + str(trial) + "/iteration_" + str(iteration) + ".pickle"            
                with open(filename, 'r') as f:
                    log_i = cPickle.load(f)
                
                ms = np.zeros(344, dtype=np.float16)
                ms[range(164) + range(304, 344)] = log_i["ms"]
                for mid in ["mod1", "mod2", "mod3", "mod4", "mod5", "mod6", "mod7", "mod8", "mod9", "mod10", "mod11", "mod12", "mod13", "mod14", "mod15"]:
                    log["sm_data"][mid][0].append(get_m(ms))
                    log["sm_data"][mid][1].append(get_s(ms, mid))
                    log["interests_evolution"][mid].append(log_i["interests"][mid])
                log["chosen_modules"].append(log_i["chosen_module"])
                
            print len(log["chosen_modules"])
    
            filename = log_dir + experiment_name + "/" + experiment_name + "_" + config + "_" + str(trial) + ".pickle"            
            with open(filename, 'wb') as f:
                cPickle.dump(log, f)
        except:
            print "Failed", config, trial, iteration