import os
import sys
import cPickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import brewer2mpl
bmap = brewer2mpl.get_map('Dark2', 'qualitative', 8)
#colors = bmap.mpl_colors
colors = {
      "motor_babbling":bmap.mpl_colors[0], 
      "Hand": bmap.mpl_colors[1], 
      "Joystick_L": bmap.mpl_colors[2], 
      "Joystick_R":bmap.mpl_colors[3], 
      "Ergo":bmap.mpl_colors[4], 
      "Ball":bmap.mpl_colors[5], 
      "Light":bmap.mpl_colors[6], 
      "Sound":bmap.mpl_colors[7]
      }




simu = False

if simu:
    
    # From SIMU
    path = "/home/sforesti/catkin_ws/src/nips2017/logs/"
    experiment_name = "experiment"
    configs = dict(AMB=9)#, RMB=3, RmB=3, FC=1, OS=3)
    n = 10000
    j_error = 0.1
else:
    
    # PARAMS
    experiment_name = "apex1268"
    path = "/data/APEX/" + experiment_name + "/"
    configs = dict(RMB=3)#, RMB=3, RmB=1, FC=3, OS=3)
    n = 20000
    j_error = 0.02


p = 10 # Number of checkpoints
x = range(n)



def discovery(data, print_data=False):
    n = len(data)
    result = [0.]
    for i in range(1, n):
        x = data[i]
        if np.linalg.norm(np.array(x)[::2] - np.mean(np.array(x)[::2])) < 0.05 and np.linalg.norm(np.array(x)[1::2] - np.mean(np.array(x)[1::2])) < 0.05:
            result.append(0.)
        else:
            if print_data:
                print i, x
#             min_dist = np.inf
#             for j in range(i):
#                 y = data[j]
#                 min_dist = min(min_dist, np.linalg.norm(np.array(x) - np.array(y)))
#            result.append(min_dist)    
            result.append(1.)    
    return result

    
s_spaces = ["Hand", "Joystick_L", "Joystick_R", "Ergo", "Ball", "Light", "Sound"]


if False:

    explo = {}
    explo_gain = {}
    explo_touch = {}
    bootstrapped_s = {}
    
    
    for s_space in s_spaces:
        explo[s_space] = {}
        explo_gain[s_space] = {}
        bootstrapped_s[s_space] = {}
    
    

    for config in configs.keys():
        
    
        for s_space in s_spaces:
            explo[s_space][config] = {}
            explo_gain[s_space][config] = {}
            bootstrapped_s[s_space][config] = {}
    
        explo_touch[config] = {}


        for trial in range(configs[config]):
            
            explo_touch[config][trial] = {}
            
            print "\nLoading", config, trial
            
            filename = path + experiment_name + "_" + config + "_" + str(trial) + ".pickle"
            with open(filename, 'r') as f:
                log = cPickle.load(f)
            f.close()
            
            print log["chosen_modules"]
            
            dims = {"motor_babbling":"motor_babbling",
                    "Hand":"mod1",
                    "Joystick_L":"mod3",
                    "Joystick_R":"mod2",
                    "Ergo":"mod4",
                    "Ball":"mod5",
                    "Light":"mod6",
                    "Sound":"mod7"}
            
            cdims = dict(Hand=0,
                        Joystick_L=0,
                        Joystick_R=0,
                        Ergo=1,
                        Ball=2,
                        Light=2,
                        Sound=2)
            
            
            for s_space in explo.keys():
                print "Analysis", s_space
                
                explo_gain[s_space][config][trial] = discovery([log["sm_data"][dims[s_space]][1][i][cdims[s_space]:] for i in range(n)], True if s_space == 'Ergo' else False)
                bootstrapped_s[s_space][config][trial] = next((i for i, x in enumerate(explo_gain[s_space][config][trial]) if x), n)
                #print s_space, explo[s_space][config][trial][:20]
                
            bootstrapped_s["motor_babbling"] = {}
            bootstrapped_s["motor_babbling"][config] = {}
            bootstrapped_s["motor_babbling"][config][trial] = 0
            
            for s_space1 in explo.keys()+ ["motor_babbling"]:
                explo_touch[config][trial][s_space1] = {}
                for s_space2 in explo.keys():
                    explo_touch[config][trial][s_space1][s_space2] = np.zeros(p)
                    
            for i in range(1,n-1):
#                 if abs(log["sm_data"]["mod4"][1][i][0] - log["sm_data"]["mod4"][1][i][-2]) > 0.1 or i in [321, 322]:
#                     print i, "\nJ1:", log["sm_data"]["mod2"][1][i]
#                     print i, "\nJ2:", log["sm_data"]["mod3"][1][i]
#                     print i, "\nErgo:", log["sm_data"]["mod4"][1][i]
#                     print i, "\nBall:", log["sm_data"]["mod5"][1][i]
                for s_space1 in explo_touch[config][trial].keys():
                    mid = dims[s_space1]
                    if mid == log["chosen_modules"][i]:
                        for s_space2 in explo.keys():
                            #print
                            g = explo_gain[s_space2][config][trial][i]
                            #if g > 0:# and s_space2 in ["Joystick_L", "Ergo"]:
                                #m = log["sm_data"]["mod1"][0][i]
                                #print "Iteration:", i, ", Chosen space:", s_space1, ", Gain in", s_space2, ":", g, "bootstrapped_s:", bootstrapped_s[s_space1][config][trial]
#                                 print i, "\nJR:", log["sm_data"]["mod2"][1][i]
#                                 print i, "\nJL:", log["sm_data"]["mod3"][1][i]
#                                 print i, "\nErgo:", log["sm_data"]["mod4"][1][i]
#                                 print i, "\nBall:", log["sm_data"]["mod5"][1][i]
                            if g > 0:# and bootstrapped_s[s_space1][config][trial] < i:
                                explo_touch[config][trial][s_space1][s_space2][i/(n/p)] += g
                    
                    
            for s_space1 in explo_touch[config][trial].keys():
                #print trial, s_space1, [log["chosen_modules"][i*(n/10):(i+1)*(n/10)].count(dims[s_space1]) for i in range(10)]
                chosen = [log["chosen_modules"][j*(n/p):(j+1)*(n/p)].count(dims[s_space1]) for j in range(p)]
                print "chosen", s_space1, chosen
                for s_space2 in explo.keys():
                    explo_touch[config][trial][s_space1][s_space2] /= chosen
#                     if s_space2 == "Hand":
#                         print "   ", s_space2, explo_touch[config][trial][s_space1][s_space2]
                    
                    
    with open(path + 'analysis_touch.pickle', 'wb') as f:
        cPickle.dump(explo_touch, f)

else:
    
    with open(path + 'analysis_touch.pickle', 'r') as f:
        explo_touch = cPickle.load(f)
    f.close()
    
    print explo_touch
    
    config = "RMB"
    #trials = range(configs[config])
    trials = [0, 1, 2]
    print trials
    
    labels = {
              "motor_babbling":"$Random$", 
              "Hand": "$Hand$", 
              "Joystick_L": "$Joystick_L$", 
              "Joystick_R":"$Joystick_R$", 
              "Ergo":"$Ergo$", 
              "Ball":"$Ball$", 
              "Light":"$Light$", 
              "Sound":"$Sound$"
              }
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    for s_space2 in ["Joystick_L", "Joystick_R", "Ergo", "Ball", "Light", "Sound"]:
            
                
        fig, ax = plt.subplots()
        fig.canvas.set_window_title(s_space2)
        #plt.title("\% Touching $"+s_space2+"$ while exploring...", fontsize=24)
            
        for s_space1 in ["motor_babbling", "Hand", "Joystick_L", "Joystick_R", "Ergo", "Ball", "Light", "Sound"]:
#             if s_space1 == "Hand" and s_space2 == "Ball":
#                 print s_space1, s_space2, [explo_touch[config][trial][s_space1][s_space2] for trial in range(configs[config])]
            plt.plot(np.linspace(n/p, n, p), 100. * np.mean([explo_touch[config][trial][s_space1][s_space2] for trial in trials], axis=0), lw=3, color=colors[s_space1], label=labels[s_space1])
            
                
        legend = plt.legend(frameon=True, fontsize=20)
        plt.xticks(fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.xlabel("Iterations", fontsize=20)
        plt.ylabel("\% Reach", fontsize=20)
        frame = legend.get_frame()
        frame.set_facecolor('1.')
        frame.set_edgecolor('0.')
        
        #plt.savefig("/home/sforesti/scm/PhD/cogsci2016/include/obj-explo.pdf", format='pdf', dpi=100, bbox_inches='tight')
        plt.savefig(path + "figs/touch_" + s_space2 + '.pdf', format='pdf', dpi=100, bbox_inches='tight')
        plt.savefig(path + "figs/touch_" + s_space2 + '.png', format='png', dpi=100, bbox_inches='tight')
        
        
        plt.show(block=False)
    plt.show()
            
        