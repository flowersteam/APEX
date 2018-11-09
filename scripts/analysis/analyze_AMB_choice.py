import os
import sys
import cPickle
import numpy as np
import matplotlib.pyplot as plt


# PARAMS
plt.rc('text', usetex=True)
plt.rc('font', family='serif')


simu = False

if simu:
    
    # From SIMU
    path = "/home/sforesti/catkin_ws/src/nips2017/logs/"
    experiment_name = "experiment"
    configs = dict(AMB=10)
    n = 10000
    j_error = 0.1
else:
    
    # PARAMS
    path = "/data/APEX/nips_24_juillet-newbounds-4/"
    experiment_name = "nips_24_juillet-newbounds-4"
    configs = dict(AMB=6)
    n = 5000
    j_error = 0.3



    
data = {}
for i in range(configs["AMB"]):
        
    filename = path + experiment_name + "_AMB_" + str(i) + ".pickle"
    
    with open(filename, 'r') as f:
        log = cPickle.load(f)
    f.close()
    data[i] = log
    
    print i, len(log["sm_data"]["mod2"][1])
    #print len(data[i]["interests_evolution"])
    #print data[i]["interests_evolution"]


labels = dict(mod1="Hand", 
             mod2="Joystick Right", 
             mod3="Joystick Left", 
             mod4="Ergo", 
             mod5="Ball", 
             mod6="Light", 
             mod7="Sound",
             mod8="HR",
             mod9="BA",
             mod10="AR",
             mod11="O1",
             mod12="O2",
             mod13="O3",
             mod14="R1",
             mod15="R2",
             )

for i in range(configs["AMB"]):
    plt.figure()
    for mid in ["mod1", "mod3", "mod2", "mod4", "mod5", "mod6", "mod7", "mod14", "mod15"]:
        plt.plot(range(0, n, 100), [data[i]["chosen_modules"][100*j:100*(j+1)].count(mid) for j in range(n/100)], lw=2, label=labels[mid])
    
    
    legend = plt.legend(frameon=True, fontsize=18, loc="left")
    plt.xticks(fontsize = 16)
    plt.yticks(fontsize = 16)
    plt.xlabel("Iterations", fontsize=18)
    plt.ylabel("Interest", fontsize=18)
    frame = legend.get_frame()
    frame.set_facecolor('1.')
    frame.set_edgecolor('0.')
        
        
    plt.savefig("/home/sforesti/scm/Flowers/NIPS2016/data/figs/interest_AMB_" + str(i) + '.pdf', format='pdf', dpi=100, bbox_inches='tight')
    plt.show(block=False)
plt.show()
