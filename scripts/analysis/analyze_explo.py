import os
import sys
import cPickle
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

import brewer2mpl
bmap = brewer2mpl.get_map('Dark2', 'qualitative', 8)
colors = bmap.mpl_colors


if len(sys.argv) > 2 and sys.argv[2] in ["compute", "show"]:
    action = sys.argv[2]
else:
    raise NotImplementedError

# PARAMS
experiment_name = sys.argv[1]
path = "/data/APEX/" + experiment_name + '/'

print "Experiment files:", path

configs = dict(RMB=range(12), AMB=range(11), FC=range(9), OS=range(6))
n = 20000
j_error = 0.2



config_colors = dict(AMB=colors[0], RmB=colors[1], FC=colors[2], RMB=colors[3], OS=colors[5])


sw = 20
n_logs = 1
p = 100
x = np.array(np.linspace(0,n,n/p+1), dtype=int)

gss = [0, 1000, 100, 20, 10, 6, 5, 4, 3, 3]


def compute_explo(data, min_, max_, checkpoints=None):
    if checkpoints is None:
        checkpoints = [len(data)]
    nd = len(data[0]) / 10

    gs = gss[nd]
    epss = [(max_ - min_) / gs] * nd
    #print gs, nd, epss
    grid = np.zeros([gs] * nd)

    res = [0]
    for c in range(1, len(checkpoints)):
        for i in range(checkpoints[c-1], checkpoints[c]):
            for timepoint in range(10):
                s = np.array(data[i])[nd*timepoint:nd*(timepoint+1)]
                idxs = np.array((s - min_) / epss, dtype=int)
                #print c, i, idxs
                idxs[idxs>=gs] = gs-1
                idxs[idxs<0] = 0
                #print idxs
                grid[tuple(idxs)] = grid[tuple(idxs)] + 1
        grid[grid > 1] = 1
        res.append(np.sum(grid))
    return np.array(res) / gs ** nd
    

if action == "compute":

    
    explo = {}
    explo['Hand'] = {}
    explo['Joystick_L'] = {}
    explo['Joystick_R'] = {}
    explo['Ergo'] = {}
    explo['Ball'] = {}
    explo['Light'] = {}
    explo['Sound'] = {}

    for config in configs.keys():
        
    
        explo['Hand'][config] = {}
        explo['Joystick_L'][config] = {}
        explo['Joystick_R'][config] = {}
        explo['Ergo'][config] = {}
        explo['Ball'][config] = {}
        explo['Light'][config] = {}
        explo['Sound'][config] = {}
    
        for trial in configs[config]:
            
            filename = path + experiment_name + "_" + config + "_" + str(trial) + ".pickle"
            print "\nLoading", config, trial, filename
            
            try:
                with open(filename, 'r') as f:
                    log = cPickle.load(f)
                f.close()
            except:
                continue
            
            if not (config == "FGB"):
                jl_touch = 0
                jr_touch = 0
                e_touch = 0
                b_touch = 0
                l_touch = 0
                s_touch = 0
                
                for i in range(min(n,len(log["sm_data"]["mod1"][1]))):
                    #print i , "ball", list(log["sm_data"]["mod5"][1][i])[1]
                    #print i, "joystick right", list(log["sm_data"]["mod2"][1][i])
                    #print "ergo", list(log["sm_data"]["mod4"][1][i])[1:]
                    #print "ball", list(log["sm_data"]["mod5"][1][i])[2:]
#                     if i in[819, 820]:
#                         print i, "Ergo:", log["sm_data"]["mod4"][1][i]
#                         print i, "Ball:", log["sm_data"]["mod5"][1][i]
                    #print i
                    if np.linalg.norm(log["sm_data"]["mod2"][1][i] - np.array([np.mean(log["sm_data"]["mod2"][1][i][::2]), np.mean(log["sm_data"]["mod2"][1][i][1::2])] * 10)) > j_error:
                        jr_touch += 1
                        #print
                        #print i, "joystick right", list(log["sm_data"]["mod2"][1][i]), np.linalg.norm(log["sm_data"]["mod2"][1][i] - np.array([-1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1.,  0., -1., 0., -1.,  0., -1.,  0., -1.,  0.]))
                    if abs(list(log["sm_data"]["mod4"][1][i])[1:][-2] - list(log["sm_data"]["mod4"][1][i])[1:][0]) > 0.02:
                        e_touch += 1
                        #print i, log["chosen_modules"][i]
                        #print
                        #print i, "joystick right", list(log["sm_data"]["mod2"][1][i])
                        #print i, "ergo", list(log["sm_data"]["mod4"][1][i])[-2], list(log["sm_data"]["mod4"][1][i])
                    if np.linalg.norm(log["sm_data"]["mod3"][1][i] - np.array([np.mean(log["sm_data"]["mod3"][1][i][::2]), np.mean(log["sm_data"]["mod3"][1][i][1::2])] * 10)) > j_error:
                        jl_touch += 1
                        #print i, "joystick left", list(log["sm_data"]["mod3"][1][i]), [np.mean(log["sm_data"]["mod3"][1][i][::2]), np.mean(log["sm_data"]["mod3"][1][i][1::2])], np.linalg.norm(log["sm_data"]["mod3"][1][i] - np.array([np.mean(log["sm_data"]["mod3"][1][i][::2]), np.mean(log["sm_data"]["mod3"][1][i][1::2])] * 10))
                    
                            
                    if abs(list(log["sm_data"]["mod5"][1][i])[2:][-2] - list(log["sm_data"]["mod5"][1][i])[2:][0]) > 0.02:# or np.linalg.norm(list(log["sm_data"]["mod5"][1][i])[2:][1::2]  - np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])) > 0.1:
                        b_touch += 1
                        #print i, "joystick right", list(log["sm_data"]["mod2"][1][i])
                        #print "ergo", list(log["sm_data"]["mod4"][1][i])
                        #print i , "ball", list(log["sm_data"]["mod5"][1][i]), np.linalg.norm(list(log["sm_data"]["mod5"][1][i])[2:][::2])
                        
                        if np.linalg.norm(log["sm_data"]["mod6"][1][i][2:] - np.array([-1., -1., -1., -1., -1., -1., -1., -1., -1., -1.])) > 0.01:
                            l_touch += 1
                                                 
                                                 
                        if np.linalg.norm(log["sm_data"]["mod7"][1][i][2:] - np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])) > 0.01:
                            s_touch += 1
                            #print
                            #print i, "ball", list(log["sm_data"]["mod5"][1][i])
                            #print "sound", list(log["sm_data"]["mod7"][1][i])
                        
                     
                it = min(n,len(log["sm_data"]["mod1"][1]) )
                print
                print "Iterations:", it
                print "Joystick Left touched:", jl_touch, "percentage:", 100. * jl_touch / it, "%"
                print "Joystick Right touched:", jr_touch, "percentage:", 100. * jr_touch / it, "%"
                print "Ergo touched:", e_touch, "percentage:", 100. * e_touch / it, "%"
                print "Ball touched:", b_touch, "percentage:", int(100. * b_touch / it), "%"
                print "Light touched:", l_touch, "percentage:", int(100. * l_touch / it), "%"
                print "Sound touched:", s_touch, "percentage:", int(100. * s_touch / it), "%"
                print
                
        
                    
            dims = dict(Hand="mod1",
                        Joystick_R="mod2",
                        Joystick_L="mod3",
                        Ergo="mod4",
                        Ball="mod5",
                        Light="mod6",
                        Sound="mod7")
            
            cdims = dict(Hand=0,
                        Joystick_L=0,
                        Joystick_R=0,
                        Ergo=1,
                        Ball=2,
                        Light=2,
                        Sound=2)
            
            
            for s_space in explo.keys():
                #print "Analysis", s_space
                
                try:
                    explo[s_space][config][trial] = compute_explo([log["sm_data"][dims[s_space]][1][i][cdims[s_space]:] for i in range(len(log["sm_data"][dims[s_space]][1]))], -1., 1., x)
                except:
                    print i#, len(log["sm_data"][dims[s_space]][1]), log["sm_data"][dims[s_space]][1][i][cdims[s_space]:]
                    
                #print explo[s_space][config][trial]
             
                 
    with open(path + 'analysis_explo.pickle', 'wb') as f:
        cPickle.dump(explo, f)

else:
    
    with open(path + 'analysis_explo.pickle', 'r') as f:
        explo = cPickle.load(f)
    f.close()
    
    
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    
    for s_space in explo.keys():
                 
        fig, ax = plt.subplots()
        fig.canvas.set_window_title(s_space)
        #plt.title('Exploration of $' + s_space + "$ space", fontsize=24)
        
        for config in explo[s_space].keys():
            ys = [100.*explo[s_space][config][trial] for trial in explo[s_space][config].keys()] 
            ymean = np.mean(ys, axis=0)
            ymed = np.percentile(ys, 50, axis=0)
            print config, s_space, ymed
            ymax = np.percentile(ys, 100, axis=0)
            ymin = np.percentile(ys, 0, axis=0)
            
            plt.plot(x, ymed, lw=5, color=config_colors[config], label=config, alpha=0.9)
            for y in ys:
                plt.plot(x, y, lw=2, color=config_colors[config], alpha=0.3)
                if config == "AMB" and s_space == "Joystick_L":
                    print y
            #plt.fill_between(x, ymin, ymax, color=config_colors[config], alpha=0.2)
        
            
        legend = plt.legend(frameon=True, loc="upper left", fontsize=24)
        plt.xticks([0, 5000, 10000, 15000, 20000], fontsize = 20)
        plt.yticks(fontsize = 20)
        plt.xlabel("Iterations", fontsize=24)
        plt.ylabel("Exploration \%", fontsize=24)
        frame = legend.get_frame()
        frame.set_facecolor('1.')
        frame.set_edgecolor('0.')
        
        #plt.savefig("/home/sforesti/scm/PhD/cogsci2016/include/obj-explo.pdf", format='pdf', dpi=100, bbox_inches='tight')
        plt.savefig(path + "/figs/explo_" + s_space + '.pdf', format='pdf', dpi=100, bbox_inches='tight')
        #plt.savefig(path + "/figs/explo_" + s_space + '.png', format='png', dpi=100, bbox_inches='tight')
        
        
        plt.show(block=False)
    plt.show()
            
