import numpy as np
from tqdm import tqdm

from environment_explauto.armball import TestArmBallEnv, TestArmBallObsEnv
from environment_explauto.armballs import TestArmBallsEnv, TestArmBallsObsEnv
from mugllearning import Learning

        
if __name__ == "__main__":
    
    print("Create environment")
    environment = TestArmBallEnv()
    environment = TestArmBallsObsEnv()
    
    print("Create agent")
    learning = Learning(dict(m_mins=environment.conf.m_mins,
                             m_maxs=environment.conf.m_maxs,
                             s_mins=environment.conf.s_mins,
                             s_maxs=environment.conf.s_maxs),
                        condition="MGEVAE", explo_noise=0.01, choice_eps=0.1)
    learning.start()
    
    print()
    print("Do 100 autonomous steps:") 
    for i in range(100):
        context = environment.get_current_context()
        m = learning.produce(context)
        s = environment.update(m)
        learning.perceive(s)
        learning.save(experiment_name="test", task="mge_fi", trial=0, folder="../../../../../data/test")
    
    print()
    print("Saving current data to file")
    learning.save(experiment_name="test", task="mge_fi", trial=0, folder="../../../../../data/test")
    
    print("Data before saving")
    print(learning.agent.t)
    print(learning.agent.interests_evolution["mod1"][-10:])
    print(learning.agent.progresses_evolution["mod1"][-10:])
    print(learning.agent.chosen_modules[-10:])
    print(len(learning.agent.modules["mod1"].sensorimotor_model.model.imodel.fmodel.dataset))
    print(len(learning.agent.modules["mod2"].sensorimotor_model.model.imodel.fmodel.dataset))
    print(learning.agent.modules["mod1"].interest_model.current_interest)

    print()
    print("Do 150 autonomous steps:")
    for i in range(150):
        context = environment.get_current_context()
        m = learning.produce(context)
        s = environment.update(m)
        learning.perceive(s)
    
    print("Rebuilding agent from file")
    learning.restart_from_files(experiment_name="test", task="mge_fi", trial=0, iteration=101, folder="../../../../../data/test")
        
    print("Data after rebuilding")
    print(learning.agent.t)
    print(learning.agent.interests_evolution["mod1"][-10:])
    print(learning.agent.progresses_evolution["mod1"][-10:])
    print(learning.agent.chosen_modules[-10:])
    print(len(learning.agent.modules["mod1"].sensorimotor_model.model.imodel.fmodel.dataset))
    print(len(learning.agent.modules["mod2"].sensorimotor_model.model.imodel.fmodel.dataset))
    print(learning.agent.modules["mod1"].interest_model.current_interest)
    
    print()
    print("Do 1000 autonomous steps:")
    for i in tqdm(range(5000)):
        context = environment.get_current_context()
        m = learning.produce(context)
        s = environment.update(m)
        learning.perceive(s)
        

    context = environment.get_current_context()
    print("motor babbling", learning.motor_babbling())
    print("\nPloting interests...")
    learning.plot()
