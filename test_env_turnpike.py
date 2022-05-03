# -*- coding: utf-8 -*-

from maml_rl.envs.turnpike.env_turnpike import TurnpikeEnvironment
import numpy as np
import tqdm


if __name__ == '__main__':
    env = TurnpikeEnvironment(cfg_file='maml_rl/envs/turnpike/nets/turnpike_single/net/turnpike/turnpike.single.sumocfg',
                              load_file='maml_rl/envs/turnpike/nets/turnpike_single/net/turnpike/turnpike.single.savedstate.xml',
                              vsl_files='maml_rl/envs/turnpike//nets/turnpike_single/net/turnpike/vsl2.0',
                              use_gui=0,
                              num_seconds=3600,
                              delta_time=60,
                              single_agent=True
                              )
    
    
    
    obs, act, rwd = [], [], []
    for i in tqdm.trange(2):
        action = np.random.randint(low=0, high=63)
        observation, reward, done, _ = env.step(action)
        obs.append(obs)
        rwd.append(reward)
        act.append(action)
        print(_['time to collision'].shape)
        
    env.close()
    print(rwd) 