# -*- coding: utf-8 -*-
from sumo_rl.environment.env_turnpike import TurnpikeEnvironmnet
from maml_rl.envs.turnpike import TurnpikeMeta
import numpy as np
import tqdm


if __name__ == '__main__':
    env = TurnpikeMeta(cfg_file='nets/turnpike_single/net/turnpike/turnpike.single.sumocfg',
                              load_file='nets/turnpike_single/net/turnpike/turnpike.single.savedstate.xml',
                              vsl_files='/nets/turnpike_single/net/turnpike/vsl2.0',
                              use_gui=0,
                              num_seconds=3600,
                              delta_time=60,
                              single_agent=True
                             )
    
    state = env.reset()
    print(env.seed(200))
    obs, act, rwd = [], [], []
    for i in tqdm.trange(2):
        action = np.random.randint(low=0, high=63)
        observation, reward, done, _ = env.step(action)
        obs.append(obs)
        rwd.append(reward)
        act.append(action)
        #print(_['time to collision'].shape)
        
    env.close()
    print(rwd) 