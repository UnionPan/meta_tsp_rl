from sqlite3 import Time
import gym
#from maml_rl.envs.navigation import Navigation2DEnv
#from maml_rl.envs.turnpike import TurnpikeMeta
#from gym.wrappers.time_limit import TimeLimit
#from gym.wrappers.order_enforcing import OrderEnforcing



class make_env(object):
    def __init__(self, env_name, env_kwargs={}, seed=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.seed = seed

    def __call__(self):
        env = gym.make(self.env_name, **self.env_kwargs)
        #env = OrderEnforcing(env)
        #env = TimeLimit(env)
        #env = gym.wrappers.Monitor(env, "recording", force=True)
        if hasattr(env, 'seed'):
            env.seed(self.seed)
        return env


"""""
def make_tp_envs(env_name, env_kwargs={}, seed=None):
    def _make_env():
        env = TurnpikeMeta(cfg_file='nets/turnpike_single/net/turnpike/turnpike.single.sumocfg',
                           load_file='nets/turnpike_single/net/turnpike/turnpike.single.savedstate.xml',
                           vsl_files='/nets/turnpike_single/net/turnpike/vsl2.0',
                           use_gui=0,
                           num_seconds=3600,
                           delta_time=60,
                           single_agent=True
                           )
        if hasattr(env, 'seed'):
            env.seed(seed)
        return env
    return _make_env


def make_env_2(env_name, env_kwargs={}, seed=None):
    def _make_env():
        env = gym.make(env_name, **env_kwargs)
        if hasattr(env, 'seed'):
            env.seed(seed)
        return env
    return _make_env


class make_env2(object):
    def __init__(self, env_name, env_kwargs={}, seed=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.seed = seed

    def __call__(self):
        env = gym.make(self.env_name, **self.env_kwargs)
        if hasattr(env, 'seed'):
            env.seed(self.seed)
        return env
"""""


class Sampler(object):
    def __init__(self,
                 env_name,
                 env_kwargs,
                 batch_size,
                 policy,
                 seed=None,
                 env=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.batch_size = batch_size
        self.policy = policy
        self.seed = seed

        if env is None:
            env = gym.make(env_name, **env_kwargs)
        self.env = env
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)
        self.env.close()
        self.closed = False

    def sample_async(self, *args, **kwargs):
        raise NotImplementedError()

    def sample(self, *args, **kwargs):
        return self.sample_async(*args, **kwargs)


if __name__ == "__main__":
    #env_fns1 = [make_env(env_name = 'turnpike') for i in range(5)]
    #env_fns2 = [make_env2(env_name = '2DNavigation-v0') for i in range(5)]
    #print(env_fns1, env_fns2)
    0