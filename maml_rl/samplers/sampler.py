import gym
from maml_rl.envs.turnpike import TurnpikeMeta

class make_env(object):
    def __init__(self, env_name, env_kwargs={}, seed=None):
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.seed = seed

    def __call__(self):
        env = TurnpikeMeta(cfg_file='nets/turnpike_single/net/turnpike/turnpike.single.sumocfg',
                                  load_file='nets/turnpike_single/net/turnpike/turnpike.single.savedstate.xml',
                                  vsl_files='/nets/turnpike_single/net/turnpike/vsl2.0',
                                  use_gui=0,
                                  num_seconds=3600,
                                  delta_time=60,
                                  single_agent=True
                                  )
        #env = gym.wrappers.Monitor(env, "recording", force=True)
        if hasattr(env, 'seed'):
            env.seed(self.seed)
        return env

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
