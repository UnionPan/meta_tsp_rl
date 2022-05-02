import numpy as np
import gym

from gym import spaces
from gym.utils import seeding

from envs import Turnpike

class TurnpikeMeta(Turnpike):
    def __init__(self):
        super(Turnpike, self).__init__()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        raise NotImplementedError

    def reset_task(self, task):
        raise NotImplementedError

    def step(self, action):
        action = np.clip(action, -0.1, 0.1)
        assert self.action_space.contains(action)
        self._state = self._state + action

        x = self._state[0] - self._goal[0]
        y = self._state[1] - self._goal[1]
        reward = -np.sqrt(x ** 2 + y ** 2)
        done = ((np.abs(x) < 0.01) and (np.abs(y) < 0.01))

        return self._state, reward, done, {'task': self._task}
