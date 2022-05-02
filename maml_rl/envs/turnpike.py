import numpy as np
import gym

from gym import spaces
from gym.utils import seeding
from typing import Optional, Union, Tuple

from sumo_rl.environment.env_turnpike import TurnpikeEnvironmnet


class TurnpikeMeta(TurnpikeEnvironmnet):
    
    def __init__(self, 
                 cfg_file: str = 'nets/turnpike_single/net/turnpike/turnpike.single.sumocfg',
                 load_file: str = None,
                 out_csv_name: Optional[str] = None,
                 vsl_files: list = None,
                 use_gui: bool = False,
                 virtual_display: Optional[Tuple[int, int]] = None,
                 begin_time: int = 0,
                 num_seconds: int = 600,
                 delta_time: float = 120,
                 single_agent: bool = False,
                 sumo_seed: Union[str, int] = 'random',
                 sumo_warnings: bool = True,
                 step_length=0.5,
                 mode=0,
                 task={}
                 ):
        self._task =  task
        self._mode = task.get('mode', 0)
        super(TurnpikeMeta, self).__init__()
        _ = self.reset()
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        modes = self.np_random.binomial(1, p=0.4, size = (num_tasks,))
        tasks = [{'mode': mode} for mode in modes]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._mode = task['mode']

    
