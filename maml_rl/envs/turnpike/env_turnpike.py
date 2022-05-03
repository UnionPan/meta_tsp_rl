from logging import raiseExceptions
import os
import sys
from pathlib import Path
from typing import Optional, Union, Tuple
#import sumo_rl
from maml_rl.envs.turnpike.nets.turnpike_single.net.turnpike.VSLController2 import VSLController
from maml_rl.envs.turnpike.nets.turnpike_single.net.turnpike.metrics import getAverageDetectorFlow, getAverageTTC

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")
import traci
import sumolib
import gym
from gym.envs.registration import EnvSpec
import numpy as np
import pandas as pd

from gym.utils import EzPickle, seeding
from pettingzoo import AECEnv
from pettingzoo.utils.agent_selector import agent_selector
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector, wrappers
from pettingzoo.utils.conversions import parallel_wrapper_fn
from gym.spaces.space import Space
from gym.spaces import Box, Discrete

LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ


def env(**kwargs):
    env = TurnpikeEnvironment(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env


parallel_env = parallel_wrapper_fn(env)


class TurnpikeEnvironment(gym.Env):
    """


    """

    CONNECTION_LABEL = 0

    def __init__(self, 
                 cfg_file: str='maml_rl/envs/turnpike/nets/turnpike_single/net/turnpike/turnpike.single.sumocfg',
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
                 step_length = 0.5,
                 task = {}
                 ):
        super(TurnpikeEnvironment, self).__init__()
        self._task = task
        self._mode = task.get('mode', 0)
        self.spec = EnvSpec('Turnpike-v0')
        self._cfg = cfg_file
        self.use_gui = use_gui 
        if self.use_gui:
            self._sumo_binary = sumolib.checkBinary('sumo-gui')
        else:
            self._sumo_binary = sumolib.checkBinary('sumo')
        
        self.virtual_display = virtual_display
        
        self.step_length = step_length
        self.begin_time = begin_time
        self.sim_max_time = num_seconds
        self.delta_time = delta_time
        self.delta_step = int(self.delta_time/self.step_length)
        self.sumo_seed = sumo_seed
        self.label = str(TurnpikeEnvironment.CONNECTION_LABEL)
        TurnpikeEnvironment.CONNECTION_LABEL += 1
        self.sumo = None
        self.vsl_files = 'maml_rl/envs/turnpike/nets/turnpike_single/net/turnpike/vsl2.0',
        self.run = 0
        self.sumo_warnings = False
        self.out_csv_name = 'turnpike.vsl2.0.save'
        seed = self.seed(1)
        self.init_state = self.reset(seed[0])
       
        
        # if LIBSUMO:
        #     traci.start([sumolib.checkBinary('sumo'), '-n', self._net])
        #     conn = traci
        # else:
        #     traci.start([sumolib.checkBinary('sumo'), '-n',
        #                  self._net], label='init_connection'+self.label)
        #     conn = traci.getConnection('init_connection'+self.label)

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def sample_tasks(self, num_tasks):
        modes = self.np_random.binomial(1, p=0.4, size=(num_tasks,))
        tasks = [{'mode': mode} for mode in modes]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._mode = task['mode']
        

    def _start_simulation(self):
        sumo_cmd = [self._sumo_binary,
                    '-c', self._cfg,
                    "--step-length", str(self.step_length),
                    # '--waiting-time-memory', '10000'
                    ]
        if self.begin_time > 0:
            sumo_cmd.append('-b {}'.format(self.begin_time))
        if self.sumo_seed == 'random':
            sumo_cmd.append('--random')
        else:
            sumo_cmd.extend(['--seed', str(self.sumo_seed)])
        if not self.sumo_warnings:
            sumo_cmd.append('--no-warnings')
        if self.use_gui:
            sumo_cmd.extend(['--start', '--quit-on-end'])
            if self.virtual_display is not None:
                sumo_cmd.extend(
                    ['--window-size', f'{self.virtual_display[0]},{self.virtual_display[1]}'])
                from pyvirtualdisplay.smartdisplay import SmartDisplay
                print("Creating a virtual display.")
                self.disp = SmartDisplay(size=self.virtual_display)
                self.disp.start()
                print("Virtual display started.")
        print (sumo_cmd)
        
        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            #traci.close()
            #self.label = '1'
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
    
    def reset(self, seed:  Optional[int] = None, **kwargs):
        #if self.run != 0:
        #    self.close()
        #    self.save_csv(self.out_csv_name, self.run)
        self.run += 1
        self.metrics = []

        if seed is not None:
            self.sumo_seed = seed
        self._start_simulation()

        self.vehicles = dict()
        
        self.speed_limits_set = np.array([-1, 30, 25, 20])
        self.detectors = traci.inductionloop.getIDList()
        #print (self.detectors)
        self.observations = np.zeros(len(self.detectors))
        self.vsl_controller = VSLController(*self.vsl_files)
        self.num_vsl = len(self.vsl_controller.controlzones)
        self.observation_space = Box(low=0, high=1, shape=(len(self.detectors), ) )
        self.action_space = Discrete(len(self.speed_limits_set)**self.num_vsl)
        return self.observations

    @property
    def sim_time(self):
        """
        Return current simulation second on SUMO
        """
        return self.sumo.simulation.getTime()
    
    def step(self, action):
        metrics = { 'occupancy': np.zeros(shape=(self.delta_step, len(self.detectors))),
                    'time to collision': np.zeros(shape=(self.delta_step)),
                    'flow': np.zeros(shape=(self.delta_step))
                   }
        
        for i in range(self.delta_step):
            if i == 0:
                self._apply_actions(action)
                self._sumo_step(metrics, i)
            else:
                self._sumo_step(metrics, i) 
        
        observations = self._compute_observations(metrics)
        rewards = self._compute_rewards(metrics)
        done = self._compute_dones()
        info = self._compute_info(metrics)
        # self.check_vehicle_maxspeed()

        return observations, rewards, done, info 

    def _sumo_step(self, metrics, i):
        self.sumo.simulationStep()
        self.vsl_controller._search_and_apply()
        for m, loopID in enumerate(traci.inductionloop.getIDList()):
            metrics['occupancy'][i][m] = traci.inductionloop.getLastStepOccupancy(loopID)
        # metrics['time to collision'][i] = getAverageTTC()
        metrics['time to collision'][i] = 0
        metrics['flow'][i] = getAverageDetectorFlow()
    
    def _apply_actions(self, action):
        basenum = len(self.speed_limits_set)

        actions = np.base_repr(action, base=basenum)
        #print(actions, 'before')
        padding_length = self.num_vsl - len(actions) if actions != '0' else self.num_vsl
        actions = np.base_repr(action, base=basenum, padding=padding_length)
        #print(actions[0])
        #print(actions, 'after')
        speed_limit_decisions = np.zeros(self.num_vsl)
        for i in range(self.num_vsl):
            speed_limit_decisions[i] = self.speed_limits_set[int(actions[i])] 
        self.vsl_controller._update_speed_limit(speed_limit_decisions)
        # print ('Speed limit updated!')
    
    def _compute_observations(self, metrics):
        #print(metrics['occupancy'].shape)
        return np.average(metrics['occupancy'], axis=0)/100

    def _compute_rewards(self, metrics):
        if self._mode == 0:
            r = np.average(metrics['flow'])
        else:
            r = np.average(metrics['time to collision'])
        return r

    #@property
    #def observation_space(self):
    #    return self.vsl_controller.controlzones
    
    #@property
    #def action_space(self):
    #    return self.vsl_controller._update_speed_limit

    

    def _compute_dones(self):
        dones = False
        if self.sim_time > self.sim_max_time:
            dones = True
        return dones

    def _compute_info(self, metrics):
        info = self._compute_step_info(metrics)
        self.metrics.append(info)
        return info

    def _compute_step_info(self, metrics):
        return metrics
    
    def close(self):
        if self.sumo is None:
            return
        if not LIBSUMO:
            traci.switch(self.label)
        traci.close()
        try:
            self.disp.stop()
        except AttributeError:
            pass 
        self.sumo = None


    def __del__(self):
        self.close()

    def render(self, mode='human'):
        if self.virtual_display:
            #img = self.sumo.gui.screenshot(traci.gui.DEFAULT_VIEW,
            #                          f"temp/img{self.sim_time}.jpg",
            #                          width=self.virtual_display[0],
            #                          height=self.virtual_display[1])
            img = self.disp.grab()
            if mode == 'rgb_array':
                return np.array(img)
            return img

    def save_csv(self, out_csv_name, run):
        if out_csv_name is not None:
            df = pd.DataFrame(self.metrics)
            Path(Path(out_csv_name).parent).mkdir(parents=True, exist_ok=True)
            df.to_csv(out_csv_name + '_conn{}_run{}'.format(self.label, run) + '.csv', index=False)
    
    
    def check_vehicle_maxspeed(self):
        for veh in traci.vehicle.getIDList():
            print (traci.vehicle.getMaxSpeed(veh))




if __name__ == '__main__':
    0
    
