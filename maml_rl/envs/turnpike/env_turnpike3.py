from logging import raiseExceptions
import os
import sys
from pathlib import Path
from typing import Optional, Union, Tuple
#import sumo_rl
from maml_rl.envs.turnpike.nets.turnpike_single.net.turnpike.VSLController2 import VSLController
from maml_rl.envs.turnpike.nets.turnpike_single.net.turnpike.metrics import getAverageDetectorFlow, getAverageTTC
from maml_rl.envs.turnpike.nets.turnpike_single.net.flow_generator import SumoNet
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
from gym.spaces.space import Space
from gym.spaces import Box, Discrete

from os import walk
import time
import shutil
import random

from collections import deque

LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ
WORK_PATH = os.path.abspath(os.path.join(os.getcwd()))

class TurnpikeEnvironment(gym.Env):
    """


    """

    CONNECTION_LABEL = 0

    def __init__(self, 
                 net_file: str = 'maml_rl/envs/turnpike/nets/turnpike_single/net/turnpike/turnpike.single.net.xml',
                 route_file: str = 'maml_rl/envs/turnpike/nets/turnpike_single/net/turnpike/turnpike.single.rou.xml',
                 additional_files: str = 'maml_rl/envs/turnpike/nets/turnpike_single/net/turnpike/turnpike.single.E1asDetectors.additional.xml, maml_rl/envs/turnpike/nets/turnpike_single/net/turnpike/turnpike.single.E2asControlRecoveryPoints.additional.xml',
                 save_folder: str='maml_rl/envs/turnpike/nets/turnpike_single/net/turnpike/states/saved',
                 out_csv_name: Optional[str] = None,
                 vsl_files: list = None,
                 use_gui: bool = False,
                 virtual_display: Optional[Tuple[int, int]] = None,
                 begin_time: int = 0,
                 num_seconds: int = 600,
                 delta_time: float = 120,
                 single_agent: bool = False,
                 sumo_seed: Union[str, int] = 'random',
                 sumo_warnings: bool = False,
                 step_length: float = 0.5,
                 obs_horizon: int = 2,
                 rwd_horizon: int = 1,
                 mode: int = 0,
                 task = {}
                 ):
        super(TurnpikeEnvironment, self).__init__()
        self._task = task
        self._mode = task.get('mode', 0)
        self.spec = EnvSpec('Turnpike-v0')
        self._net = net_file
        self._rou = route_file
        self._add = additional_files
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
        self.flowGenerator = SumoNet(self._net)
        self.flowGenerator.get_info()
        self.sim_state_file = None
        
        self.duplicate_folder = 'maml_rl/envs/turnpike/nets/turnpike_single/net/turnpike/states/' + str(int(time.time()))
        self.duplicate_sim_state_folder(save_folder, self.duplicate_folder)
        self.save_sim_folder = self.duplicate_folder
        self.saved_sim = self._get_files_under_folder(WORK_PATH + '/' + self.save_sim_folder)
        self.mode = mode
        self.obs_horizon = int(obs_horizon)
        self.rwd_horizon = int(rwd_horizon)
        self.out_csv_name = self.save_sim_folder + 'res.csv'
        self.init_state = self.reset()
        
        # if LIBSUMO:
        #     traci.start([sumolib.checkBinary('sumo'), '-c', self._cfg])
        #     conn = traci
        # else:
        #     traci.start([sumolib.checkBinary('sumo'), '-c',
        #                   self._cfg], label='init_connection'+self.label)
        #     conn = traci.getConnection('init_connection'+self.label)
    def _get_files_under_folder(self, folder):
        f = []
        mypath = folder
        for (dirpath, dirnames, filenames) in walk(mypath):
            f.extend(filenames)
            break
        return f
    
    def duplicate_sim_state_folder(self, copyfolder, parsefolder):
        if not os.path.isfile(parsefolder):
            shutil.copytree(copyfolder, parsefolder)
        
    def save_state(self):
        save_path = self.save_sim_folder + '/' + str(int(time.time())) + '.xml'
        # print ('Simulation state at has been saved. Please check the local file ', save_path)
        
        traci.simulation.saveState(save_path)
    

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
        # print (self.sim_state_file)
        # print ('####################################')
        if self.sim_state_file is None:
            sumo_cmd = [self._sumo_binary,
                        '-n', self._net,
                        '-r', self._rou,
                        '-a', self._add,
                        "--step-length", str(self.step_length),
                        # '--waiting-time-memory', '10000'
                        ]
        else:
            sumo_cmd = [self._sumo_binary,
                        '-n', self._net,
                        '-r', self._rou,
                        '-a', self._add,
                        "--step-length", str(self.step_length),
                        "--load-state", self.sim_state_file,
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
        # print (sumo_cmd)
        
        if LIBSUMO:
            traci.start(sumo_cmd)
            self.sumo = traci
        else:
            try:
                traci.start(sumo_cmd, label=self.label)
            except:
                traci.switch(self.label)
                traci.close()
                traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)
            print ('###########################', self.label, '###########################')

        if self.use_gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
        '70% chance with lane closed'
        if random.uniform(0, 1) < 1.0:
            traci.lane.setDisallowed('38913001#1.83_2', ["passenger"])
            traci.lane.setDisallowed('38913001#1.83_3', ["passenger"])

    
    
    def reset(self, seed:  Optional[int] = None, **kwargs):
        # if self.run != 0:
        #     self.close()
        
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
        ''' simulation related attributes '''
        self.action_applied = 0
        self.sim_step = 0
        ''' metrics related attributes '''
        self.metrics = { 'occupancy': [ deque([], int(self.delta_step*self.obs_horizon)) for dq in range(self.num_vsl+1)],
                        'speed': [ deque([], int(self.delta_step*self.obs_horizon)) for dq in range(self.num_vsl+1)],
                    'halting': [ deque([], int(self.delta_step*self.rwd_horizon)) for dq in range(self.num_vsl+1)],
                    'flow': deque([], int(self.delta_step*self.rwd_horizon)),
                   }
        return self.observations

    @property
    def sim_time(self):
        """
        Return current simulation second on SUMO
        """
        return self.sumo.simulation.getTime()
    
    def step(self, action):
        observations = np.zeros(2)
        # print (len(self.metrics['occupancy'][0]) , '!!!!!!!!!!!!!!!!!!!!!')
        
        observations = None
        
        if len(self.metrics['occupancy'][0]) < self.metrics['occupancy'][0].maxlen:
            for st_ in range( int(self.delta_step*(self.rwd_horizon + self.obs_horizon))):
                if st_ == self.delta_step * self.obs_horizon - 1:
                    observations = self._compute_observations()
                    self._apply_actions(action)
                    self._sumo_step(st_)
                else:
                    self._sumo_step(st_)
                
        else:
            for st_ in range( int(self.delta_step*self.rwd_horizon)):
                if st_ == 0:
                    observations = self._compute_observations()
                    self._apply_actions(action)
                    self._sumo_step(st_)
                else:
                    self._sumo_step(st_)
        rewards = self._compute_rewards()
        done = self._compute_dones()
        info = self._compute_info()
        print (action, rewards)
        return observations, rewards, done, info 

    def _sumo_step(self, i):
        self.sumo.simulationStep()
        self.vsl_controller._search_and_apply()
        self._get_metrics() 
    
    def _apply_actions(self, actions):
        basenum = len(self.speed_limits_set)

        actions = np.base_repr(actions, base=basenum)
        #print(actions, 'before')
        padding_length = self.num_vsl - \
            len(actions) if actions != '0' else self.num_vsl
        actions = np.base_repr(int(actions), base=basenum, padding=padding_length)
        #print(actions[0])
        #print(actions, 'after')
        speed_limit_decisions = np.zeros(self.num_vsl)
        for i in range(self.num_vsl):
            speed_limit_decisions[i] = self.speed_limits_set[int(actions[i])]
        self.vsl_controller._update_speed_limit(speed_limit_decisions)
    
    def _compute_observations(self):
        num_d = len(self.metrics['occupancy'])
        obs = np.zeros(shape=(num_d*2, self.obs_horizon))
        for _m in range(num_d):
            for _n in range(self.obs_horizon):
                occupancies = np.array(self.metrics['occupancy'])
                speeds = np.array(self.metrics['speed'])
                mean_flow = np.mean(occupancies[_m][_n*self.delta_step: (_n+1)*self.delta_step ])
                mean_speed = np.mean(speeds[_m][_n*self.delta_step: (_n+1)*self.delta_step])
                obs[_m][_n] = min(mean_flow/100, 1)
                obs[num_d+_m][_n] = min(mean_speed/50, 1)
        return obs.flatten()
 
    def _compute_rewards(self):
        coef_ = np.array([0.25, 0.50, 0.15, 0.10])
        if self.mode == 0:
            meanspeeds = np.average(np.array(self.metrics['speed']), axis=1) 
            r = max(min( np.sum((meanspeeds-40)*coef_/40), 0), -1)
        elif self.mode == 1:
            # brakerates = np.array(self.metrics['brake_rate']) / traci.vehicle.getIDCount()
            meannumhalt = np.average(np.array(self.metrics['halting']), axis=1) 
            r =  - min(np.sum(meannumhalt/(self.delta_time)*3600/2400* coef_), 1)
        return r 
    
    def _compute_dones(self):
        dones = False
        if self.sim_time > self.sim_max_time:
            dones = True
        return dones

    def _compute_info(self):
        return self.metrics
    
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
    
