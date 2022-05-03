from logging import raiseExceptions
import os
import sys
from pathlib import Path
from typing import Optional, Union, Tuple
#import sumo_rl
from maml_rl.envs.turnpike.nets.turnpike_single.net.turnpike.VSLController2 import VSLController
from maml_rl.envs.turnpike.nets.turnpike_single.net.turnpike.metrics import getAverageDetectorFlow
from maml_rl.envs.turnpike.nets.turnpike_single.net.turnpike.metrics import getHaltingVehNum
from maml_rl.envs.turnpike.nets.turnpike_single.net.flow_generator import SumoNet
from collections import deque
import time
import shutil

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
from os import walk

LIBSUMO = 'LIBSUMO_AS_TRACI' in os.environ


def env(**kwargs):
    env = TurnpikeEnvironment(**kwargs)
    env = wrappers.AssertOutOfBoundsWrapper(env)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

WORK_PATH = os.path.abspath(os.path.join(os.getcwd()))
parallel_env = parallel_wrapper_fn(env)


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
                 mode: int = 0
                 ):
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
        
        self.duplicate_folder = WORK_PATH + '/nets/turnpike_single/net/turnpike/states/' + str(int(time.time()))
        self.duplicate_sim_state_folder(WORK_PATH + '/' + save_folder, self.duplicate_folder)
        self.save_sim_folder = self.duplicate_folder
        self.saved_sim = self._get_files_under_folder(self.save_sim_folder)
        
        self.mode = mode
        self.obs_horizon = int(obs_horizon)
        self.rwd_horizon = int(rwd_horizon)
        self.out_csv_name = self.save_sim_folder + 'res.csv'
        self.init_state = self.reset()
        # if LIBSUMO:
        #     traci.start([sumolib.checkBinary('sumo'), '-n', self._net])
        #     conn = traci
        # else:
        #     traci.start([sumolib.checkBinary('sumo'), '-n',
        #                  self._net], label='init_connection'+self.label)
        #     conn = traci.getConnection('init_connection'+self.label)
    
    def _get_files_under_folder(self, folder):
        f = []
        mypath = WORK_PATH +  folder
        for (dirpath, dirnames, filenames) in walk(mypath):
            f.extend(filenames)
            break
        return f
    
    def duplicate_sim_state_folder(self, copyfolder, parsefolder):
        shutil.copytree(copyfolder, parsefolder)
        
    def save_state(self):
        save_path = self.save_sim_folder + '/' + str(int(time.time())) + '.xml'
        # print ('Simulation state at has been saved. Please check the local file ', save_path)
        
        traci.simulation.saveState(save_path)

    def _start_simulation(self):
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
            traci.start(sumo_cmd, label=self.label)
            self.sumo = traci.getConnection(self.label)

        if self.use_gui:
            self.sumo.gui.setSchema(traci.gui.DEFAULT_VIEW, "real world")
    
    def generate_flow(self):
        self.flowGenerator.generateGaussianFlowsByProfiles()
        self.flowGenerator.generate_route_file(self._rou)
    
    def load_state_file(self):
        loc = np.random.choice(len(self.saved_sim)+1, 1)[0]
        # print (loc, len(self.saved_sim))
        if loc < len(self.saved_sim):    
            sim_state_file = WORK_PATH + self.save_sim_folder + '/' + self.saved_sim[loc]
            return sim_state_file
        else:
            return None
    
    def reset(self, seed:  Optional[int] = None, **kwargs):
        if self.run != 0:
            self.close()
            # self.save_csv(self.out_csv_name, self.run)
        self.run += 1

        if seed is not None:
            self.sumo_seed = seed
        self.generate_flow()
        self.sim_state_file = self.load_state_file()
        
        self._start_simulation()

        self.vehicles = dict()
        
        self.speed_limits_set = np.array([-1, 30, 25, 20])
        self.detectors = traci.lanearea.getIDList()
        # print (self.detectors)
        self.observations = np.zeros(len(self.detectors))
        self.vsl_controller = VSLController(*self.vsl_files)
        self.num_vsl = len(self.vsl_controller.controlzones)
        self.observation_space = Box(low=0, high=1, shape=((self.num_vsl + 1) * 2 * 2, ) )
        self.action_space = Discrete(len(self.speed_limits_set)**self.num_vsl)
        
        ''' simulation related attributes '''
        self.action_applied = 0
        self.sim_step = 0
        ''' metrics related attributes '''
        self.metrics = { 'occupancy': [ deque([], int(self.delta_step*self.obs_horizon)) for dq in range(self.num_vsl+1)],
                        'speed': [ deque([], int(self.delta_step*self.obs_horizon)) for dq in range(self.num_vsl+1)],
                    'brake_rate':deque([], int(self.delta_step*self.rwd_horizon)),
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
        
        if len(self.metrics['occupancy'][0]) < self.metrics['occupancy'][0].maxlen:
            for st_ in range( int(self.delta_step*(self.rwd_horizon + self.obs_horizon))):
                if st_ == self.delta_step * self.obs_horizon - 1:
                    self._apply_actions(action)
                    self._sumo_step(st_)
                else:
                    self._sumo_step(st_)
                
        else:
            for st_ in range( int(self.delta_step*self.rwd_horizon)):
                if st_ == 0:
                    self._apply_actions(action)
                    self._sumo_step(st_)
                else:
                    self._sumo_step(st_)
        
        observations = self._compute_observations()
        rewards = self._compute_rewards()
        done = self._compute_dones()
        info = self._compute_info()
        return observations, rewards, done, info 

    def _get_metrics(self):
        for z, CZ in enumerate(self.vsl_controller.e2Ref):
            occ_measures = []
            vel_measures = []
            E2s = self.vsl_controller.e2Ref[CZ]
            for detID in E2s:
                occ_measures.append(traci.lanearea.getLastStepOccupancy(detID))
                vel_measures.append(traci.lanearea.getLastStepMeanSpeed(detID))
            self.metrics['occupancy'][z].append(np.mean(occ_measures))
            self.metrics['speed'][z].append(np.mean(vel_measures))
        self.metrics['flow'].append(getAverageDetectorFlow())
        # self.metrics['brake_rate'].append(traci.simulation.getEndingTeleportNumber())
        self.metrics['brake_rate'].append(getHaltingVehNum())


    def _sumo_step(self, i):
        self.sumo.simulationStep()
        self.vsl_controller._search_and_apply()
        self._get_metrics() 
        # self.check_vehicle_maxspeed()
       
    
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

        '''
        check if vsl is applied
        '''
        #self.check_vsl()
        
        # print ('Speed limit updated!')
    
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
        if self.mode == 0:
            flows = np.array(self.metrics['flow'])
            r = np.sum(flows) / (self.delta_time) * 3600 
        elif self.mode == 1:
            # brakerates = np.array(self.metrics['brake_rate']) / traci.vehicle.getIDCount()
            brakerates = np.array(self.metrics['brake_rate'])
            r = np.average(brakerates ) / (self.delta_time) * 3600 
            # print (r)
        return r
    
    def _compute_dones(self):
        dones = False
        if self.sim_time > self.sim_max_time:
            dones = True
        return dones

    def _compute_info(self):
        return self.metrics

    # def _compute_step_info(self, metrics):
    #     return metrics
    
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
    
    
    def check_vsl(self):
        print (self.sim_time, self.vsl_controller.vslActions)



if __name__ == '__main__':   
    
    f = []
    
    mypath = WORK_PATH + '/saved'
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    
    source_dir = r'D:\Productivity\PythonProjects\sumo-rl-2\nets\turnpike_single/net/turnpike/saved'
    destin_dir = r'D:\Productivity\PythonProjects\sumo-rl-2\nets/turnpike_single/net/turnpike/'+str(int(time.time()))
    shutil.copytree(source_dir, destin_dir)
    
    x = deque([], 3)
    x.append(5)
    x
    
