# -*- coding: utf-8 -*-
import os, sys
import traci
import tqdm
import traci.constants as tc
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
from nets.turnpike_single import sumo
from nets.turnpike_single.net.turnpike.VSLController2 import VSLController
from nets.turnpike_single.net.turnpike.metrics import getAverageDetectorFlow, getAverageTTC

class Simulation(sumo.Sumo):
    
    def initialize_vsl_controller(self, vslfile):
        return VSLController(vslfile)
    
    def __init__(self):
        super().__init__()
        self.time = 0
        self.occ = []
        self.TTCs = []
        self.avgFlows = []
        self.vsl_controller = self.initialize_vsl_controller('/net/turnpike-single/vsl2.0')
        
    def get_loop_info(self):
        loops = {}
        for loopID in traci.inductionloop.getIDList():
            loops[loopID] = {
                'lane': traci.inductionloop.getLaneID(loopID),
                'pos': traci.inductionloop.getPosition(loopID)
                }
        self.loops = loops
        
    def Run(self, capture=None, savetime=200):    
        for _step in tqdm.trange(self.n_steps, desc=self.time):
            ''' time moves forward '''
            self.time = round(self.time + self.step_length, 2)
            traci.simulationStep()
            if capture:
                self.Capture(capture)
            time = traci.simulation.getCurrentTime()
            self.detector_state_retrieve()
            
            if self.time == 100:
                self.vsl_controller._update_speed_limit(np.full(len(self.vsl_controller.controlzones), 15))
                print ('Speed limit actions applied!')
            elif self.time == 200:
                self.vsl_controller._update_speed_limit(np.full(len(self.vsl_controller.controlzones), 30))
                print ('Speed limit actions applied!')
                
                
            self.vsl_controller._search_and_apply()
            
            # if savetime and self.time == savetime:
            #     traci.simulation.saveState('net/turnpike/turnpike.single.savedstate.xml')
            #     print ('Simulation state at ', savetime, ' has been saved. Please check the local file')
            
            # self.TTCs.append(getAverageTTC())
            self.avgFlows.append(getAverageDetectorFlow())
        
    def detector_state_retrieve(self):
        occ = {}
        for loopID in traci.inductionloop.getIDList():
            occ[loopID] = traci.inductionloop.getLastStepOccupancy(loopID)
        self.occ.append(occ)

        
if __name__ == '__main__':
    print("Starting sumo service ... ")
    try:
        traci.close()
        traci.start(sumo.SUMO_CMD)
    except traci.FatalTraCIError:
        traci.start(sumo.SUMO_CMD)
    sim = Simulation()
    sim.get_loop_info()
    sim.Run()
    OCCs = sim.occ
    TTCs = sim.TTCs
    avgFlows = sim.avgFlows
    
    
    
# -*- coding: utf-8 -*-

