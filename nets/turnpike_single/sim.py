# -*- coding: utf-8 -*-
import os, sys
import traci
import tqdm
import traci.constants as tc
import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import sumo as sumo

class Simulation(sumo.Sumo):
    def __init__(self):
        super().__init__()
        self.time = 0
        self.occ = []
        
    def get_loop_info(self):
        loops = {}
        for loopID in traci.inductionloop.getIDList():
            loops[loopID] = {
                'lane': traci.inductionloop.getLaneID(loopID),
                'pos': traci.inductionloop.getPosition(loopID)
                }
        self.loops = loops
        
    def Run(self, capture=None, savetime=0):        
        for _step in tqdm.trange(self.n_steps, desc=self.time):
            ''' time moves forward '''
            self.time += self.step_length
            traci.simulationStep()
            if capture:
                self.Capture(capture)
            time = traci.simulation.getCurrentTime()
            self.detector_state_retrieve()
            if savetime and round(self.time,2) == savetime:
                traci.simulation.saveState('net/turnpike/turnpike.savedstate.xml')
                print ('Simulation state at ', savetime, ' has been saved. Please check the local file')
            
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
    occ = sim.occ
    
    
    
