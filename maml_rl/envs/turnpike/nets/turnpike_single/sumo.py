# -*- coding: utf-8 -*-
import os, sys
import traci

TRACI_START = 0
BEGIN_VALUE = 0
END_VALUE = 300
STEP_LENGTH = 0.2
WORK_PATH = os.path.abspath(os.path.join(os.getcwd(), "."))
SUMO_BINARY = "sumo-gui.exe"
LOAD_PATH = 'net/turnpike/turnpike.single.savedstate.xml'
SUMO_CMD = [ SUMO_BINARY, "-c", os.path.join(WORK_PATH, "net/turnpike/turnpike.single.sumocfg"),
             "--step-length", str(STEP_LENGTH),
                "--load-state", LOAD_PATH
             # "--begin value", str(BEGIN_VALUE)
             # "--end value", str(END_VALUE)
             ]
NET_PATH = 'net/turnpike/turnpike.single.net.xml'

# LOAD_PATH = Noned


# if LOAD_PATH is not None:
#     traci.simulation.loadState(LOAD_PATH)
#     print ('Loading simulation state ', LOAD_PATH)
TRACI_START = 1

class Sumo:
    def __init__(self):
        self.begin = BEGIN_VALUE
        self.end = END_VALUE
        self.step_length = STEP_LENGTH
        self.n_steps = int((END_VALUE-BEGIN_VALUE)/STEP_LENGTH)
        # self.load_path = LOAD_PATH
        
        
        
