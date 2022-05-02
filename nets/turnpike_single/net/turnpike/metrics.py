# -*- coding: utf-8 -*-
import traci 
import numpy as np
from nets.turnpike_single import sumo

def getAverageDetectorFlow():
    tallied_flow = 0
    allLoops = traci.inductionloop.getIDList()
    for loopID in allLoops:
        tallied_flow += traci.inductionloop.getLastStepVehicleNumber(loopID) / sumo.STEP_LENGTH
    average_flow = tallied_flow / len(allLoops)
    return average_flow

def getAverageTTC():
    count_cf = 0
    TTCs = []
    for vehID in traci.vehicle.getIDList():
        leaderInfo = traci.vehicle.getLeader(vehID)
        if not leaderInfo:
            continue
        leaderID, dist_  = leaderInfo
        count_cf += 1
        leadVel = traci.vehicle.getSpeed(leaderID)
        folVel = traci.vehicle.getSpeed(vehID)
        ttc = dist_/(leadVel - folVel) if leadVel - folVel != 0 else 10e9
        TTCs.append(ttc)
    avgTTC = sum(TTCs)/count_cf if count_cf else 0
    return avgTTC
        
        
    
