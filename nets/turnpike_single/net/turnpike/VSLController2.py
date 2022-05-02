# -*- coding: utf-8 -*-
import traci
import pickle
from nets.turnpike_single import sumo
import numpy as np

class VSLController:
    def __init__(self, vslfile = '/vsl2.0'):
        self.e2Dict, self.e2Ref = pickle.load(open(sumo.WORK_PATH + vslfile,'rb'))
        self.controlzones = [ list(self.e2Ref.keys())[_i] for _i in range(len(list(self.e2Ref.keys()))) if list(self.e2Ref.keys())[_i][0] == 'C']
        self.recoveryzones = [ list(self.e2Ref.keys())[_i] for _i in range(len(list(self.e2Ref.keys()))) if list(self.e2Ref.keys())[_i][0] == 'R']
        self.vslActions = np.full(len(self.controlzones), -1)
        
    def _update_speed_limit(self, speedlimits):
        if len(speedlimits) != len(self.controlzones):
            raise ("The shape of action space doesn't match the number of controllers!")
        self.vslActions = speedlimits
    
    def _recover_vehicle_speed(self, veh, lane):
        '''
        Parameters
        ----------
        pos : tuple<roadID, pos>
        '''
        traci.vehicle.setMaxSpeed(veh, traci.lane.getMaxSpeed(lane))
        
    def _search_and_apply(self):
        for c, czone_ in enumerate(self.controlzones):
            for e2 in self.e2Ref[czone_]:
                # print (e2)
                affectedVehs = traci.lanearea.getLastStepVehicleIDs(e2)
                # print (affectedVehs)
                for veh_ in affectedVehs:
                    lane_ = traci.vehicle.getLaneID(veh_)
                    if self.vslActions[c] == -1:
                        self._recover_vehicle_speed(veh_, lane_)
                    else:
                        traci.vehicle.setMaxSpeed(veh_, min(self.vslActions[c], 
                                                             traci.lane.getMaxSpeed(lane_))
                                                  )
        for r, rzone_ in enumerate(self.recoveryzones):
            for e2 in self.e2Ref[rzone_]:
                affectedVehs = traci.lanearea.getLastStepVehicleIDs(e2)
                for veh_ in affectedVehs:
                    lane_ = traci.vehicle.getLaneID(veh_)
                    self._recover_vehicle_speed(veh_, lane_)

if __name__ == '__main__':
  
    vslC = VSLController()

