# -*- coding: utf-8 -*-
import traci
import pickle
from nets.turnpike_single import sumo
import numpy as np

class VSLController:
    def __init__(self, vslfile = '/turnpike_vsl_zones', bufferfile='/turnpike_buffer_zones'):
        self.controlzones = pickle.load(open(sumo.WORK_PATH + vslfile,'rb'))
        self.bufferzones = pickle.load(open(sumo.WORK_PATH + bufferfile,'rb'))
        self.vslActions = np.full(len(self.controlzones), -1)
        
    def _update_speed_limit(self, speedlimits):
        if len(speedlimits) != len(self.controlzones):
            raise ("The shape of action space doesn't match the number of controllers!")
        self.vslActions = speedlimits
        
    def _if_vehicle_in_zone(self, vehiclePos, Zone):
        '''
        Parameters
        ----------
        pos : tuple<roadID, pos>
        Zone : list of segments list<roadID, startPos, endPos>
        Returns
        -------
        in_zone: boolindicator if vehicle is within this vsl zone
        '''
        for sgmt_ in Zone:
            # print (sgmt_)
            if vehiclePos[0]==sgmt_[0] and (sgmt_[1]<=vehiclePos[1]<=sgmt_[2] or sgmt_[2]<=vehiclePos[1]<=sgmt_[1]):
                return 1
        return 0
    
    def _if_vehicle_see_sign(self, vehiclePos, Zone):
        '''
        Parameters
        ----------
        pos : tuple<roadID, pos>
        Zone : list of segments list<roadID, startPos, endPos>
        Returns
        -------
        in_zone: boolindicator if vehicle is within this vsl zone
        '''
        for sgmt_ in Zone:
            # print (sgmt_)
            if vehiclePos[0]==sgmt_[0] and (sgmt_[1]<=vehiclePos[1]<=sgmt_[2] or sgmt_[2]<=vehiclePos[1]<=sgmt_[1]):
                return 1
        return 0
    
    def _recover_vehicle_speed(self, veh, lane):
        '''
        Parameters
        ----------
        pos : tuple<roadID, pos>
        '''
        traci.vehicle.setMaxSpeed(veh, traci.lane.getMaxSpeed(lane))
        
    def _search_and_apply(self):
        for veh_ in traci.vehicle.getIDList():
            vehRoad = traci.vehicle.getRoadID(veh_)
            vehLanePos = traci.vehicle.getLanePosition(veh_)
            vehLane = traci.vehicle.getLaneID(veh_)
            vehiclePos = tuple([vehRoad, vehLanePos])
            for bufferZone in self.bufferzones:
                if self._if_vehicle_in_zone(vehiclePos, bufferZone):
                    self._recover_vehicle_speed(veh_, vehLane)
            for z_, vslZone in enumerate(self.controlzones):
                if self._if_vehicle_in_zone(vehiclePos, vslZone):
                    if self.vslActions[z_] != -1:
                        traci.vehicle.setMaxSpeed(veh_, self.vslActions[z_])
                    else:
                        self._recover_vehicle_speed(veh_, vehLane)

if __name__ == '__main__':
  
    vslC = VSLController()
    len(vslC.controlzones)

