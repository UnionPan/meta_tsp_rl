# -*- coding: utf-8 -*-
import sumolib
import numpy as np
import copy

class LanePathFinder:
    ''' 
    PathFinder is an object to find a lane-path starting from a certain lane position 
    A path consists of lane id and the length covered 
    '''
    
    def __init__(self):
        pass
    
    def get_path(self, start_lane:str, target_length:float, start_lane_pos:float, mode="down")->dict:
        _sumLen = 0
        _dictLaneDist = {}
        _dictOfDict = {}
        thisLen = sumolib.Lane.getLength(start_lane)
        
        start_lane_pos = min(start_lane_pos, thisLen)
        if self.NOT_A_SEGMENT(start_lane):
            raise ("The starting lane was on a ramp!")
            
        _index = 0
        
        if target_length <= thisLen - start_lane_pos:
            _dictLaneDist[start_lane] = [start_lane_pos, start_lane_pos + target_length, _index]
            _dictOfDict[start_lane] = _dictLaneDist
        else:
            _connLanes = sumolib.lane.getLinks(start_lane)
            _dictLaneDist[start_lane] = [start_lane_pos, thisLen]
            _sumLen += thisLen - start_lane_pos
            _ = self.GET_NEXT_CONNECTED_LANE(_connLanes, _dictLaneDist, _sumLen, _dictOfDict, target_length)
        return _dictOfDict
    
    def GET_NEXT_CONNECTED_LANE(self, connLanes, dictLaneDist, sumLen, dictOfDict, target_length):
        for _next in connLanes:
            _nextLane = _next[0]
            # print (_nextLane)
            thisLen = sumolib.lane.getLength(_nextLane)
            if self.NOT_A_SEGMENT(_nextLane):
                continue
            if target_length <= thisLen + sumLen:
                # print (_nextLane)
                # print (target_length - sumLen)
                copy_dictLaneDist = copy.deepcopy(dictLaneDist)
                copy_dictLaneDist[_nextLane] = [0, target_length - sumLen]
                dictOfDict[_nextLane] = copy_dictLaneDist
            
            else:
                copy_dictLaneDist = copy.deepcopy(dictLaneDist)
                copy_dictLaneDist[_nextLane] = [0, thisLen]
                _sumLen = sumLen + thisLen
                _connLanes = sumolib.lane.getLinks(_nextLane)
                # print (_connLanes)
                self.GET_NEXT_CONNECTED_LANE(_connLanes, copy_dictLaneDist, _sumLen, dictOfDict, target_length)
            # print (copy_dictLaneDist)
            
    def NOT_A_SEGMENT(self, thisLane):
        edge = sumolib.lane.getEdgeID(thisLane)
        LaneNo = sumolib.edge.getLaneNumber(edge)
        if LaneNo < 2:
            return True
        
        
if __name__ == '__main__':
    path = 'D:/Productivity/Coursework/7353/project/net/turnpike/turnpike.single.net.xml'
    net = sumolib.net.readNet(path)
    net.getLane
    
    0