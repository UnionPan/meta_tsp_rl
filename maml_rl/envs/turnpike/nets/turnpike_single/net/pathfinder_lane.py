# -*- coding: utf-8 -*-
import sumolib
import numpy as np
import copy

class LanePathFinder:
    ''' 
    PathFinder is an object to find a lane-path starting from a certain lane position 
    A path consists of lane id and the length covered 
    '''
    
    def __init__(self, path):
        self.net = sumolib.net.readNet(path)
    
    def getLanePath(self, start_lane:str, target_length:float, start_lane_pos:float, mode="down")->dict:
        _sumLen = 0
        _dictLaneDist = {}
        _dictOfDict = {}
        thisLen = self.net.getLane(start_lane).getLength()
        
        if thisLen < start_lane_pos:
            raise ("The startpoint of search is invalid: lane position > road length")
        
        start_lane_pos = min(start_lane_pos, thisLen)
        if self.NOT_A_SEGMENT(start_lane):
            raise ("The starting lane was on a ramp!")
            
        _index = 0
        
        if target_length <= thisLen - start_lane_pos:
            _dictLaneDist[start_lane] = [start_lane_pos, start_lane_pos + target_length]
            _dictOfDict[start_lane] = _dictLaneDist
        else:
            _conn = self.net.getLane(start_lane).getOutgoing()
            _connLanes = [ _conn[_l].getToLane().getID() for _l in range(len(_conn)) ]
            _dictLaneDist[start_lane] = [start_lane_pos, thisLen]
            _sumLen += thisLen - start_lane_pos
            _ = self.GET_NEXT_CONNECTED_LANE(_connLanes, _dictLaneDist, _sumLen, _dictOfDict, target_length)
        return list(_dictOfDict.values())
    
    def GET_NEXT_CONNECTED_LANE(self, connLanes, dictLaneDist, sumLen, dictOfDict, target_length):
        for _nextLane in connLanes:
            # print (_next)
            # _nextLane = _next
            # print (_nextLane)
            thisLen = self.net.getLane(_nextLane).getLength()
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
                _conn = self.net.getLane(_nextLane).getOutgoing()
                _connLanes = [ _conn[_l].getToLane().getID() for _l in range(len(_conn)) ]
                # print (_connLanes)
                self.GET_NEXT_CONNECTED_LANE(_connLanes, copy_dictLaneDist, _sumLen, dictOfDict, target_length)
            # print (copy_dictLaneDist)
            
    def NOT_A_SEGMENT(self, thisLane):
        edge = self.net.getEdge(thisLane[:-2])
        LaneNo = edge.getLaneNumber()
        if LaneNo < 2:
            return True
        
        
if __name__ == '__main__':
    path = 'D:/Productivity/Coursework/7353/project/net/turnpike/turnpike.single.net.xml'
    lpf = LanePathFinder(path)
    lanePath = lpf.getLanePath('38913000_2',30,0,'down')
    list(lanePath[0].values())[-1][1]
    
    # net = sumolib.net.readNet(path)
    # 0
    
    # net.getEdge('38913001#1.26').getLaneNumber()
