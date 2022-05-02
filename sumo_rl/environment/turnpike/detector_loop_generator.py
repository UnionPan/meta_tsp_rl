# -*- coding: utf-8 -*-
import traci 
import sumolib
import numpy as np
from net.flow_generator import SumoNet
import os
from net.pathfinder_edge import PathFinder

''' 
    This .py is used to generate inductive loops through the network that satisfy :
    1. there must be one and only one inductive loop located within 300-800m from each on-ramps
    2. inductive loops are not required for off-ramps
    3. 3-road junctions require loops 
        (a way to identify what junctions require loops:
         just find the nodes having at least two 'from' edges)
    4. the distance between two consecutive locations should not be less than 500m
'''

WORK_PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))
NET_PATH = WORK_PATH + "/net/turnpike/turnpike.net.xml"

class LoopGenerator(SumoNet):
    
    def __init__(self, path, ideal_distance = 200):
        super().__init__(path)
        ''' store edge and node connectivity information in the object '''
        self.get_edges_info()
        self.get_nodes_info_from_edges()
        self.get_start_end_edges()
        self.ideal_distance = ideal_distance
        
    def generate_loops_for_onramps(self):
        pass
        
    def find_junctions(self):
        ''' 
        browse through the networks to find junctions that have at least 2 'from' edges 
        '''
        _targetJunctions = []
        for node in self.nodes_info:
            len_in = len(self.nodes_info[node]['in'])
            if len_in > 1:
                _targetJunctions.append(node)
        self._targetJunctions = _targetJunctions
        
    def get_length(self, edge):
        return list(self.edges_info[edge]['lanes'].values())[0]['length']
        
    def get_ideal_loop_locations(self):
        ''' 
        find loop locations that satisfy the conditions 
        '''
        ''' find the most ideal upstream & downstream locations for each ramp '''
        locations = []
        for _junc in self._targetJunctions:
            pathfinder = PathFinder(self.edges_info, self.nodes_info)
            ''' the starting edge to search upstream '''
            start_up_edges = self.nodes_info[_junc]['in']
            ''' get upstream paths '''
            upPaths = [ pathfinder.get_upstream_paths(edge, self.ideal_distance, self.get_length(edge)) \
                       for edge in start_up_edges]
            ''' get upstream ideal locations '''
            for _Paths in upPaths:
                locations += [ tuple([_path[-1][0], _path[-1][2]]) for _path in _Paths ]
            ''' the starting edge to search downstream '''
            start_down_edges = self.nodes_info[_junc]['out']
            ''' get downstream paths '''
            downPaths = [ pathfinder.get_downstream_paths(edge, self.ideal_distance, 0) \
                       for edge in start_down_edges]
            ''' get downstream ideal locations '''
            for _Paths in downPaths:
                locations += [ tuple([_path[-1][0], _path[-1][2]]) for _path in _Paths ]
        self.ideal_locations = locations
        
    
    def merge_loop_locations(self):
        ''' merge detectors that are too close to each other (<200) '''
        detectors = self.ideal_locations
        edges_with_detectors = np.asarray(detectors)[:,0]
        for det_ed, det_pos in detectors:
            ''' search upstream 200m for other detectors '''
            pathfinder = PathFinder(self.edges_info, self.nodes_info)
            upPaths = pathfinder.get_upstream_paths(det_ed, self.ideal_distance, det_pos)
            ''' search downstream 200m for other detectors '''
            pathfinder = PathFinder(self.edges_info, self.nodes_info)
            downPaths = pathfinder.get_downstream_paths(det_ed, self.ideal_distance, det_pos)
            ''' determine if any detector is inside the dead zone '''
            ''' let's do this stupidly by using two loops '''
            for _path in upPaths:
                for _edgeAlongPath, _startPos, _endPos in _path:
                    _idx = np.where(edges_with_detectors==_edgeAlongPath)[0]
                    d = np.asarray(detectors)[_idx]
                    print (d)
        pass
    
    def write_detectors_to_net(self, multiclass=False, savepath=WORK_PATH+"/net/turnpike/turnpike.w_detectors.additional.xml"):
        optParser = sumolib.options.ArgumentParser()
        options = optParser.parse_args()
        options.routefile = savepath
        with open(options.routefile, 'w') as loopfile:
            loopfile.write('<additional>\n')
            _c = 0
            for loc_ in self.ideal_locations:
                edge = loc_[0]
                pos_ = loc_[1]
                for lane_index in self.edges_info[edge]['lanes']:
                    laneid = edge + "_" + str(lane_index)
                    loopfile.write(('    <inductionLoop id="%s" lane="%s" pos="%s" freq="900" file="%s"/>\n' ) %
                                   (str(_c), str(laneid), str(pos_), ("turnpike.w_detectors.net.test.xml")))
                    _c += 1
            loopfile.write('</additional>\n')
            
            
    # def optimize_loop_locations(self):
    #     ''' 
    #     use DP to optimize locations of detectors
    #     DP objective: minimize the total distances between detectors and their ideal locations
    #     DP constraints: distance of two consecutive detectors >= 500 
    #     DP actions: distance within [300-800], offset step distance +- 50
    #     '''
    #     offsetActions = [-200, -150, -100, -50, 0, 50, 100, 150, 200, 250, 300]
    #     n_juncs = len(self._targetJunctions)
    #     n_actions = len(offsetActions)
    #     dpSpace = np.full(float('inf'), (n_juncs, n_actions))      
        
    #     for _j, _junc in enumerate(self._targetJunctions):
    #         for _a, _act in enumerate(offsetActions):
    #             loc = self.ideal_locations 
        # pass
    
            
    
if __name__ == "__main__":
    
    loopGen = LoopGenerator(NET_PATH)
    edges = loopGen.edges
    edges_info = loopGen.edges_info
    nodes_info = loopGen.nodes_info
    loopGen.find_junctions()
    targetJunctions = loopGen._targetJunctions
    loopGen.get_ideal_loop_locations()
    ideal_locations = loopGen.ideal_locations
    loopGen.write_detectors_to_net()
    # loopGen.merge_loop_locations()
    
    
    # pathfinder = PathFinder(edges_info, nodes_info)
    # upPaths = np.asarray(pathfinder.get_upstream_paths('24794617', 800, 550))
    # downPaths = np.asarray(pathfinder.get_downstream_paths('24794617', 800, 100))
    
