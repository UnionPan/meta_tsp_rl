# -*- coding: utf-8 -*-

import traci
import sumo
import net.pathfinder_edge
from net.flow_generator import SumoNet
from net.pathfinder_edge import PathFinder
import sumolib
import pickle

class Args:
    def __init__(self):
        self.UP_VSL_NUM = 2
        self.DOWN_VSL_NUM = 1
        self.DIST = 1000

args = Args()

class VSLControllerGenerator(SumoNet):
    
    def __init__(self, path= sumo.WORK_PATH + "/turnpike/turnpike.single.net.xml"):
        super().__init__(path)
        self.get_edges_info()
        self.get_nodes_info_from_edges()
        self.pathfinder = PathFinder(self.edges_info, self.nodes_info)
        
    def _find_onramps(self):
        ''' 
        browse through the networks to find junctions that have at least 2 'from' edges 
        '''
        _onramps = []
        for node in self.nodes_info:
            len_in = len(self.nodes_info[node]['in'])
            if len_in > 1:
                _onramps.append(node)
        self._onramps = _onramps
        return self._onramps
        
    def _get_road_type(self, junction, edge):
        '''
        check if the edge of a junction is main road
        '''
        if self.edges_info[edge]["no.lanes"] > 2:
            if edge in self.nodes_info[junction]['in']:
                return 'upstream'
            elif edge in self.nodes_info[junction]['out']:
                return 'downstream'
    
    def _get_length(self, edge):
        return list(self.edges_info[edge]['lanes'].values())[0]['length']
    
    def get_controller_locations(self):
        _targetJunctions = self._onramps
        Locs = []
        Zones = []
        for _junc in _targetJunctions:
            '''   
            
            '''
            for edge in self.nodes_info[_junc]['out']:
                if self.edges_info[edge]["no.lanes"] < 3:
                    continue
                _startpoint = tuple([edge, 0]) 
                self._search_next_upstream(_startpoint, 0, args.DIST, Zones)
                self._search_next_downstream(_startpoint, 0, args.DIST, Zones)
        self.ControlZones = Zones
        return Zones
        
                    
    def _search_next_upstream(self, checkpoint, count, dist, Zones):
        _count = count + 1
        PathsToNext = self._find_up_checkpoints(checkpoint, dist)
        for path_ in PathsToNext:
            nextcheckpoint = tuple([path_[-1][0], path_[-1][2]])
            if self.edges_info[nextcheckpoint[0]]["no.lanes"] < 3:
                continue
            Zones.append(path_)
            if _count <= args.UP_VSL_NUM:
                self._search_next_upstream(nextcheckpoint, _count, dist, Zones)
                
    def _search_next_downstream(self, checkpoint, count, dist, Zones):
        _count = count + 1
        PathsToNext = self._find_down_checkpoints(checkpoint, dist)
        for path_ in PathsToNext:
            nextcheckpoint = tuple([path_[-1][0], path_[-1][2]])
            if self.edges_info[nextcheckpoint[0]]["no.lanes"] < 3:
                continue
            Zones.append(path_)
            if _count <= args.DOWN_VSL_NUM:
                self._search_next_downstream(nextcheckpoint, _count, dist, Zones)
            
    def ramp_in_path(self, Paths):
        _Paths = []
        for path in Paths:
            for link in path:
                if self.edges_info[link[0]]['no.lanes'] < 3:
                    continue
            _Paths.append(path)
        return _Paths
            
                    
    def _find_up_checkpoints(self, checkpoint, dist):
        ''' checkpoint: tuple(name_of_edge, pos_on_edge) 
            dist: float, distance to search upstream
        '''
        Paths =  self.pathfinder.get_upstream_paths(checkpoint[0], dist, checkpoint[1])
        Paths = self.ramp_in_path(Paths)
        # return [ tuple([Paths[k][-1][0], Paths[k][-1][2]]) for k in range(len(Paths)) ]
        return Paths
            
    def _find_down_checkpoints(self, checkpoint, dist):
        ''' checkpoint: tuple(name_of_edge, pos_on_edge) 
            dist: float, distance to search upstream
        '''
        Paths =  self.pathfinder.get_downstream_paths(checkpoint[0], dist, checkpoint[1])
        Paths = self.ramp_in_path(Paths)
        # return [ tuple([Paths[k][-1][0], Paths[k][-1][2]]) for k in range(len(Paths)) ]
        return Paths
        
    def write_as_additional(self, to_file=sumo.WORK_PATH+"/turnpike/turnpike.single.E1asDetectors.additional.xml"):
        CheckPoints = []
        for path_ in ControlZones:
   
            checkpoint0 =  tuple([path_[0][0], path_[0][1]])
            checkpoint1 = tuple([path_[-1][0], path_[-1][2]])

            CheckPoints.append(checkpoint0)
            CheckPoints.append(checkpoint1)
        optParser = sumolib.options.ArgumentParser()
        options = optParser.parse_args()
        options.routefile = to_file
        with open(options.routefile, 'w') as vslfile:
            vslfile.write('<additional>\n')
            _c = 0
            for loc_ in CheckPoints:
                edge = loc_[0]
                pos_ = loc_[1]
                for lane_index in self.edges_info[edge]['lanes']:
                    laneid = edge + "_" + str(lane_index)
                    vslfile.write(('    <inductionLoop id="%s" lane="%s" pos="%s" freq="900" file="%s"/>\n' ) %
                                   (str(_c), str(laneid), str(pos_), ("turnpike.w_detectors.net.test.xml")))
                    _c += 1
            vslfile.write('</additional>\n')
            
        
    
if __name__ == '__main__':
    vsl = VSLControllerGenerator()
    juncs = vsl._find_onramps()
    nodesInfo = vsl.nodes_info
    edgesInfo = vsl.edges_info
    ControlZones = vsl.get_controller_locations()
    # vsl.write_as_additional()
    # # 

    
    
    
    
    # pickle.dump(ControlZones, open(sumo.WORK_PATH + "/turnpike/turnpike_vsl_zones", 'wb'))
    
