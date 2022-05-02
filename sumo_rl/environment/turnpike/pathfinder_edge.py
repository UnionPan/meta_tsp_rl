# -*- coding: utf-8 -*-
import traci
import numpy as np
import copy
from net.flow_generator import SumoNet

class PathFinder:
    ''' 
    PathFinder is an object to find a lane-path starting from a certain lane position 
    A path consists of lane id and the length covered 
    '''
    
    def __init__(self, edges, nodes):
        self.edges_info = edges
        self.nodes_info = nodes
    
    def get_upstream_paths(self, start_edge:str, target_length:float, start_pos:float)->list:
        ''' 
        return a list of paths. each path: tuple([edge_id, start_pos, end_pos]) 
        '''
        ''' initial the list of paths '''
        Paths = []
        ''' initial the first path '''
        original_path = []
        
        ''' use recursion to search for upstream edges '''
        def find_upstream_edges(path, rem, _curr_edge, _curr_start_pos, Paths):
            _curr_eff_edge_len = _curr_start_pos
            _rem = rem
            _path = copy.deepcopy(path)
            if self.if_end_within(_rem, _curr_eff_edge_len):
                _end_pos = _curr_eff_edge_len - _rem
                _path.append(tuple([_curr_edge, _curr_start_pos, _end_pos]))
                Paths.append(_path)
            else:
                _path.append(tuple([_curr_edge, _curr_start_pos, 0]))
                imm_up_edges = self.get_immediate_upstream_edges(_curr_edge)
                _rem -= _curr_eff_edge_len
                for nextedge in imm_up_edges:
                    find_upstream_edges(_path, _rem, nextedge, self.get_length(nextedge), Paths)
        ''' run the recursions '''          
        find_upstream_edges(original_path, target_length, start_edge, start_pos, Paths)
        return Paths
    
    def get_downstream_paths(self, start_edge:str, target_length:float, start_pos:float)->list:
        ''' 
        return a list of paths. each path: tuple([edge_id, start_pos, end_pos]) 
        '''
        ''' initial the list of paths '''
        Paths = []
        ''' initial the first path '''
        original_path = []
        ''' use recursion to search for upstream edges '''
        def find_downstream_edges(path, rem, _curr_edge, _curr_start_pos, Paths):
            _curr_eff_edge_len = self.get_length(_curr_edge) - _curr_start_pos
            _rem = rem + start_pos
            _path = copy.deepcopy(path)
            if self.if_end_within(_rem, _curr_eff_edge_len):
                _end_pos = _rem
                _path.append(tuple([_curr_edge, _curr_start_pos, _end_pos]))
                if not self.if_ramp(_curr_edge):
                    Paths.append(_path)
            else:
                _path.append(tuple([_curr_edge, _curr_start_pos, self.get_length(_curr_edge)]))
                imm_down_edges = self.get_immediate_downstream_edges(_curr_edge)
                _rem -= _curr_eff_edge_len
                for nextedge in imm_down_edges:
                    find_downstream_edges(_path, _rem, nextedge, 0, Paths)
        ''' run the recursions '''          
        find_downstream_edges(original_path, target_length, start_edge, start_pos, Paths)
        return Paths
    
    def if_end_within(self, remaining_length:float, curr_edge_length:float)->bool:
        if remaining_length < curr_edge_length:
            return True
        else:
            return False
        
    def if_ramp(self, edge):
        if self.edges_info[edge]['no.lanes'] < 2:
            return True
        
    def get_length(self, edge):
        return list(self.edges_info[edge]['lanes'].values())[0]['length']
    
    def get_immediate_upstream_edges(self, edge):
        from_node = self.edges_info[edge]['from']
        imm_up_eds = self.nodes_info[from_node]['in']
        return imm_up_eds
    
    def get_immediate_downstream_edges(self, edge):
        to_node = self.edges_info[edge]['to']
        imm_down_eds = self.nodes_info[to_node]['out']
        return imm_down_eds
    
    

    