# -*- coding: utf-8 -*-
import sumolib
import numpy as np
import random
import os
import xml.etree.cElementTree as ET


SIM_BEGIN = 0
SIM_END = 3599
WORK_PATH = os.path.abspath(os.path.join(os.getcwd(), ".."))

class SumoNet:        
    def get_edges_info(self):
        edges_info = {}
        for i, edge in enumerate(self.edges):
            ''' exclude all internal edges '''
            if not (edge.hasAttribute('attr_from') and edge.hasAttribute('to')):
                continue
            ''' pick up information from xml '''
            edges_info[edge.id] = {
                "from": edge.attr_from,
                "to": edge.to,
                "no.lanes": len(edge._child_dict["lane"]),
                "lanes":{}
                }
            ''' build a dictionary about information of each lane '''
            lanes_info = {}
            for lane_index, info in enumerate(edge._child_dict["lane"]):
                lanes_info[lane_index] = {
                    "id": info.id,
                    "allow": info.allow,
                    "length": float(info.length),
                    "speed": float(info.speed)
                    }
            edges_info[edge.id]["lanes"] = lanes_info
        self.edges_info = edges_info
        
    def get_nodes_info_from_edges(self):
        ''' this function is to retrieve nodes and the links they connect. ''' 
        nodes_info = {}
        if self.edges_info == {}:
            raise ("Please retrieve edges information first!")
        for edge_id in self.edges_info:
            edge_info = self.edges_info[edge_id]
            from_node, to_node = edge_info['from'], edge_info['to']
            ''' create a new key if node_id is not included in the dictionary '''
            if from_node not in nodes_info.keys():
                nodes_info[from_node] = {'in':[], 'out':[]}
            if to_node not in nodes_info.keys():
                nodes_info[to_node] = {'in':[], 'out':[]}
            ''' the out_node is the startpoint of link, while the in_node is the endpoint of link '''
            nodes_info[from_node]['out'].append(edge_id)
            nodes_info[to_node]['in'].append(edge_id)
        self.nodes_info = nodes_info
        
    def search_connected_graphs(self):
        # print ("Checking all connected components under network ...")
        Conn_ = {}
        ''' depth first search recursion '''
        def DFS(edge, start):
            if edge not in Conn_.keys():
                if edge == start:
                    Conn_[edge] = []
                else:
                    Conn_[edge] = [start]
            else:
                if start not in Conn_[edge]:
                    Conn_[edge].append(start)
            to_node = self.edges_info[edge]['to']
            next_Edges = self.nodes_info[to_node]['out']
            for next_edge in next_Edges:
                DFS(next_edge, start)
        ''' label if an edge can be reached from a given startpoint '''
        for start in self.start_edges:
            DFS(start, start)
        self.Conn_ = Conn_
        
    def get_start_end_edges(self):
        start_edges = []
        end_edges = []
        if self.edges_info == {} or self.nodes_info == {}:
            raise ("Please retrieve edges and nodes information first!")
        for node in self.nodes_info:
            len_in = len(self.nodes_info[node]['in'])
            len_out = len(self.nodes_info[node]['out'])
            if len_in == 0 and len_out != 0:
                start_edges += self.nodes_info[node]['out']
            if len_in != 0 and len_out == 0:
                end_edges += self.nodes_info[node]['in']
        self.start_edges = start_edges
        self.end_edges = end_edges
        
    def get_info(self):
        self.get_edges_info()   
        # print ('---------------------', self.edges_info)
        self.get_nodes_info_from_edges()
        self.get_start_end_edges()
        self.search_connected_graphs()
        
    def __init__(self, path:str):
        self.net = sumolib.net.readNet(path)
        self.nodes = sumolib.xml.parse(path, ['node'])
        self.edges = sumolib.xml.parse(path, ['edge'])
        self.connections = sumolib.xml.parse(path, ['connection'])
        self.edges_info = {}
        self.nodes_info = {}
        self.start_edges = []
        self.end_edges = []
        self.Conn_ = {}
        self.filepath = path
        self.get_info()
    
        
    def generate_OD_by_Gaussian(self, flow_mean=150, flow_sd=100):
        # print ("Generating random traffics by O-D ...")
        if self.Conn_ == {}:
            raise("Please get network connectivity beforehand!")
        if flow_mean < flow_sd:
            raise("The standard deviation of flows must not be greater than the mean flow!")
        OD = np.zeros(shape=(len(self.start_edges), len(self.end_edges)))
        for o in range(OD.shape[0]):
            for d in range(OD.shape[1]):
                start = self.start_edges[o]
                end = self.end_edges[d]
                ''' check O-D matrix integrity with a selected end edge '''
                # if end == '866596939#1':
                #     print(("start %s end %s: %s")%(start, end, start in self.Conn_[end] ))
                
                OD[o][d] = 0 if start not in self.Conn_[end] else int(max(np.random.normal(flow_mean, flow_sd, 1)[0], 20))
        self.OD = OD
        
    def generate_OD_by_Gaussian_w_multipler(self, flow_mean=3000, flow_sd=100):
        # print ("Generating random traffics by O-D ...")
        if self.Conn_ == {}:
            raise("Please get network connectivity beforehand!")
        if flow_mean < flow_sd:
            raise("The standard deviation of flows must not be greater than the mean flow!")
        OD = np.zeros(shape=(len(self.start_edges), len(self.end_edges)))
        for o in range(OD.shape[0]):
            for d in range(OD.shape[1]):
                multiplier = 1
                start = self.start_edges[o]
                end = self.end_edges[d]
                multiplier *= self.edges_info[start]['no.lanes']

                ''' check O-D matrix integrity with a selected end edge '''
                # if end == '866596939#1':
                #     print(("start %s end %s: %s")%(start, end, start in self.Conn_[end] ))
                
                OD[o][d] = 0 if start not in self.Conn_[end] else int(max(np.random.normal(flow_mean, flow_sd, 1)[0], 20)* multiplier)
        self.OD = OD
        
    def generateGaussianFlowsByProfiles(self, profiles=np.array([
                                                        [3000, 400],
                                                        [500, 100],
                                                        [1000, 200],
                                                        [2000, 200]
                                                        ]
                                                )
                                         ):
        rand_index = np.random.choice(profiles.shape[0], 1)[0]
        flow_mean, flow_sd = profiles[rand_index]
        # print ("Generating random traffics by O-D ...")
        if self.Conn_ == {}:
            raise("Please get network connectivity beforehand!")
        if flow_mean < flow_sd:
            raise("The standard deviation of flows must not be greater than the mean flow!")
        OD = np.zeros(shape=(len(self.start_edges), len(self.end_edges)))
        for o in range(OD.shape[0]):
            for d in range(OD.shape[1]):
                multiplier = 1
                start = self.start_edges[o]
                end = self.end_edges[d]
                multiplier *= self.edges_info[start]['no.lanes']

                ''' check O-D matrix integrity with a selected end edge '''
                # if end == '866596939#1':
                #     print(("start %s end %s: %s")%(start, end, start in self.Conn_[end] ))
                
                OD[o][d] = 0 if start not in self.Conn_[end] else int(max(np.random.normal(flow_mean, flow_sd, 1)[0], 20)* multiplier)
        # print(OD)
        self.OD = OD    
        print (OD)
    
    def generate_route_file(self, filename, multiclass=False):
        # print ("Building route xml file " + filename)
        if self.OD is None:
            raise("OD Matrix must be defined beforehand!")
        optParser = sumolib.options.ArgumentParser()
        options = optParser.parse_args()
        options.routefile = filename
        with open(options.routefile, 'w') as flows:
            flows.write('<flows>\n')
            for o, start in enumerate(self.start_edges):
                for d, end in enumerate(self.end_edges):
                    if not self.OD[o][d]:
                        continue
                    flows.write(('        <interval begin="%s" end="%s">\n' ) % (str(SIM_BEGIN), str(SIM_END)))
                    flows.write(('                <flow arrivalLane="%s" departLane="%s" from="%s" to="%s" number="%s" id="%s"/>\n') %("random", "random", start, end, int(self.OD[o][d]), str(start+"to"+end) ))
                    flows.write('        </interval>\n')
            flows.write('</flows>\n')
        print (filename)
 
    

class FlowGenerator:
    def __init__(self, Conn, start_edges, end_edges, edges_info):
        self.OD = None
        self.Conn_ = Conn
        self.start_edges = start_edges
        self.end_edges = end_edges
        self.edges_info = edges_info
        
    def generateGaussianFlowsByProfiles(self, profiles=np.array([
                                                        [3000, 400],
                                                        [500, 100],
                                                        [1000, 200],
                                                        [2000, 200]
                                                        ]
                                                )
                                         ):
        rand_index = np.random.choice(profiles.shape[0], 1)[0]
        flow_mean, flow_sd = profiles[rand_index]
        # print ("Generating random traffics by O-D ...")
        if self.Conn_ == {}:
            raise("Please get network connectivity beforehand!")
        if flow_mean < flow_sd:
            raise("The standard deviation of flows must not be greater than the mean flow!")
        OD = np.zeros(shape=(len(self.start_edges), len(self.end_edges)))
        for o in range(OD.shape[0]):
            for d in range(OD.shape[1]):
                multiplier = 1
                start = self.start_edges[o]
                end = self.end_edges[d]
                multiplier *= self.edges_info[start]['no.lanes']

                ''' check O-D matrix integrity with a selected end edge '''
                # if end == '866596939#1':
                #     print(("start %s end %s: %s")%(start, end, start in self.Conn_[end] ))
                
                OD[o][d] = 0 if start not in self.Conn_[end] else int(max(np.random.normal(flow_mean, flow_sd, 1)[0], 20)* multiplier)
        # print(OD)
        self.OD = OD    
        # print (OD)
    
    def generate_route_file(self, filename, SIM_BEGIN=0, SIM_END=3600, multiclass=False):
        # print ("Building route xml file " + filename)
        if self.OD is None:
            raise("OD Matrix must be defined beforehand!")
        flows = ET.Element("flows")
        for o, start in enumerate(self.start_edges):
            for d, end in enumerate(self.end_edges):
                if not self.OD[o][d]:
                    continue
                interval = ET.SubElement(flows, "interval", begin=str(SIM_BEGIN), end=str(SIM_END))
                
                flow = ET.SubElement(interval, "flow", attrib={'arrivalLane': "random", 
                                                               'departLane': "random",
                                                               'from': start,
                                                               'to': end,
                                                               'number': str(int(self.OD[o][d])),
                                                               'id': str(start+"to"+end)
                                                               })
        tree = ET.ElementTree(flows)
        tree.write(filename)
        

# def generateGaussianFlowsByProfiles(OD, Conn, start_edges, end_edges, edges_info, profiles=np.array([
#                                                     [3000, 400],
#                                                     [500, 100],
#                                                     [1000, 200],
#                                                     [2000, 200]
#                                                     ]
#                                             )
#                                      ):
#     rand_index = np.random.choice(profiles.shape[0], 1)[0]
#     flow_mean, flow_sd = profiles[rand_index]
#     # print ("Generating random traffics by O-D ...")

#     if flow_mean < flow_sd:
#         raise("The standard deviation of flows must not be greater than the mean flow!")
#     OD = np.zeros(shape=(len(start_edges), len(end_edges)))
#     for o in range(OD.shape[0]):
#         for d in range(OD.shape[1]):
#             multiplier = 1
#             start = start_edges[o]
#             end = end_edges[d]
#             multiplier *= edges_info[start]['no.lanes']

#             ''' check O-D matrix integrity with a selected end edge '''
#             # if end == '866596939#1':
#             #     print(("start %s end %s: %s")%(start, end, start in self.Conn_[end] ))
            
#             OD[o][d] = 0 if start not in Conn[end] else int(max(np.random.normal(flow_mean, flow_sd, 1)[0], 20)* multiplier)
#     # print(OD)
#     return OD    
        
# def generate_route_file(filename, OD, start_edges, end_edges, multiclass=False):
#     if OD is None:
#         raise("OD Matrix must be defined beforehand!")
#     optParser = sumolib.options.ArgumentParser()
#     options = optParser.parse_args()
#     options.routefile = filename
#     with open(options.routefile, 'w') as flows:
#         flows.write('<flows>\n')
#         for o, start in enumerate(start_edges):
#             for d, end in enumerate(end_edges):
#                 if not OD[o][d]:
#                     continue
#                 flows.write(('        <interval begin="%s" end="%s">\n' ) % (str(SIM_BEGIN), str(SIM_END)))
#                 flows.write(('                <flow arrivalLane="%s" departLane="%s" from="%s" to="%s" number="%s" id="%s"/>\n') %("random", "random", start, end, int(OD[o][d]), str(start+"to"+end) ))
#                 flows.write('        </interval>\n')
#         flows.write('</flows>\n')     
    
            
        
if __name__ == "__main__":
    0
    
    
    path = WORK_PATH + "/net/turnpike/turnpike.single.net.xml"
    us101_net = SumoNet(path)
    fg = FlowGenerator(us101_net.Conn_, us101_net.start_edges, us101_net.end_edges, us101_net.edges_info)
    fg.generateGaussianFlowsByProfiles()
    tree = fg.generate_route_file('1.net.xml')
    tree.write('1.net.xml')
    
    # us101_net.edges_info
    # us101_net.get_info()
    # us101_net.generate_OD_by_Gaussian_w_multipler()
    # us101_net.generate_route_file(filename=WORK_PATH + "/net/turnpike/turnpike.single.rou.xml")
    
    # od = us101_net.OD
    
    # od[6][5], start_edges[6], end_edges[5]
        
    
    
    # conn['973926051']
    
    # optParser = sumolib.options.ArgumentParser()
    # options = optParser.parse_args()
    # options.routefile = "data/test.rou.xml"
    
    # with open(options.routefile, 'w') as flows:
    #     flows.write('<flows>\n')
    #     flows.write('sss')
    
    
    ''' run the following tests '''
    # path = "data/us101.net.xml"
    # net = sumolib.net.readNet("data/us101.net.xml")
    # edges = sumolib.xml.parse(path, ['edge'])
    
    # sample = None
    # count_from = 0
    # count_to = 0
    # for i, edge in enumerate(edges):
    #     if edge.hasAttribute('attr_from') and edge.hasAttribute('to'):
    #         dict_ = edge._child_dict
    #         break
    
    # lane_info_ = float(dict_["lane"][0].speed)
        
    
    
    