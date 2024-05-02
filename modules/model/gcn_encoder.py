
'''
# --------------------------------------------
# Graph Conv Utils
# --------------------------------------------
# Ultra Inertial Poser: Scalable Motion Capture and Tracking from Sparse Inertial Sensors and Ultra-Wideband Ranging (SIGGRAPH 2024)
# https://github.com/eth-siplab/UltraInertialPoser
# Sensing, Interaction & Perception Lab,
# Department of Computer Science, ETH Zurich
'''

import torch
import torch.nn as nn
import numpy as np
import math
import config.config as config
from modules.model.model_base import BaseModel
from torch_geometric.nn import (
    GlobalAttention,
    MessagePassing,
    Set2Set,
    global_add_pool,
    global_max_pool,
    global_mean_pool,
)
from articulate.utils.torch import *
from config.config import *
__all__ = ["ResGraphEncoder","GraphConv","Graph_JP_estimator"]
class _GraphConv(nn.Module):

    def __init__(self, in_features, out_features, adj_feat_dim, bias=True, edge_dropout = None):
        super(_GraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.adj_feat_dim = adj_feat_dim

        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.M = nn.Parameter(torch.zeros(size=(adj_feat_dim, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.M.data, gain=1.414)
        
        self.edge_b = nn.Parameter(torch.ones(adj_feat_dim,adj_feat_dim))        
        nn.init.constant_(self.edge_b, 1e-6)
        
        # self.Affine_matrix = nn.Parameter(torch.ones(size=(adj_feat_dim, adj_feat_dim), dtype=torch.float))
        # nn.init.constant_(self.M.data, 1.0)
        # nn.init.constant_(self.Affine_matrix, 1e-6)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)
        
        if edge_dropout is not None:
            self.edge_dropout = nn.Dropout(edge_dropout)
        else:
            self.edge_dropout = None

    def forward(self, input_nodes, input_edges):
        h0 = torch.matmul(input_nodes, self.W[0])
        h1 = torch.matmul(input_nodes, self.W[1])
        
        edge_features = input_edges.to(input_nodes.device) + self.edge_b.to(input_nodes.device)
        edge_features = input_edges.to(input_nodes.device) + self.edge_b.to(input_nodes.device)
        edge_features = (edge_features.permute(0,2,1) + edge_features)/2
        
        if self.edge_dropout is not None:
            edge_features = self.edge_dropout(edge_features)
        
        E = torch.eye(self.adj_feat_dim, dtype=torch.float).to(input_nodes.device)
        output = torch.matmul(edge_features * E, self.M*h0) + torch.matmul(edge_features * (1 - E), self.M*h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

        
class GraphConv(nn.Module):
    '''
    Input G = (V,E), V-Imu measurement(N,D), E-UWB measurement(distance between nodes) (N,N)
    
    '''
    def __init__(self,in_feature, out_feature, node_number=6, p_dropout=None, edge_dropout=None) -> None:
        super().__init__()

        self.gconv = _GraphConv(in_feature,out_feature,node_number,edge_dropout=edge_dropout)
        self.bn = nn.BatchNorm1d(out_feature)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None
            
    def forward(self, v, e):    
        x = self.gconv(v, e).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            #node feature drop out
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x
    
class ResGraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, node_number, p_dropout, edge_dropout):
        super(ResGraphConv, self).__init__()

        self.gconv1 = GraphConv(input_dim, hidden_dim, node_number, p_dropout, edge_dropout)
        self.gconv2 = GraphConv(hidden_dim, output_dim, node_number, p_dropout, edge_dropout)

    def forward(self, v, e):
        residual = v
        out = self.gconv1(v,e)
        out = self.gconv2(out,e)
        return residual + out
    
class ResGraphEncoder(nn.Module):
    """
    Encode Node (N,D) and Edge (N,N) into (N,D')

    """
    def __init__(self, input_dim, output_dim, hidden_dim,num_res_layer=1, node_number=6, p_dropout=None, edge_dropout=None) -> None:
        super().__init__()
        self.num_res_block = num_res_layer
        self.add_module("input_layer", GraphConv(input_dim,hidden_dim,node_number,p_dropout))
        
        for l in range(num_res_layer):
            self.add_module(f"ResGraphConv_{l}",ResGraphConv(hidden_dim,hidden_dim,hidden_dim,node_number,p_dropout,edge_dropout))
        
        self.add_module("output_layer", GraphConv(hidden_dim,output_dim,node_number,p_dropout))
             
    
    def forward(self,v,e):
        out = self._modules["input_layer"](v,e)
        for l in range(self.num_res_block):
            out = self._modules[f"ResGraphConv_{l}"](out,e)
        out = self._modules["output_layer"](out,e)
        return out

class Graph_JP_estimator(BaseModel):
    name = "GNN_JP"
    imu_m = ["vrot","vuwb"]
    model_output = ["imu_node_position"]
    """
    Given IMU + UWB data estimate orientation and position of imu nodes
    
    input (6,D_imu + D_pe) + (6,6)
    output (6, 3)
    
    TODO:
    1.estimated results: global/local(in pelvis coordinate)
    2.Since the graph network only consider one frame, how to make it leverage the temporal info
    """
    @staticmethod
    def add_args(parser):
        BaseModel.add_args(parser)
        group = parser.add_argument_group("Graph_JP_estimator Config")
        group.add_argument("--node_number", type=int, default=6, help="node(sensor) number")
        group.add_argument("--cls_embed_layers", type=int, default=1, help="number of hidden layers for cls graph embed")
        group.add_argument("--hidden_dim", type=int, default=32, help="dimension of the hidden layer")
        group.add_argument("--dropout", type=float, default=0.1)
        group.add_argument("--edge_dropout", type=float, default=0.1)
        
    @staticmethod
    def get_config(args):
        config_str = f"{args.network}-input{Graph_JP_estimator.imu_m},-output{Graph_JP_estimator.model_output}"
        config_str += f"-node_number{args.node_number}-layer_num{args.cls_embed_layers}-hidden_dim{args.hidden_dim}"
        return config_str

    def __init__(self, args, pe_dim = -1) -> None:
        super().__init__(args)
        
        self.imu_start_idx = 0 if "vacc" in self.imu_m else config.INPUT_DATA_SIZE["vacc"]
        self.imu_end_idx = config.INPUT_DATA_SIZE["vacc"] if "vrot" not in self.imu_m else config.INPUT_DATA_SIZE["vacc"] + config.INPUT_DATA_SIZE["vrot"]
        self.in_dim = (self.imu_end_idx - self.imu_start_idx)//config.IMU_NUM
        self.node_number = args.node_number
        self.output_dim = 3
        self.gconv = ResGraphEncoder(max(self.in_dim,1),self.output_dim,
                                     hidden_dim=args.hidden_dim,
                                     num_res_layer=args.cls_embed_layers,
                                     node_number=args.node_number,
                                     p_dropout=args.dropout,
                                     edge_dropout=0.1)

    def _forward(self,v,e):
        bz,seq_l,n,dim = v.size()
        
        e = e.view(-1, self.node_number, self.node_number)
        v = v.view(-1, self.node_number, dim)
        
        output = self.gconv(v, e).view(bz,seq_l,n,-1)

        return output
    
    def forward(self,x):
        x, *res = list(zip(*x))
        bz = len(x)
        seq_length = x[0].size(0)
        if self.in_dim == 0:
            x_imu = torch.ones((bz,seq_length,self.node_number,1),device=x[0].device)
        else:
            x_imu = torch.stack([_[:,self.imu_start_idx:self.imu_end_idx] for _ in list(x)]).view(bz,-1,self.node_number,self.in_dim)
        
        x_uwb = torch.stack([_[:,-config.INPUT_DATA_SIZE["vuwb"]:] for _ in list(x)]).view(bz,-1,self.node_number,self.node_number)
        
        leaf_joint_pos = self._forward(x_imu,1 / (1 + x_uwb)).reshape(bz, -1, self.node_number * self.output_dim)
        
        leaf_joint_pos = leaf_joint_pos[:,:,self.output_dim:]
        
        return [leaf_joint_pos[i] for i in range(bz)]

