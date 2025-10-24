
from torch_scatter import scatter
from torch_scatter.scatter import scatter_sum
import torch
import torch.nn as nn
import inspect
import numpy as np
import numpy as np, sys, os, random, pdb, json, uuid, time, argparse
from torch.nn import functional as F
from torch.nn import Parameter
from torch.nn.init import xavier_uniform_,xavier_normal_,kaiming_uniform_,kaiming_normal_
device = torch.device("cuda:0" )
from typing import Optional, Tuple
from message_passing import MessagePassing
from torch_geometric.utils import add_self_loops, degree

def get_param(shape):
    param = Parameter(torch.Tensor(*shape));
    
    kaiming_uniform_(param.data)
    return param




class NSPN(MessagePassing):
    def __init__(self, in_channels, out_channels, num_ent, act=lambda x: x, params=None
                ):
        super(self.__class__, self).__init__()

        self.p = params
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ent =num_ent
        self.num_heads=2
        self.act = act
        self.device = None
 
        
        self.w1_out = get_param((in_channels, out_channels))
        
        self.w2_out = get_param((in_channels, out_channels))
        
        self.w3_out = get_param((in_channels, out_channels))
        
        self.w4_out = get_param((in_channels, out_channels))
        
        self.w5_out = get_param((in_channels, out_channels))
        
        self.w6_out = get_param((in_channels, out_channels))
        
        self.w7_out = get_param((in_channels, out_channels))
        
        self.w8_out = get_param((in_channels, out_channels))

        
        self.dropout = torch.nn.Dropout(0.3)
        self.bn1 = torch.nn.BatchNorm1d(in_channels)

        self.leakyrelu = nn.LeakyReLU(0.2)
        



    def forward(self, x,  edge_index):

        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        self.norm =  self.compute_norm(edge_index, self.num_ent)
        
        
        out = self.propagate(edge_index, x=x, norm=self.norm)

        
        
        out=self.gelu(out)
        
        out=self.bn1(out)
        

        out1=torch.mm(out,self.w1_out)
        out2=torch.mm(out,self.w2_out)
        out3=torch.mm(out,self.w3_out)
        out4=torch.mm(out,self.w4_out)
        out5=torch.mm(out,self.w1_out)
        out6=torch.mm(out,self.w2_out)
        out7=torch.mm(out,self.w3_out)
        out8=torch.mm(out,self.w4_out)
        
        out=(out1+out2+out3+out4+out5+out6+out7+out8)/8
        
        

        return out
    
    
    def gelu(self, x):
        return 0.5 * x * (1 + torch.tanh(0.7978845608 * (x + 0.044715 * x**3)))
    
    def message(self, x_j, norm):
        
        return norm.view(-1, 1) * x_j


    

    def compute_norm(self, edge_index, num_ent):
        row, col = edge_index
        deg = degree(col, num_ent)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        

        return norm

