import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class _GraphConv(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        #in here if our batch size equal to 64

        x = self.gconv(x).transpose(1, 2).contiguous()
        x = self.bn(x).transpose(1, 2).contiguous()
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x

class _GraphConv_no_bn(nn.Module):
    def __init__(self, adj, input_dim, output_dim, p_dropout=None):
        super(_GraphConv_no_bn, self).__init__()

        self.gconv = SemGraphConv(input_dim, output_dim, adj)

    def forward(self, x):
        #in here if our batch size equal to 64
        x = self.gconv(x).transpose(1, 2).contiguous()
        return x



class _ResGraphConv_Attention(nn.Module):
    def __init__(self, adj, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv_Attention, self).__init__()

        self.gconv1 = _GraphConv(adj, input_dim, hid_dim//2, p_dropout)


        self.gconv2 = _GraphConv_no_bn(adj, hid_dim//2, output_dim, p_dropout)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()
        self.attention = Node_Attention(output_dim)

    def forward(self, x,joint_features):
        if joint_features is None:
            residual = x
        else:
            # joint_features = joint_features.transpose(1,2).contiguous()
            x = torch.cat([joint_features,x],dim=2)
            residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)

        out = self.bn(residual.transpose(1,2).contiguous() + out)
        out = self.relu(out)

        out = self.attention(out).transpose(1,2).contiguous()
        return out


class Node_Attention(nn.Module):
    def __init__(self,channels):
        '''
        likely SElayer
        '''
        super(Node_Attention,self).__init__()
        self.avg = nn.AdaptiveAvgPool1d(1)
        self.squeeze = nn.Sequential(
            nn.Linear(channels,channels//4),
            nn.ReLU(),
            nn.Linear(channels//4,12),
            nn.Sigmoid()
        )
    def forward(self, x):
        out = self.avg(x).squeeze(2)
        out = self.squeeze(out)
        out = out[:,None,:]
        out = out
        out = (x+x*out)
        return out

class SemGraphConv(nn.Module):
    """
    Semantic graph convolution layer
    """

    def __init__(self, in_features, out_features, adj, bias=True):
        super(SemGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        #very useful demo means this is Parameter, which can be adjust by bp methods
        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.adj = nn.Parameter(torch.zeros_like(adj, dtype=torch.float))
        with torch.no_grad():
            self.adj.data = adj.clone()
        # self.m = (self.adj > 0)
        # self.e = nn.Parameter(torch.zeros(len(self.m.nonzero()), dtype=torch.float))
        # nn.init.constant_(self.e.data, 1)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):

        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])



        # adj = -9e15 * torch.ones_like(self.adj).to(input.device)

        # adj[self.m] = self.e
        # adj = F.softmax(adj, dim=1)
        adj = self.adj




        M = torch.eye(adj.size(0), dtype=torch.float).to(input.device)
        output = torch.matmul(adj * M, h0) + torch.matmul(adj * (1 - M), h1)
        if self.bias is not None:
            return output + self.bias.view(1, 1, -1)
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class EdgeAggregate(nn.Module):
    def __init__(self, input_dim_joint, input_dim_edge, output_dim):
        super().__init__()
        self.aggregate_edges = nn.Sequential(
            nn.Linear(input_dim_edge * 2, output_dim),
            nn.ReLU(True)
        )
        nn.init.normal_(self.aggregate_edges[0].weight, std=0.01)
        nn.init.constant_(self.aggregate_edges[0].bias, 0)
        if input_dim_joint > input_dim_edge:
            self.linear = nn.Linear(input_dim_joint, input_dim_edge)
            nn.init.normal_(self.linear.weight, std=0.01)
            nn.init.constant_(self.linear.bias, 0)
        else:
            self.linear = None

    def forward(self, gout, eout, sub_matrix):
        if self.linear is not None:
            gout = self.linear(gout)
        edge_input = torch.cat([eout, sub_matrix.matmul(gout)], dim=2)
        eout = self.aggregate_edges(edge_input)

        return eout


class JointAggregate(nn.Module):
    def __init__(self, input_dim_joint, input_dim_edge, output_dim, num_joints):
        super().__init__()
        self.num_joints = num_joints
        self.aggregate_joints = nn.Sequential(
            nn.Linear(input_dim_edge+input_dim_joint, output_dim),
            nn.ReLU(True)
        )
        nn.init.normal_(self.aggregate_joints[0].weight, std=0.01)
        nn.init.constant_(self.aggregate_joints[0].bias, 0)
        self.aggregate_feats = nn.Linear(input_dim_edge * 3, input_dim_edge)
        if input_dim_joint > input_dim_edge:
            self.linear = nn.Linear(input_dim_joint, input_dim_edge)
            nn.init.normal_(self.linear.weight, std=0.01)
            nn.init.constant_(self.linear.bias, 0)
        else:
            self.linear = None

    def forward(self, gout, eout, start_shift, end_shift, shift):
        if self.linear is not None:
            gout_small = self.linear(gout)
        else:
            gout_small = gout
        start_joints_feats = start_shift.matmul(gout_small)
        joints_plus = start_joints_feats + eout
        end_joints_feats = end_shift.matmul(gout_small)
        joints_minus = end_joints_feats - eout
        joints_aggregates = torch.cat([joints_plus, joints_minus], dim=1)

        joints_input = shift.matmul(joints_aggregates)
        joints_input = torch.cat([joints_input[:,:self.num_joints], 
            joints_input[:,self.num_joints:self.num_joints*2], joints_input[:,self.num_joints*2:]],dim=2)
        joints_input = self.aggregate_feats(joints_input)
        gout = self.aggregate_joints(torch.cat([gout, joints_input], dim=2))

        return gout