import torch.nn as nn
import torch
import torch.nn.functional as F
from engineer.models.registry import BACKBONES
# from engineer.models.common.helper import *
from engineer.models.common.semgcn_helper import _ResGraphConv_Attention,SemGraphConv,_GraphConv
from engineer.models.common.HM import HM_Extrect
from scipy import sparse as sp
import numpy as np

@BACKBONES.register_module
class SemGCN_Heatmaps(nn.Module):
    def __init__(self, adj, num_joints, hid_dim, coords_dim, p_dropout=None):
        '''
        :param adj:  adjacency matrix using for
        :param hid_dim:
        :param coords_dim:
        :param num_layers:
        :param nodes_group:
        :param p_dropout:
        '''
        super().__init__()

        self.heat_map_generator = HM_Extrect(num_joints)
        self.num_joints = num_joints
        self.adj = self._build_adj_mx_from_edges(num_joints, adj)
        adj = self.adj_matrix

        self.gconv_input = _GraphConv(adj, coords_dim[0], hid_dim[0], p_dropout=p_dropout)
        # in here we set 4 gcn model in this part
        self.gconv_layers1 = _ResGraphConv_Attention(adj, hid_dim[0], hid_dim[1], hid_dim[0], p_dropout=p_dropout)
        self.gconv_layers2 = _ResGraphConv_Attention(adj, hid_dim[1]+256, hid_dim[2]+256, hid_dim[1]+256, p_dropout=p_dropout)
        self.gconv_layers3 = _ResGraphConv_Attention(adj, hid_dim[2]+384, hid_dim[3]+384, hid_dim[2]+384, p_dropout=p_dropout)
        self.gconv_layers4 = _ResGraphConv_Attention(adj, hid_dim[3]+512, hid_dim[4]+512, hid_dim[3]+512, p_dropout=p_dropout)


        self.gconv_output1 = SemGraphConv(384, coords_dim[1], adj)
        self.gconv_output2 = SemGraphConv(512, coords_dim[1], adj)
        self.gconv_output3 = SemGraphConv(640, coords_dim[0], adj)


    def extract_joints_features(self, merged_features, heatmaps):
        '''
        :param merged_features: three chunks of features from different layers
        :param heatmaps: heatmaps for all joints, num_joints x H x W
        return: list of features, B x num_joints x C
        '''
        joint_features = []

        for features in merged_features:
            B,C,H,W = features.shape
            hm_s = F.interpolate(heatmaps, size=[H, W])
            assert B==heatmaps.size(0)
            joint_feats_list = []
            for joint_idx in range(self.num_joints):
                hm_i = hm_s[:, joint_idx].unsqueeze(1).repeat(1,C,1,1)
                features_i = features * hm_i
                feature_vector_i = F.adaptive_avg_pool2d(features_i, 1) + F.adaptive_max_pool2d(features_i, 1)
                feature_vector_i.squeeze_()
                joint_feats_list.append(feature_vector_i)
            joint_features.append(torch.stack(joint_feats_list, dim=1))

        return joint_features


    @property
    def adj_matrix(self):
        return self.adj


    @adj_matrix.setter
    def adj_matrix(self,adj_matrix):
        self.adj = adj_matrix


    def _build_adj_mx_from_edges(self,num_joints,edge):
        def adj_mx_from_edges(num_pts, edges, sparse=True):
            edges = np.array(edges, dtype=np.int32)
            data, i, j = np.ones(edges.shape[0]), edges[:, 0], edges[:, 1]
            adj_mx = sp.coo_matrix((data, (i, j)), shape=(num_pts, num_pts), dtype=np.float32)

            # build symmetric adjacency matrix
            adj_mx = adj_mx + adj_mx.T.multiply(adj_mx.T > adj_mx) - adj_mx.multiply(adj_mx.T > adj_mx)
            adj_mx = normalize(adj_mx + sp.eye(adj_mx.shape[0]))
            if sparse:
                adj_mx = sparse_mx_to_torch_sparse_tensor(adj_mx)
            else:
                adj_mx = torch.tensor(adj_mx.todense(), dtype=torch.float)
            return adj_mx

        def sparse_mx_to_torch_sparse_tensor(sparse_mx):
            """Convert a scipy sparse matrix to a torch sparse tensor."""
            sparse_mx = sparse_mx.tocoo().astype(np.float32)
            indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
            values = torch.from_numpy(sparse_mx.data)
            shape = torch.Size(sparse_mx.shape)
            return torch.sparse.FloatTensor(indices, values, shape)

        def normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx

        return adj_mx_from_edges(num_joints, edge, False)


    def forward(self, x, heatmaps, ret_features):
        # print(f"x.shape:{x.shape}, heatmaps.shape:{heatmaps.shape}")
        for feats in ret_features:
            print(feats.shape)
        exit()
        results, _ = self.heat_map_generator(ret_features)
        
        joint_feats = self.extract_joints_features(results, heatmaps)
        # print(f"len(joint_feats):{len(joint_feats)}")
        # for i, elem in enumerate(joint_feats):
        #     print(f"elem[{i}].shape is {[elem.shape]}")
        # exit()
        
        out = self.gconv_input(x)
        out = self.gconv_layers1(out, None)
     
        out = self.gconv_layers2(out, joint_feats[0])
        out1 = self.gconv_output1(out)

        out = self.gconv_layers3(out, joint_feats[1])
        out2 = self.gconv_output2(out)

        out = self.gconv_layers4(out, joint_feats[2])
        out3 = self.gconv_output3(out)

        return [out1, out2, out3]