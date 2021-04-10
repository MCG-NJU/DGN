import torch.nn as nn
import torch
import torch.nn.functional as F
from engineer.models.registry import BACKBONES
# from engineer.models.common.helper import *
from engineer.models.common.semgcn_helper import _ResGraphConv_Attention,SemGraphConv,_GraphConv,\
    EdgeAggregate, JointAggregate
from engineer.models.common.HM import HM_Extrect
from scipy import sparse as sp
import numpy as np

@BACKBONES.register_module
class SemGCN_Heatmaps(nn.Module):
    def __init__(self, adj, edge_adj, num_joints, num_edges, hid_dim, coords_dim, p_dropout=None):
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
        self.num_edges = num_edges

        self.register_buffer('sub_matrix', self._build_sub_matrix(adj))
        self.register_buffer('avg_matrix', self._build_avg_matrix(adj))
        # self.register_buffer('shift', self._build_shift_matrix(adj))

        ms, me = self._build_joints_shift_matrix(adj)
        self.register_buffer('start_shift', ms)
        self.register_buffer('end_shift', me)

        ev, ve = self._build_ev_ve_matrix()
        ev, ve = torch.tensor(ev, dtype=torch.float), torch.tensor(ve, dtype=torch.float)
        self.register_buffer('ev_shift', ev)
        self.register_buffer('ve_shift', ve)

        self.adj = self._build_adj_mx_from_edges(num_joints, adj)
        self.edge_adj = self._build_adj_mx_from_edges(num_edges, edge_adj)

        adj = self.adj_matrix
        edge_adj = self.edge_adj_matrix

        self.gconv_input = _GraphConv(adj, coords_dim[1], hid_dim[0], p_dropout=p_dropout)
        self.econv_input = _GraphConv(edge_adj, coords_dim[1], hid_dim[0], p_dropout=p_dropout)
        self.aggregate_edges = EdgeAggregate(hid_dim[0], hid_dim[0])
        self.aggregate_joints = JointAggregate(hid_dim[0], hid_dim[0], hid_dim[0], self.num_joints)
        # in here we set 4 gcn model in this part
        self.gconv_layers1 = _ResGraphConv_Attention(adj, hid_dim[0], hid_dim[1], hid_dim[0], p_dropout=p_dropout)
        self.econv_layers1 = _ResGraphConv_Attention(edge_adj, hid_dim[0], hid_dim[1], hid_dim[0], p_dropout=p_dropout)

        self.aggregate_edges1 = EdgeAggregate(hid_dim[1], hid_dim[1])
        self.aggregate_joints1 = JointAggregate(hid_dim[1], hid_dim[1], hid_dim[1], self.num_joints)

        self.gconv_layers2 = _ResGraphConv_Attention(adj, hid_dim[1]+256, hid_dim[2]+256, hid_dim[1]+256, p_dropout=p_dropout)
        self.econv_layers2 = _ResGraphConv_Attention(edge_adj, hid_dim[1], hid_dim[2], hid_dim[1], p_dropout=p_dropout)

        self.aggregate_edges2 = EdgeAggregate(384, hid_dim[2])
        self.aggregate_joints2 = JointAggregate(384, hid_dim[2], hid_dim[2]+256, self.num_joints)

        self.gconv_layers3 = _ResGraphConv_Attention(adj, hid_dim[2]+384, hid_dim[3]+384, hid_dim[2]+384, p_dropout=p_dropout)
        self.econv_layers3 = _ResGraphConv_Attention(edge_adj, hid_dim[2], hid_dim[3], hid_dim[2], p_dropout=p_dropout)

        self.aggregate_edges3 = EdgeAggregate(512, hid_dim[3])
        self.aggregate_joints3 = JointAggregate(512, hid_dim[3], hid_dim[3]+384, self.num_joints)

        self.gconv_layers4 = _ResGraphConv_Attention(adj, hid_dim[3]+512, hid_dim[4]+512, hid_dim[3]+512, p_dropout=p_dropout)
        self.econv_layers4 = _ResGraphConv_Attention(edge_adj, hid_dim[3], hid_dim[4], hid_dim[3], p_dropout=p_dropout)

        self.gconv_output1 = SemGraphConv(384, coords_dim[1], adj)
        self.econv_output1 = SemGraphConv(hid_dim[2], coords_dim[1], edge_adj)
        self.gconv_output2 = SemGraphConv(512, coords_dim[1], adj)
        self.econv_output2 = SemGraphConv(hid_dim[3], coords_dim[1], edge_adj)
        self.gconv_output3 = SemGraphConv(640, coords_dim[0], adj)
        self.econv_output3 = SemGraphConv(hid_dim[4], coords_dim[0], edge_adj)


    def _build_shift_matrix(self, adj):
        assert len(adj)==self.num_edges
        shift = torch.zeros((3*self.num_edges, 2*self.num_edges))
        idx_dict = {}
        count_dict = {}
        for i in range(12):
            count_dict[i] = 0
        for idx, e in enumerate(adj):
            i,j = e
            
            # process i
            base = count_dict[i] * self.num_joints
            count_dict[i] += 1
            new_idx = base + i
            assert new_idx not in idx_dict
            idx_dict[new_idx] = self.num_joints + idx
            
            # process j
            base = count_dict[j] * self.num_joints
            count_dict[j] += 1
            new_idx = base + j
            assert new_idx not in idx_dict
            idx_dict[new_idx] = idx
        for i,j in idx_dict.items():
            shift[i,j]=1

        return shift


    def _build_sub_matrix(self, adj):
        """
        build a matrix to do vertices' features subtraction, according to vertices' adjacency
        in order to generate information for edges to aggregate
        """
        sub_matrix = torch.zeros((self.num_edges, self.num_joints))
        for idx, e in enumerate(adj):
            sub_matrix[idx][e[0]] = -1
            sub_matrix[idx][e[1]] = 1

        return sub_matrix


    def _build_avg_matrix(self, adj):
        """
        build a matrix to do average operation, to average scores
        """
        avg_matrix = torch.zeros((self.num_edges, self.num_joints))
        for idx, e in enumerate(adj):
            avg_matrix[idx][e[0]] = 0.5
            avg_matrix[idx][e[1]] = 0.5

        return avg_matrix


    def _build_joints_shift_matrix(self, adj):
        start_joints_shift = torch.zeros((self.num_edges, self.num_joints))
        end_joints_shift = torch.zeros((self.num_edges, self.num_joints))
        for idx, e in enumerate(adj):
            start_joints_shift[idx][e[0]] = 1
            end_joints_shift[idx][e[1]] = 1
        
        return start_joints_shift, end_joints_shift


    def _build_ev_ve_matrix(self):
        ev_shift_idx = [1,2,4,3,5,6,7,7,8,10,9,11]
        ve_shift_idx = [0,0,2,1,3,0,1,6,6,8,7,9]

        ev_shift_arr = np.zeros((12,12))
        for idx, vec in zip(ev_shift_idx, ev_shift_arr):
            vec[idx] = 1

        ve_shift_arr = np.zeros((12,12))
        for idx, vec in zip(ve_shift_idx, ve_shift_arr):
            vec[idx] = 1

        return ev_shift_arr, ve_shift_arr


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
            for joint_idx in range(5, self.num_joints+5):
                hm_i = hm_s[:, joint_idx].unsqueeze(1) #.repeat(1,C,1,1)
                max_result = torch.max(hm_i.view(B,1,-1),dim=2)
                hm_i = hm_i / max_result.values.view(B,1,1,1)
                features_i = features * hm_i
                feature_vector_i = F.adaptive_avg_pool2d(features_i, 1) + F.adaptive_max_pool2d(features_i, 1)
                feature_vector_i.squeeze_()
                joint_feats_list.append(feature_vector_i)
            joint_features.append(torch.stack(joint_feats_list, dim=1))

        return joint_features


    @property
    def adj_matrix(self):
        return self.adj


    @property
    def edge_adj_matrix(self):
        return self.edge_adj


    @adj_matrix.setter
    def adj_matrix(self,adj_matrix):
        m = (adj_matrix == 0)
        assert len(m) == 108, f"len(m) is {len(m)}"
        adj_matrix[m] = 0.001
        self.adj = adj_matrix


    @edge_adj_matrix.setter
    def edge_adj_matrix(self,edge_adj_matrix):
        m = (edge_adj_matrix==0)
        edge_adj_matrix[m] = 0.001
        self.edge_adj = edge_adj_matrix


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


    def forward(self, x, heatmaps, ret_features, gt=None):
        # print(f"x.shape:{x.shape}, heatmaps.shape:{heatmaps.shape}")
        # for feats in ret_features:
        #     print(feats.shape)
        # exit()
        results, _ = self.heat_map_generator(ret_features)
        
        joint_feats = self.extract_joints_features(results, heatmaps)
        # print(f"len(joint_feats):{len(joint_feats)}")
        # for i, elem in enumerate(joint_feats):
        #     print(f"elem[{i}].shape is {[elem.shape]}")
        # exit()

        assert x.shape[2] == 3
        assert x.dim() == 3
        if gt is not None:
            assert gt.dim() == 3

        # compute detected edges
        y_coords = self.sub_matrix.matmul(x[:,:,:2])
        # y_score = self.avg_matrix.matmul(x[:,:,2].unsqueeze(dim=2))
        # y = torch.cat([y_coords, y_score], dim=2)
        
        gout = self.gconv_input(x[:,...,:2])
        eout = self.econv_input(y_coords)

        # aggregation
        eout1 = self.aggregate_edges(gout, eout, self.start_shift, self.end_shift)       
        gout1 = self.aggregate_joints(gout, eout, self.ev_shift, self.ve_shift)

        gout = self.gconv_layers1(gout1, None)
        eout = self.econv_layers1(eout1, None)

        # aggregation
        eout1 = self.aggregate_edges1(gout, eout, self.start_shift, self.end_shift)
        gout1 = self.aggregate_joints1(gout, eout, self.ev_shift, self.ve_shift)
     
        gout = self.gconv_layers2(gout1, joint_feats[0])
        eout = self.econv_layers2(eout1, None)

        joints_out1 = self.gconv_output1(gout)
        edges_out1 = self.econv_output1(eout)
        
        # aggregation
        eout1 = self.aggregate_edges2(gout, eout, self.start_shift, self.end_shift)
        gout1 = self.aggregate_joints2(gout, eout, self.ev_shift, self.ve_shift)

        gout = self.gconv_layers3(gout1, joint_feats[1])
        eout = self.econv_layers3(eout1, None)

        joints_out2 = self.gconv_output2(gout)
        edges_out2 = self.econv_output2(eout)

        # aggregation
        eout1 = self.aggregate_edges3(gout, eout, self.start_shift, self.end_shift)
        gout1 = self.aggregate_joints3(gout, eout, self.ev_shift, self.ve_shift)        
        
        gout = self.gconv_layers4(gout1, joint_feats[2])
        eout = self.econv_layers4(eout1, None)

        joints_out3 = self.gconv_output3(gout)
        edges_out3 = self.econv_output3(eout)

        if gt is not None:
            # generate gt edges
            labels = (gt[:,:,2] > 0).float()
            labels_start = self.start_shift.matmul(labels.unsqueeze(dim=2))
            labels_end = self.end_shift.matmul(labels.unsqueeze(dim=2))
            edge_labels = labels_start * labels_end

            gt_edges = self.sub_matrix.matmul(gt[:,:,:2])

            return [joints_out1, joints_out2, joints_out3], [edges_out1, edges_out2, edges_out3], [gt_edges, edge_labels]
        else:
            return [joints_out1, joints_out2, joints_out3], edges_out3


    