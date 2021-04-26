#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

from engineer.models.registry import GENERATOR
from opt import Mscoco
from engineer.SPPE.src.main_fast_inference import *
try:
    from utils.img import transformBox_batch
except ImportError:
    from engineer.SPPE.src.utils.img import transformBox_batch
@GENERATOR.register_module




class Pose_Generator():
    def __init__(self,model,outputResH,outputResW,inputResW,inputResH,GPUS,device ="cuda"):
        self.outputResH=outputResH
        self.outputResW=outputResW
        self.inputResW=inputResW
        self.inputResH=inputResH
        self.model = self._build_generator(model)
        self.model = torch.nn.DataParallel(self.model, device_ids=GPUS)
        self.model.eval()
        if device =="cuda":
            self.model.cuda()
    def __call__(self,inps, boxes, pt1, pt2,gts_list,dts_list,flip_test=False):
        if boxes is None or boxes.nelement() == 0:
            return None
        inps = inps.cuda()
        with torch.no_grad():
            hm, ret_features = self.model(inps)
        gts_epoch =[]
        dts_epoch = []
        h_4,w_4 = self.outputResH,self.outputResW
        if gts_list is None:
            dts_epoch = np.asarray(dts_list).reshape(-1,12,3)
            pre_keypoints = dts_epoch[:, ..., :2]
            # hm_1_4 = transformBox_batch(pre_keypoints, pt1, pt2, self.inputResH, self.inputResW, self.outputResH,self.outputResW)
            # self.hm_normalize(hm_1_4,h_4,w_4)
            self.normalize_only(dts_epoch,pt1,pt2,flip_test)
            dts = torch.from_numpy(dts_epoch).float()
            return dts,ret_features,hm
        else:
            for gts,dts in zip(gts_list,dts_list):
                gts = np.asarray(gts)
                gts_epoch.append(gts)
                dts = np.asarray(dts)
                dts_epoch.append(dts)
            gts_epoch = np.concatenate(gts_epoch,axis=0).astype(np.float32).copy()
            dts_epoch = np.concatenate(dts_epoch,axis=0).copy()

            # extract_feature_from_here
            # pre_keypoints = dts_epoch[:, ..., :2]
            # hm_1_4 = transformBox_batch(pre_keypoints, pt1, pt2, self.inputResH, self.inputResW, self.outputResH,
            #                             self.outputResW)
            # self.hm_normalize(hm_1_4,h_4,w_4)
            self.normalize(dts_epoch,gts_epoch,pt1,pt2)
            dts = torch.from_numpy(dts_epoch).float()
            gts = torch.from_numpy(gts_epoch).float()

            return dts,gts,ret_features,hm


    def extract_features_joints(self,ret_features,hms):
        '''
        extract features from joint feature_map

        :return:
        '''

        joint_features = []


        for feature, hm_pred in zip(ret_features, hms):
            joint_feature = torch.zeros([feature.shape[0], feature.shape[1], hm_pred.shape[1]])
            for bz in range(feature.shape[0]):
                for joint in range(hm_pred.shape[1]):
                    joint_feature[bz, :, joint] = feature[bz, :, hm_pred[bz, joint, 1], hm_pred[bz, joint, 0]]
            joint_features.append(joint_feature)
        return joint_features

    def normalize_only(self,dts,pt1,pt2,flip=False):
        dts[:,:,:2] = dts[:,:,:2] - pt1.numpy()[:,np.newaxis,:]
        wh = pt2 - pt1
        dts[:,:,:2] = self.normalize_screen_coordinates_bz(dts[:,:,:2], wh.float().numpy())
        if flip:
            dts[:,:,0] *= -1

        return dts

    def inverse_normalize_only(self,dts,pt1,pt2,flip=False):
        wh = pt2 - pt1

        if flip:
            dts[:,:,0] *= -1
        dts[:,:,:2] = self.inverse_normalize_bz(dts[:,:,:2],wh.float().numpy())
        dts[:,:,:2] = dts[:,:,:2] + pt1.numpy()[:,np.newaxis,:]
    
    def inverse_normalize_edges(self,edges,pt1,pt2,flip=False):
        wh = (pt2 - pt1).numpy()
        edges[:,:,:2] = edges[:,:,:2] * wh[:,np.newaxis,:] / 2

        if flip:
            edges[:,:,0] *= -1

    def normalize(self,dts,gts,pt1,pt2,flip_flag):

        dts[:,:,:2] = dts[:,:,:2] - pt1.numpy()[:,np.newaxis,:]
        gts[:,:,:2] = gts[:,:,:2] - pt1.numpy()[:,np.newaxis,:]
        wh = pt2 - pt1
        dts[:,:,:2] = self.normalize_screen_coordinates_bz(dts[:,:,:2], wh.float().numpy())
        gts[:,:,:2] = self.normalize_screen_coordinates_bz(gts[:,:,:2], wh.float().numpy())
        dts[:,:,0] *= flip_flag
        gts[:,:,0] *= flip_flag

    def normalize_screen_coordinates_bz(self,X,wh):
        assert X.shape[-1] == 2
        # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
        #Normalize
        X = X / wh[:,np.newaxis,:] - 0.5
        return X*2

    def inverse_normalize_bz(self,Y, wh):
        assert Y.shape[-1] == 2

        Y/=2.
        Y+=0.5
        Y = Y * wh[:,np.newaxis,:]
        return Y
    
    # def hm_normalize(self,x,h,w):
    #     x[:,:,0] /=w
    #     x[:,:,1] /=h
    #     x-=0.5
    #     x*=2
    def _build_generator(self,model):
        pose_dataset = Mscoco()

        if model == "faster":
            pose_model = InferenNet_fast(4 * 1 + 1, pose_dataset)
        else:
            pose_model = InferenNet(4 * 1 + 1, pose_dataset)
        return pose_model