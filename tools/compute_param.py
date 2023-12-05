'''
@author:lingteng qiu
@name:OPEC_GCN
'''
import sys
sys.path.append("./")
from opt import opt
from mmcv import Config
from engineer.SPPE.src.main_fast_inference import *
import torch.nn as nn
import torch
from engineer.models.builder import build_backbone

import numpy as np
from thop import profile





if __name__ == "__main__":

    args = opt
    assert args.config is not None,"you must give your model config"
    cfg = Config.fromfile(args.config)

    model_pos = build_backbone(cfg.model)

    Total_param = 0
    Trainable_param = 0
    NonTrainable_param = 0

    for param in model_pos.parameters():
        mulValue = np.prod(param.size())
        Total_param += mulValue
        if param.requires_grad:
            Trainable_param += mulValue
        else:
            NonTrainable_param += mulValue

    print(f'Total params: {Total_param}')
    print(f'Trainable params: {Trainable_param}')
    print(f'Non-trainable params: {NonTrainable_param}')

    # x = torch.randn(64, 12, 3)
    # heatmaps = torch.randn(64, 17, 80, 64)
    # ret_feats = []
    # feat1 = torch.randn(64, 512, 20, 16)
    # ret_feats.append(feat1)
    # feat2 = torch.randn(64, 256, 40, 32)
    # ret_feats.append(feat2)
    # feat3 = torch.randn(64, 128, 80, 64)
    # ret_feats.append(feat3)

    # flops, params = profile(model_pos, inputs=(x,heatmaps,ret_feats))
    # print(f"flops:{flops}")
    # print(f"params:{params}")