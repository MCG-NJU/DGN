from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn

class ScoreLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, pred_coords, gt_coords, pred_score):
        
        dist = nn.functional.pairwise_distance(pred_coords, gt_coords)

        gt_score = torch.exp(-dist)
        loss = self.criterion(pred_score, gt_score)

        return loss