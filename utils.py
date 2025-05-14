import os
import time
import shutil

import torch
import numpy as np


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


from sklearn.metrics import precision_recall_curve

from sklearn.metrics import roc_auc_score,recall_score,precision_score
import cv2

def calc_iou(y_pred,y_true):
    # import pdb;pdb.set_trace()
    #y_pred, y_true = y_pred.squeeze(-1), y_true.squeeze(-1)
    y_pred, y_true = y_pred.squeeze(0), y_true.squeeze(0)
    iou_sum=0
    with torch.no_grad():
        assert y_pred.shape == y_true.shape
        # pos_err, neg_err, ber = 0, 0, 0
        y_true = y_true
        y_pred = y_pred#.cpu()
        # for i in range(batchsize):
        #     # import pdb;pdb.set_trace()
        #     true = y_true[i]#.flatten()
        #     pred = y_pred[i]#.flatten()
        pred = y_pred * 255
        gt = y_true * 255
        gt = (gt > 125)
        pred = (pred> 125)
        iou_sum = get_iou_matrix_input_tensor_batch(pred,gt)#.mean()

        iou_sum= iou_sum.cpu().numpy()
        # import pdb;pdb.set_trace()
        # iiii=1
    return iou_sum, np.array(0), np.array(0), np.array(0)

def get_iou_matrix_input_tensor_batch(masks_gt, masks_pred):
    gt=masks_gt
    dt=masks_pred
    # import pdb;pdb.set_trace()
    intersection = torch.sum((gt * dt) > 0, dim=(1,2))
    union = torch.sum((gt + dt) > 0, dim=(1,2)) 
    # import pdb;pdb.set_trace()
    return intersection / union