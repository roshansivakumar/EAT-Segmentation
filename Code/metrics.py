import torch
from keras import backend as K # Using ternsorflow as backend
import numpy as np

# y1 - y_true/output and y2 - y_pred/label
# smooth to avoid 0/0 output on compute, it is useful to add an additional
# smoothing factor to the equation to achieve more stable training results...if \
# they are used as loss functions

"""
Metrics here:
- intersection over union (IoU) a.k.a Jaccards index
- dice coefficient a.k.a F1 score
- pixel accuracy (TP, FP, TN, FN) <not included as it's not a desirable metric>

dice coeff and IoU are positively corrleated, on a scale of 0-1 (greatest similarity
b/w predicied and truth)
"""

def iou(y1: torch.Tensor, y2: torch.Tensor): # Using pytorch for a quick iou calculation
#def iou(y1: np.array, y2: np.array):
    y1 = y1.squeeze(1)  # Data x H x W
    intersection = (y1 & y2).float().sum((1, 2))
    union = (y1 | y2).float().sum((1, 2))
    iou = intersection/union
    #smooth = 1e-6 # From example
    #iou = (intersection + smooth) / (union + smooth) # iou with smoothening
    iou_thresh = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10 # delete if using numpy
    # Or thresholded.mean() for average
    return iou_thresh

def iou_imp2(y1, y2): # Another implementation using keras with ternsorflow backend
    y1 = K.flatten(y1);
    y2 = K.flatten(y2);
    intersection = K.sum(y1 * y2);
    union = K.sum(y1) + K.sum(y2) - interesection
    iou = intersection/union
    # can return jacards loss by replacing iou with (1-iou)
    return iou

def dice_coeff(y1, y2): # dice coefficient
    intersection = K.sum(K.sum(K.abs(y1 * y2), axis=-1))
    union = K.sum(K.sum(K.abs(y1) + K.abs(y2), axis=-1))
    dice =  2*intersection/union
    # dice = 2*(intersection+smooth)/(union+smooth) if 0/0 occours
    # to return dice loss just return (1-dice) * (smoothening factor if any)
    return dice
