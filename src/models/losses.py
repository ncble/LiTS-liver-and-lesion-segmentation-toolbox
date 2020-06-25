"""
Copyright (C) 2020  Lu Lin
Warning: The code is released under the license GNU GPL 3.0


"""

import os
import glob
import numpy as np
import torch


def CrossEntropyLoss(output, target, label_weights=[1., 1., 1.], ignore_index=-100, device=None):
    """
    Warning: this loss contains the log-softmax (trick) activation !

    :param output (torch tensor) shape = (N, C, H, W), C=3 three classes here
    :param target (torch tensor) shape =  (N, H, W)
    :param label_weights (list) of length C. C is the number of classes.

    """
    weight = torch.tensor(label_weights).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    return loss_fn(output, target)


def DiceLoss(output, target, label_weights=None, smooth=0.0001, device=None):
    """Dice (aka F1-score) (Multi-class dice loss)

        Dice = (2*|X & Y|) / (|X| + |Y|)
             = 2*sum(|X*Y|) / (sum(X^2) + sum(Y^2))

    :param output (torch FloatTensor) shape = (N, C, H, W)
    :param target (torch LongTensor) shape = (N, H, W)
    :param label_weights (list) of length C. C is the number of classes.
                          loss weights of each class (len(label_weights) == C)

    use smooth to avoid overflow issue, if smooth too high, the Dice goes high also.

    return 1 - Dice
    """
    label_weights = torch.tensor(label_weights).to(device)

    output = torch.nn.Sigmoid()(output)  # or approx output.exp()
    # one-hot encoding of target
    encoded_target = output.detach() * 0
    encoded_target.scatter_(1, target.unsqueeze(1), 1)

    if label_weights is None:
        label_weights = 1

    intersection = output * encoded_target
    numerator = 2 * intersection.sum(0).sum(1).sum(1) + smooth
    denominator = output + encoded_target
    denominator = denominator.sum(0).sum(1).sum(1) + smooth
    loss_per_channel = label_weights * (1 - (numerator / denominator))

    return loss_per_channel.sum() / (output.size(1)*label_weights.sum())


def FocalLoss(output, target, label_weights=None, alpha=0.5, gamma=2):
    """Multi-class Focal loss (2017)

    ref: https://arxiv.org/abs/1708.02002

    :param output (torch FloatTensor) shape = (N, C, H, W)
    :param target (torch LongTensor) shape = (N, H, W)
    :param label_weights (list) of length C. C is the number of classes.
                          loss weights of each class (len(label_weights) == C)

    """
    cross_entropy = torch.nn.CrossEntropyLoss(weight=label_weights)
    logpt = -cross_entropy(output, target)
    pt = torch.exp(logpt)
    loss = -((1 - pt) ** gamma) * alpha * logpt
    
    return loss


def IoULoss(output, target, label_weights=None):
    """
    Intersection over union
    """
    raise NotImplementedError
    return


def LovaszSoftmaxLoss(output, target, label_weights=None):
    raise NotImplementedError
    return 




if __name__ == "__main__":
    print("Start")
