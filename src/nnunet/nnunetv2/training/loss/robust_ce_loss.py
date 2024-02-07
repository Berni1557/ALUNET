import torch
from torch import nn, Tensor
import numpy as np


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # print('input123', input.shape)
        # print('target123', target.shape)
        # print('target123', target.sum())
        # print('target123', target.sum(dim=(1,2)))
        # import sys
        # sys.exit()
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

class RobustCrossEntropyLossCACS(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    
    def __init__(self):
        weight = torch.ones(14)
        weight[0]=0.1
        weight = weight/weight.sum()
        weight = weight.cuda()
        
        super(RobustCrossEntropyLossCACS, self).__init__(weight)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # print('input123', input.shape)
        # print('target123', target.shape)
        # print('target123', target.sum())
        # print('target123', target.sum(dim=(1,2)))
        # import sys
        # sys.exit()
        if len(target.shape) == len(input.shape):
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())

class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()

