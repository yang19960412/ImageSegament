import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        smooth = 1

       # probs = F.sigmoid(logits)
        m1 = logits.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        return score

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super(BCEDiceLoss, self).__init__()
    def forward(self,logits,targets):
        #probs = F.sigmoid(logits)
        BCELoss = nn.BCELoss.cuda(logits,targets)
        num = targets.size(0)
        smooth = 1


        m1 = logits.view(num, -1)
        m2 = targets.view(num, -1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1) + smooth) / (m1.sum(1) + m2.sum(1) + smooth)
        score = 1 - score.sum() / num
        loss = BCELoss*100 + score
        return loss