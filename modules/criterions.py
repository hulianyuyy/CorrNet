import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F

class SeqKD(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, T=1):
        super(SeqKD, self).__init__()
        self.kdloss = nn.KLDivLoss(reduction='batchmean')
        self.T = T

    def forward(self, prediction_logits, ref_logits, use_blank=True):
        start_idx = 0 if use_blank else 1
        prediction_logits = F.log_softmax(prediction_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        ref_probs = F.softmax(ref_logits[:, :, start_idx:]/self.T, dim=-1) \
            .view(-1, ref_logits.shape[2] - start_idx)
        loss = self.kdloss(prediction_logits, ref_probs)*self.T*self.T
        # mask_probs = F.softmax(ref_logits[:, :, 1:], dim=-1).view(-1, ref_logits.shape[2] - 1)
        # mask = torch.max(mask_probs, dim=1)[0] > 0.5
        # if torch.sum(mask) != 0:
        #     loss = torch.sum(torch.sum(loss, dim=1) * mask) / torch.sum(mask)
        # else:
        #     loss = torch.sum(torch.sum(loss, dim=1) * mask)
        return loss

