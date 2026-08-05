"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn
from mmdet.models.builder import LOSSES


@LOSSES.register_module()
class SupConMeanLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, temperature=0.01, contrast_mode='all',
                 base_temperature=1.):
        super(SupConMeanLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels, prototypes, prototypes_labels):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.view(-1, 1)
        prototypes_labels = prototypes_labels.view(-1, 1).to(labels.device)
        mask_p_f = torch.eq(prototypes_labels, labels.T).float().to(features.device)

        features = features.squeeze(1)
        # compute logits
        prototypes_dot_features = torch.div(
            torch.matmul(prototypes, features.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(prototypes_dot_features, dim=1, keepdim=True)
        logits = prototypes_dot_features - logits_max.max().detach()
        # logits = prototypes_dot_features

        # compute log_prob
        exp_logits = torch.exp(logits)
        # v,ind = exp_logits.topk(200)
        # mask_topk = torch.zeros_like(mask).scatter_(1,ind, 1)
        # exp_logits = exp_logits * mask_topk
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(20, 20))
        # plt.imshow(mask_topk.data.cpu())
        # plt.show()
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        # mask = mask[(labels>0)[:,0],:]
        # log_prob = log_prob[(labels>0)[:,0],:]
        uses = mask_p_f.sum(1) > 0
        mask_p_f = mask_p_f[uses, :]
        log_prob = log_prob[uses, :]

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask_p_f * log_prob).sum(1) / mask_p_f.sum(1)

        # loss
        loss = - self.base_temperature * mean_log_prob_pos
        loss = loss.view(1, mean_log_prob_pos.shape[0]).mean()
        return loss