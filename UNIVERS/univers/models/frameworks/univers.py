# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
# Project Home: https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2
# Modified on 2026-08-05: Based on the above open-source project for secondary development
# Modified by: Derong Yu
import os.path
import warnings

# import ipdb
import torch
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck, build_loss
from mmdet.models.detectors.base import BaseDetector
import torch.nn.functional as F
from torch import linalg as LA
import time
import pickle
import numpy as np


@DETECTORS.register_module()
class UNIVERS(BaseDetector):
    def __init__(self,
                 backbone,
                 neck=None,
                 read_out_head=None,
                 sem_neck=None,
                 superloss=None,
                 train_cfg=None,
                 test_cfg=None,
                 init_cfg=None):
        super(UNIVERS, self).__init__(init_cfg)
        self.backbone = build_backbone(backbone)
        self.backbone.init_weights()
        if neck is not None:
            self.neck = build_neck(neck)
        self.criterion = torch.nn.CrossEntropyLoss().cuda()
        self.supcriterion = build_loss(superloss)
        self.read_out_head = build_neck(read_out_head)
        if sem_neck is not None:
            self.semantic_head = build_neck(sem_neck)
        else:
            self.semantic_head = build_neck(neck)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
    
    def extract_feat(self, img, normalize=True):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        out1 = self.neck(x[1:self.neck.end_level+1])[0]
        out2 = self.read_out_head(x[self.neck.end_level+1:self.neck.end_level+1+self.read_out_head.end_level])[0]
        out3 = self.semantic_head(x[1:self.semantic_head.end_level+1])[0]
        if normalize:
            out1 = F.normalize(out1, dim=1)
            out2 = F.normalize(out2, dim=1)
            out3 = F.normalize(out3, dim=1)
        out1 = out1.type(torch.half)
        out2 = out2.type(torch.half)
        out3 = out3.type(torch.half)
        return [out1, out2, out3]
    
    def forward_train(self,
                      img,
                      img_metas,
                      **kwargs):
        """Forward function during training.
        Args:
            img (dict): a dict of keys:
                    "overlap_patches": a Tensor of shape (N, C, D, H, W)
                    "overlap_patches_girds": a Tensor of shape (N, 3, D, H, W)
                    "whole_images": a Tensor of shape (N, C, D, H, W)
                    "whole_images_labels": a Tensor of shape (N, 1, D, H, W)

            img_metas (list[dict]): a list of dict of keys:
                    "style": "overlap" or "whole"
        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        overlap_patches = img["overlap_patches"]
        overlap_patches_girds = img["overlap_patches_girds"]
        whole_images = img["whole_images"]
        whole_images_labels = img["whole_images_labels"]

        losses = dict()
        
        # forward on overlap patches
        overlap_feats = self.extract_feat(overlap_patches)
        appear_loss = self.loss_appearance(overlap_feats, overlap_patches_girds)
        losses["appear_loss"] = appear_loss

        # forward on whole images
        whole_feats = self.extract_feat(whole_images)
        seman_loss = self.loss_semantic(whole_feats, whole_images_labels)
        losses["seman_loss"] = seman_loss

        return losses
    
    def loss_appearance(self, feats, meshgrids, **kwargs):

        N, C, D, H, W = meshgrids.shape
        grid_half = F.interpolate(meshgrids, size=(int(D / 2), int(H / 2), int(W / 2)), mode='trilinear', align_corners=False)
        grid_16 = F.interpolate(meshgrids, size=(int(D / 16), int(H / 16), int(W / 16)), mode='trilinear', align_corners=False)
        
        grid_half = grid_half.type(torch.half)
        grid_16 = grid_16.type(torch.half)

        for i in range(int(N / 4)):
            result = self.single_appear_fine_loss(feats[0][4 * i : 4 * (i + 1)], feats[1][4 * i : 4 * (i + 1)],
                                                  grid_half[4 * i : 4 * (i + 1)], grid_16[4 * i : 4 * (i + 1)])
            if i == 0:
                fine_loss = result['loss']
            else:
                fine_loss += result['loss']
        fine_loss = fine_loss / int(N / 4)

        result = self.single_appear_coarse_loss(feats[1], grid_16)
        coarse_loss = result['loss']

        loss = fine_loss + coarse_loss
        return loss
    

    def single_appear_fine_loss(self, fine_feat, coarse_feat, fine_grid, coarse_grid):
        out = dict()

        N_views = fine_feat.shape[0]  # 4

        list_fine_feat = [fine_feat[i].view(fine_feat.shape[1], -1) for i in range(N_views)]

        fine_local_grid = self.meshgrid3d(fine_feat.shape[2:], device=fine_feat.device)  # z y x

        list_y_min = [fine_grid[i, 0].min() for i in range(N_views)]
        list_y_max = [fine_grid[i, 0].max() for i in range(N_views)]
        list_x_min = [fine_grid[i, 1].min() for i in range(N_views)]
        list_x_max = [fine_grid[i, 1].max() for i in range(N_views)]
        list_z_min = [fine_grid[i, 2].min() for i in range(N_views)]
        list_z_max = [fine_grid[i, 2].max() for i in range(N_views)]

        intersection_y = [max(list_y_min), min(list_y_max)]
        intersection_x = [max(list_x_min), min(list_x_max)]
        intersection_z = [max(list_z_min), min(list_z_max)]
        
        # defintely have overlap

        list_intersection_volume = [(fine_grid[i, 0] >= intersection_y[0]) * (
                                     fine_grid[i, 0] <= intersection_y[1]) \
                                  * (fine_grid[i, 1] >= intersection_x[0]) * (
                                     fine_grid[i, 1] <= intersection_x[1]) \
                                  * (fine_grid[i, 2] >= intersection_z[0]) * (
                                     fine_grid[i, 2] <= intersection_z[1])
                                    for i in range(N_views)]
        
        list_index_overlap = [fine_local_grid[list_intersection_volume[i] > 0, :] for i in range(N_views)]
        pos_index_all = fine_local_grid.view(-1, 3)

        # yxz pos_mm_overlap      zyx pos_index
        list_pos_mm_overlap = [fine_grid[i, :, list_intersection_volume[i] > 0] for i in range(N_views)]
        list_num_points = [list_pos_mm_overlap[i].shape[1] for i in range(N_views)]
        
        list_points_select = []
        for i in range(N_views):
            if list_num_points[i] < self.train_cfg.intra_cfg.pre_select_pos_number:
                list_points_select.append(torch.randperm(list_num_points[i], device=fine_feat.device))
            else:
                list_points_select.append(torch.randperm(list_num_points[i], device=fine_feat.device)[:self.train_cfg.intra_cfg.pre_select_pos_number])

        list_pos_use = [list_pos_mm_overlap[i][:, list_points_select[i]] for i in range(N_views)]
        list_pos_mm_all = [fine_grid[i, :, :, :, :].view(3, -1) for i in range(N_views)]

        with torch.no_grad():
            list_list_dist = []
            for i in range(N_views):
                list_dist = []
                for j in range(N_views):
                    list_dist.append(
                        LA.norm((list_pos_use[i].view(-1, list_pos_use[i].shape[1], 1) - \
                        list_pos_mm_all[j].view(-1, 1, list_pos_mm_all[j].shape[1])), dim=0)
                    )
                list_list_dist.append(list_dist)

        list_pos = []
        for i in range(N_views):
            reasonable_pos = torch.ones(list_pos_use[i].shape[1], dtype=torch.bool, device=fine_feat.device)
            for j in range(N_views):
                if j != i:
                    reasonable_pos = torch.logical_and(reasonable_pos,
                        list_list_dist[i][j].min(dim=1)[0] < self.train_cfg.intra_cfg.positive_distance)
            pos = torch.where(reasonable_pos)[0]
            
            if pos.shape[0] == 0:
                reasonable_pos = torch.ones(list_pos_use[i].shape[1], dtype=torch.bool, device=fine_feat.device)
                for j in range(N_views):
                    if j != i:
                        reasonable_pos = torch.logical_and(reasonable_pos,
                            list_list_dist[i][j].min(dim=1)[0] < list_list_dist[i][j].min() + 0.5)
                pos = torch.where(reasonable_pos)[0]

            if pos.shape[0] > self.train_cfg.intra_cfg.after_select_pos_number:
                pos = pos[torch.randperm(pos.shape[0])[:self.train_cfg.intra_cfg.after_select_pos_number]]
            list_pos.append(pos)

        list_list_dist = [[list_list_dist[i][j][list_pos[i], :] for j in range(N_views)] for i in range(N_views)]

        list_pos_source = [list_points_select[i][list_pos[i]] for i in range(N_views)]

        list_list_target = []
        for i in range(N_views):
            list_target = []
            for j in range(N_views):
                if j != i:
                    list_target.append(list_list_dist[i][j].min(dim=1)[1])
                else:
                    list_target.append(-1)
            list_list_target.append(list_target)

        list_neg_mask = []
        for i in range(N_views):
            list_use_mask = []
            for j in range(N_views):
                dist = list_list_dist[i][j]
                ind_ignores = torch.stack(torch.where(dist < self.train_cfg.intra_cfg.ignore_distance))
                use_mask = torch.ones_like(dist)
                use_mask[ind_ignores[0, :], ind_ignores[1, :]] = 0
                list_use_mask.append(use_mask)
            list_neg_mask.append(torch.cat(list_use_mask, dim=1))

        coarse_feat_resample = F.interpolate(coarse_feat, fine_feat.shape[2:], mode='trilinear', align_corners=False)
        coarse_feat_resample = F.normalize(coarse_feat_resample, dim=1)
        list_coarse_feat = [coarse_feat_resample[i].view(coarse_feat_resample.shape[1], -1) for i in range(N_views)]

        list_q_view_location = [list_index_overlap[i][list_pos_source[i]].type(torch.LongTensor) for i in range(N_views)]
        list_q_view_fine_feat = [fine_feat[i, :, list_q_view_location[i][:, 0],
                                 list_q_view_location[i][:, 1], list_q_view_location[i][:, 2]].transpose(0, 1)
                                 for i in range(N_views)]
        list_q_view_coarse_feat = [coarse_feat_resample[i, :, list_q_view_location[i][:, 0],
                                   list_q_view_location[i][:, 1], list_q_view_location[i][:, 2]].transpose(0, 1)
                                   for i in range(N_views)]
        
        list_k_view_fine_feat = []
        for i in range(N_views):
            k_view_fine_feat = []
            for j in range(N_views):
                if j != i:
                    k_view_location = pos_index_all[list_list_target[i][j]].type(torch.LongTensor)
                    k_view_fine_feat.append(fine_feat[j, :, k_view_location[:, 0],
                                            k_view_location[:, 1], k_view_location[:, 2]].transpose(0, 1))
            list_k_view_fine_feat.append(k_view_fine_feat)

        list_inner = []
        for i in range(N_views):
            list_inner.append([torch.einsum("nc,nc->n", list_q_view_fine_feat[i], 
                                  list_k_view_fine_feat[i][j]).view(-1, 1)
                               for j in range(N_views-1)])
            
        list_neg_fine = [torch.einsum("nc,ck->nk", list_q_view_fine_feat[i], 
                                      torch.cat(list_fine_feat, dim=1))
                         for i in range(N_views)]
        list_neg_coarse = [torch.einsum("nc,ck->nk", list_q_view_coarse_feat[i],
                                        torch.cat(list_coarse_feat, dim=1))
                           for i in range(N_views)]
        list_neg_all = [(list_neg_fine[i] + list_neg_coarse[i]) * list_neg_mask[i]
                        for i in range(N_views)]
        
        list_neg_candidate_index = [list_neg_all[i].topk(self.train_cfg.intra_cfg.pre_select_neg_number, dim=1)[1]
                                    for i in range(N_views)]
        list_neg_use = [torch.zeros((list_q_view_fine_feat[i].shape[0], self.train_cfg.intra_cfg.after_select_neg_number),
                                    device=list_neg_all[i].device)
                        for i in range(N_views)]
        for i in range(N_views):
            for j in range(list_q_view_fine_feat[i].shape[0]):
                use_index = list_neg_candidate_index[i][j, torch.randperm(list_neg_candidate_index[i][j, :].shape[0])[
                                                         :self.train_cfg.intra_cfg.after_select_neg_number]]
                list_neg_use[i][j, :] = list_neg_fine[i][j, use_index]

        list_logits_view = []
        for i in range(N_views):
            for j in range(N_views-1):
                list_logits_view.append(torch.cat([list_inner[i][j], list_neg_use[i]], dim=1))
        logits = torch.cat(list_logits_view, dim=0)
        logits = logits / self.train_cfg.intra_cfg.temperature

        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
        loss = self.criterion(logits, labels)
        out['loss'] = loss

        return out

    def single_appear_coarse_loss(self, coarse_feat, coarse_grid):
        out = dict()

        coarse_feat_flatten = coarse_feat.view(coarse_feat.shape[0], 128, -1)
        coarse_feat_flatten = coarse_feat_flatten.permute(1, 0, 2)
        coarse_feat_flatten = coarse_feat_flatten.reshape(128, -1)

        N_views = 4
        for i in range(int(coarse_feat.shape[0] / N_views)):            
            list_feat_coarse = [coarse_feat[i*N_views+j].view(coarse_feat.shape[1], -1) for j in range(N_views)]
            list_loc_coarse = [coarse_grid[i*N_views+j].view(3, -1) for j in range(N_views)]

            global_use_index_list = [j + N_views * i * list_feat_coarse[0].shape[1] for j in
                                     range(0, N_views * list_feat_coarse[0].shape[1])]
            global_index_list = [j for j in range(0, coarse_feat.shape[0] * list_feat_coarse[0].shape[1])]
            global_search_index_list = list(set(global_index_list) - set(global_use_index_list))
            global_search_index = torch.tensor(global_search_index_list).to(list_feat_coarse[0].device)
            
            with torch.no_grad():
                list_list_dist = []
                for j in range(N_views):
                    list_dist = []
                    for k in range(N_views):
                        list_dist.append(
                            LA.norm((list_loc_coarse[j].view(-1, list_loc_coarse[j].shape[1], 1) - \
                            list_loc_coarse[k].view(-1, 1, list_loc_coarse[k].shape[1])), dim=0)
                        )
                    list_list_dist.append(list_dist)

            list_pos_coarse_can = []
            for j in range(N_views):
                reasonable_pos = torch.ones(list_loc_coarse[j].shape[1], dtype=torch.bool, device=coarse_feat.device)
                for k in range(N_views):
                    if k != j:
                        reasonable_pos = torch.logical_and(reasonable_pos,
                            list_list_dist[j][k].min(dim=1)[0] < self.train_cfg.intra_cfg.coarse_positive_distance)
                pos = torch.where(reasonable_pos)[0]
                list_pos_coarse_can.append(pos)

            if all([list_pos_coarse_can[j].shape[0] > 0 for j in range(N_views)]):

                list_list_dist = [[list_list_dist[j][k][list_pos_coarse_can[j], :] for k in range(N_views)] for j in range(N_views)]

                list_list_key_coarse_can = []
                for j in range(N_views):
                    list_key_coarse_can = []
                    for k in range(N_views):
                        if k != j:
                            list_key_coarse_can.append(list_list_dist[j][k].min(dim=1)[1])
                        else:
                            list_key_coarse_can.append(-1)
                    list_list_key_coarse_can.append(list_key_coarse_can)

                list_q_view_coarse_feat = [list_feat_coarse[j][:, list_pos_coarse_can[j]].transpose(0, 1)
                                            for j in range(N_views)]
                
                list_k_view_coarse_feat = []
                for j in range(N_views):
                    k_view_coarse_feat = []
                    for k in range(N_views):
                        if k != j:
                            k_view_coarse_feat.append(
                                list_feat_coarse[k][:, list_list_key_coarse_can[j][k]].transpose(0, 1))
                    list_k_view_coarse_feat.append(k_view_coarse_feat)

                list_neg_mask = []
                for j in range(N_views):
                    list_use_mask = []
                    for k in range(N_views):
                        dist = list_list_dist[j][k]
                        ind_ignores = torch.stack(torch.where(dist < self.train_cfg.intra_cfg.coarse_ignore_distance))
                        use_mask = torch.ones_like(dist)
                        use_mask[ind_ignores[0, :], ind_ignores[1, :]] = 0
                        list_use_mask.append(use_mask)
                    list_neg_mask.append(torch.cat(list_use_mask, dim=1))

                
                list_inner = []
                for j in range(N_views):
                    list_inner.append([torch.einsum("nc,nc->n", list_q_view_coarse_feat[j], 
                                        list_k_view_coarse_feat[j][k]).view(-1, 1)
                                    for k in range(N_views-1)])
                    
                list_neg_coarse = [torch.einsum("nc,ck->nk", list_q_view_coarse_feat[j],
                                                torch.cat(list_feat_coarse, dim=1))
                                for j in range(N_views)]
                list_neg_coarse = [list_neg_coarse[j] * list_neg_mask[j] 
                                for j in range(N_views)]
                
                list_neg_candidate_coarse_index = [list_neg_coarse[j].topk(self.train_cfg.intra_cfg.coarse_pre_select_neg_number, dim=1)[1]
                                            for j in range(N_views)]
                list_neg_use_coarse = [torch.zeros((list_q_view_coarse_feat[j].shape[0], self.train_cfg.intra_cfg.coarse_after_select_neg_number),
                                            device=list_neg_coarse[j].device)
                                    for j in range(N_views)]
                for j in range(N_views):
                    for k in range(list_q_view_coarse_feat[j].shape[0]):
                        use_index = list_neg_candidate_coarse_index[j][k, torch.randperm(list_neg_candidate_coarse_index[j][k, :].shape[0])[
                                                                :self.train_cfg.intra_cfg.coarse_after_select_neg_number]]
                        list_neg_use_coarse[j][k, :] = list_neg_coarse[j][k, use_index]

                list_neg_coarse_global_index = [global_search_index[
                        torch.randperm(global_search_index_list.__len__())[
                        :self.train_cfg.intra_cfg.coarse_global_select_number]]
                        for j in range(N_views)]
                list_neg_coarse_global = [coarse_feat_flatten[:, list_neg_coarse_global_index[j]]
                        for j in range(N_views)]
                list_neg_coarse_global_view = [
                        torch.einsum("nc,ck->nk", list_q_view_coarse_feat[j],
                                    list_neg_coarse_global[j])
                        for j in range(N_views)]
                
                list_logits_view = []
                for j in range(N_views):
                    for k in range(N_views-1):
                        list_logits_view.append(torch.cat([list_inner[j][k], 
                                                        list_neg_use_coarse[j], 
                                                        list_neg_coarse_global_view[j]], dim=1))
                logits = torch.cat(list_logits_view, dim=0)
                logits = logits / self.train_cfg.intra_cfg.temperature
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
                loss = self.criterion(logits, labels)
                if out.keys().__len__() == 0:
                    out['loss'] = loss
                else:
                    out['loss'] += loss
            
            else:
                list_neg_coarse_global_index = [global_search_index[
                        torch.randperm(global_search_index_list.__len__())[
                        :self.train_cfg.intra_cfg.coarse_global_select_number]]
                        for j in range(N_views)]
                list_neg_coarse_global = [coarse_feat_flatten[:, list_neg_coarse_global_index[j]]
                        for j in range(N_views)]
                list_neg_coarse_global_view = [
                        torch.einsum("nc,ck->nk", list_feat_coarse[j].transpose(0, 1),
                                    list_neg_coarse_global[j])
                        for j in range(N_views)]
                
                list_logits_view = [torch.einsum("nc,ck->nk", list_feat_coarse[j].transpose(0, 1),
                                            torch.cat(list_feat_coarse, dim=1)).topk(
                    self.train_cfg.intra_cfg.coarse_after_select_neg_number + 1, dim=1)[0]
                    for j in range(N_views)]
                
                list_logits_view = [torch.cat([list_logits_view[j], list_neg_coarse_global_view[j]], dim=1)
                                    for j in range(N_views)]
                logits = torch.cat(list_logits_view, dim=0)
                logits = logits / self.train_cfg.intra_cfg.temperature
                labels = torch.zeros(logits.shape[0], dtype=torch.long).to(logits.device)
                loss = self.criterion(logits, labels)
                if out.keys().__len__() == 0:
                    out['loss'] = loss
                else:
                    out['loss'] += loss

        out['loss'] = out['loss'] / (coarse_feat.shape[0] / N_views)
        return out

        
    def loss_semantic(self, feats, labels, **kwargs):
        N = feats[0].shape[0]
        
        for i in range(int(N / 2)):
            result = self.single_loss_semantic([feats[2][2 * i], feats[2][2 * i + 1]], 
                                               [labels[2 * i], labels[2 * i + 1]])
            if i == 0:
                fine_loss = result['loss']
            else:
                fine_loss += result['loss']
        fine_loss = fine_loss / int(N / 2)
        loss = fine_loss
        return loss
    
    def single_loss_semantic(self, feat, mask):
        out = dict()

        view_1_fine = feat[0]
        view_2_fine = feat[1]
        view_1_fine = view_1_fine.view(view_1_fine.shape[0], -1).unsqueeze(1).permute(2, 1, 0)
        view_2_fine = view_2_fine.view(view_2_fine.shape[0], -1).unsqueeze(1).permute(2, 1, 0)
        
        mask_1 = mask[0].type(torch.half)
        mask_2 = mask[1].type(torch.half)
        mask_1_fine = F.interpolate(mask_1.unsqueeze(0), size=feat[0].shape[1:]).view(-1)
        mask_2_fine = F.interpolate(mask_2.unsqueeze(0), size=feat[1].shape[1:]).view(-1)
        
        fine_feats = torch.cat((view_1_fine, view_2_fine), dim=0)
        fine_labels = torch.cat((mask_1_fine, mask_2_fine), dim=0)
        fine_mean_feat, fine_mean_labels = self.get_mean_vector(fine_labels, fine_feats)
        
        loss = self.supcriterion(fine_feats, fine_labels, fine_mean_feat, fine_mean_labels)
        out['loss'] = loss

        return out
    
    def get_mean_vector(self, mask, features):
        labels = torch.unique(mask).tolist()
        labels_organ = [label for label in labels if label > 0]
        mean_vectors = []
        for label in labels_organ:
            all_ind = torch.where(mask == label)[0]
            mean_vector = features[all_ind, 0, :].mean(dim=0)
            mean_vectors.append(mean_vector)
        mean_vectors = torch.stack(mean_vectors)
        labels_organ = torch.tensor(labels_organ)
        return mean_vectors, labels_organ
    
    
    def meshgrid3d(self, shape, device):
        z_ = torch.linspace(0., shape[0] - 1, shape[0], device=device)
        y_ = torch.linspace(0., shape[1] - 1, shape[1], device=device)
        x_ = torch.linspace(0., shape[2] - 1, shape[2], device=device)
        z, y, x = torch.meshgrid(z_, y_, x_)
        return torch.stack((z, y, x), 3)


    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        x = self.extract_feat(img)
        # outs = []
        out1 = x[0]#.data.cpu().numpy()
        out2 = x[1]#.data.cpu().numpy()
        out3 = x[2]#.data.cpu().numpy()
        outs = [out1, out2, out3, img.data,#.cpu().numpy(),
                img_metas[0]['filename'].split('.', 1)[0]]
        output_embedding = self.test_cfg.get('output_embedding', True)
        if not output_embedding:
            if not os.path.exists(self.test_cfg.save_path):
                os.mkdir(self.test_cfg.save_path)
            outfilename = self.test_cfg.save_path + \
                          img_metas[0]['filename'].split('.', 1)[0] + '.pkl'
            f = open(outfilename, 'wb')
            pickle.dump(outs, f)
            return [
                x[0][0, 0, 0, 0, 0].data.cpu()]  # we have saved the data into harddisk, this is just for fit the code
        else:
            return outs

    def aug_test(self, imgs, img_metas, **kwargs):
        return self.simple_test(imgs, img_metas, **kwargs)
    
        