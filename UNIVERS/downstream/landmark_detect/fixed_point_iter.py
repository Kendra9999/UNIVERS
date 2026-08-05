# Copyright (c) Medical AI Lab, Alibaba DAMO Academy
# Project Home: https://github.com/alibaba-damo-academy/self-supervised-anatomical-embedding-v2
import numpy as np
import torch
import torch.nn.functional as F
import statsmodels.api as sm

def make_local_grid_patch(pt, margin_x, margin_y, margin_z, imshape_x, imshape_y, imshape_z):
    pt = np.asarray(pt).astype('int')
    pt_patch = np.zeros((2 * margin_x + 1, 2 * margin_y + 1, 2 * margin_z + 1, 3), dtype='int')
    for j in range(-margin_x, margin_x + 1):
        for k in range(-margin_y, margin_y + 1):
            for w in range(-margin_z, margin_z + 1):
                pt_patch[j + margin_x, k + margin_y, w + margin_z, :] = pt + np.array(
                    [j, k, w])
    pt_patch = pt_patch.reshape(-1, 3)

    pt_patch[:, 0] = np.where(pt_patch[:, 0] < imshape_x, pt_patch[:, 0], imshape_x - 1)
    pt_patch[:, 1] = np.where(pt_patch[:, 1] < imshape_y, pt_patch[:, 1], imshape_y - 1)
    pt_patch[:, 2] = np.where(pt_patch[:, 2] < imshape_z, pt_patch[:, 2], imshape_z - 1)
    pt_patch[:, 0] = np.where(pt_patch[:, 0] > 0, pt_patch[:, 0], 0)
    pt_patch[:, 1] = np.where(pt_patch[:, 1] > 0, pt_patch[:, 1], 0)
    pt_patch[:, 2] = np.where(pt_patch[:, 2] > 0, pt_patch[:, 2], 0)
    pt_patch = np.unique(pt_patch, axis=0)
    return pt_patch


def get_sim_semantic_embed_loc_multi_embedding_space(query_img_info, key_img_info, query_points):
    fine_query = query_img_info[0]
    coarse_query = query_img_info[1]
    sem_query = query_img_info[2]
    coarse_query = F.interpolate(coarse_query, fine_query.shape[2:], mode='trilinear', align_corners=False)
    coarse_query = F.normalize(coarse_query, dim=1)

    fine_key = key_img_info[0]
    coarse_key = key_img_info[1]
    sem_key = key_img_info[2]
    coarse_key = F.interpolate(coarse_key, fine_key.shape[2:], mode='trilinear', align_corners=False)
    coarse_key = F.normalize(coarse_key, dim=1)

    query_fine = fine_query[0, :, query_points[:, 0], query_points[:, 1], query_points[:, 2]].transpose(1, 0)
    key_fine = fine_key[0, :, :, :, :].reshape(128, -1)

    query_coarse = coarse_query[0, :, query_points[:, 0], query_points[:, 1], query_points[:, 2]].transpose(1, 0)
    key_coarse = coarse_key[0, :, :, :, :].reshape(128, -1)

    query_sem = sem_query[0, :, query_points[:, 0], query_points[:, 1], query_points[:, 2]].transpose(1, 0)
    key_sem = sem_key[0, :, :, :, :].reshape(128, -1)

    sim_fine = torch.einsum("nc,ck->nk", query_fine, key_fine)
    sim_coarse = torch.einsum("nc,ck->nk", query_coarse, key_coarse)
    sim_sem = torch.einsum("nc,ck->nk", query_sem, key_sem)

    sim_fine = sim_fine.view(query_points.shape[0], 1, fine_key.shape[2], fine_key.shape[3],
                             fine_key.shape[4])
    sim_coarse = sim_coarse.view(query_points.shape[0], 1, coarse_key.shape[2], coarse_key.shape[3],
                                 coarse_key.shape[4])
    sim_sem = sim_sem.view(query_points.shape[0], 1, fine_key.shape[2], fine_key.shape[3],
                           fine_key.shape[4])
    sim_all = (sim_fine + sim_coarse + sim_sem) / 3
    
    locs = []
    scores = []

    for i in range(sim_all.shape[0]):
        sim = sim_all[i, 0, :, :, :]
        ind = torch.where(sim == sim.max())
        x = ind[0][0].data.cpu().numpy()
        y = ind[1][0].data.cpu().numpy()
        z = ind[2][0].data.cpu().numpy()
        scores.append(sim.max().data.cpu().numpy())
        locs.append([x, y, z])
    
    return np.array(locs).astype(int), np.array(scores)


def fixed_point_iterations(emb1, emb2, query_point, imshape, 
                           x_margin=2, y_margin=2, z_margin=2, iterations=4):
    """performing fixed point iteration
    """
    pt_query = np.round(query_point).astype('int')
    pt_query_center = np.floor(pt_query * 0.5).astype(int)
    # crop a cube region centered in the template point
    pt_query_normed = make_local_grid_patch(pt_query_center, x_margin, y_margin, z_margin,
                                            emb1[0].shape[2], emb1[0].shape[3], emb1[0].shape[4])
    
    for i in range(iterations):
        if i % 2 == 0:
            if i > 0:
                pt_query_normed = np.floor(pt_query * 0.5).astype(int)
            pt_key_normed, score = get_sim_semantic_embed_loc_multi_embedding_space(emb1, emb2, pt_query_normed)
            pt_key = np.round(pt_key_normed * 2 + 0.5).astype(int)
            pt_k_final = pt_key
        else:
            pt_query_normed = np.floor(pt_query * 0.5).astype(int)
            pt_key_normed, score = get_sim_semantic_embed_loc_multi_embedding_space(emb2, emb1, pt_query_normed)
            pt_key = np.round(pt_key_normed * 2 + 0.5).astype(int)
            pt_q_final = pt_key
            pt_final_score = score
        print(f'score:{score.mean(), score.max()}')
        pt_query = pt_key

    dis = np.linalg.norm((pt_q_final - np.array(query_point)), axis=1)
    final_q_all = []
    final_k_all = []
    final_score_all = []
    for i in range(dis.shape[0]):
        # if dis[i] < 100 and pt_final_score[i] > .7:
            final_q_all.append(pt_q_final[i, :])
            final_k_all.append(pt_k_final[i, :])
            final_score_all.append(pt_final_score[i])

    final_q_all = np.array(final_q_all)
    final_k_all = np.array(final_k_all)
    final_score_all = np.array(final_score_all)

    final_k_list = list([tuple(t) for t in final_k_all])
    final_k_count = dict((k_ele, final_k_list.count(k_ele)) for k_ele in final_k_list)

    final_q = []
    final_k = []

    for keys in final_k_count:
        ind_k = np.where(np.logical_and(np.logical_and(final_k_all[:, 0] == keys[0], final_k_all[:, 1] == keys[1]),
                                        final_k_all[:, 2] == keys[2]))[0]
        s = final_score_all[ind_k]
        q = final_q_all[ind_k, :]
        final_q.append(q[0, :])
        final_k.append(keys)
    final_q = np.array(final_q)
    final_k = np.array(final_k)

    k_xs = np.unique(final_k[:, 0])
    k_ys = np.unique(final_k[:, 1])
    k_zs = np.unique(final_k[:, 2])
    q_zs = np.unique(final_q[:, 2])

    shift = final_q - query_point

    ones = np.ones((final_q.shape[0], 1))
    final_q_ext = np.concatenate((final_q, ones), axis=1)
    m = []

    if k_zs.shape[0] == 1:
        z_ratro = 1 #norm_info_1[2] / norm_info_2[2]
        z_bias = k_zs - q_zs * z_ratro
        k = [0, 1]
        for i in k:
            try:
                mod = sm.RLM(final_k[:, i],
                            np.concatenate([final_q_ext[:, 0:1], final_q_ext[:, 1:2], final_q_ext[:, 3:]], axis=1))
                res = mod.fit()
            except:
                mod = sm.OLS(final_k[:, i],
                            np.concatenate([final_q_ext[:, 0:1], final_q_ext[:, 1:2], final_q_ext[:, 3:]], axis=1))
                res = mod.fit()
            m.append(res.params)
        m.append([0, 0, z_bias[0]])
        m.append(np.array([0, 0, 1]))
        m = np.array(m)
        m = np.concatenate([m[:, :2], np.array([0, 0, z_ratro, 0]).reshape(-1, 1), m[:, 2:]], axis=1)

    else:
        k = [0, 1, 2]
        for i in k:
            try:
                mod = sm.RLM(final_k[:, i], final_q_ext[:, ])
                res = mod.fit()
            except:
                mod = sm.OLS(final_k[:, i], final_q_ext[:, ])
                res = mod.fit()
            m.append(res.params)
        m.append(np.array([0, 0, 0, 1]))
        m = np.array(m)

    shift_affine = np.matmul(m[:3, :3], shift.T).transpose(1, 0)
    pt_k_shift = final_k - shift_affine
    pt_k_shift = pt_k_shift.astype(int)
    pt_k_shift = np.where(pt_k_shift < 0, 0, pt_k_shift)
    max_shape = np.array(imshape) - 1
    max_shape = np.array([max_shape[0], max_shape[1], max_shape[2]])
    pt_k_shift = np.where(pt_k_shift > max_shape, max_shape, pt_k_shift)
    pt_k_shift_mean = np.mean(pt_k_shift, axis=0)
    # pt_k_shift_mean = pt_k_shift_mean.astype('int')

    return pt_k_shift_mean, final_score_all.mean()


