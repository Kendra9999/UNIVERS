import argparse
import os
import sys
sys.path.append('..')
sys.path.append('../..')

import numpy as np
import torch.nn.functional as F
import matplotlib
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged
)

from utils import *

os.chdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))  # go to root dir of this project

def parse_args():
    parser = argparse.ArgumentParser(description='Contrastive PreTrain')
    parser.add_argument('--config', type=str, 
                        default='configs/univers/univers.py', 
                        help='train config file path')
    parser.add_argument('--checkpoint', type=str, 
                        default='work_dirs/works/univers/20250411_095905/iter_45000.pth', 
                        help='pretrain checkpoint file path')
    parser.add_argument('--img1-file', type=str, 
                        default='/data1/ydr/Project_representation/self-supervised-anatomical-embedding-v2-main/data/raw_data/NIH_lymph_node/ABD_LYMPH_001.nii.gz', 
                        help='query image file path')
    parser.add_argument('--img2-file', type=str, 
                        default='/data1/ydr/Project_representation/self-supervised-anatomical-embedding-v2-main/data/raw_data/MR/T2WI.nii.gz',
                        # default='/data1/ydr/Project_representation/self-supervised-anatomical-embedding-v2-main/data/raw_data/NIH_lymph_node/ABD_LYMPH_002.nii.gz', 
                        help='key image file path')
    parser.add_argument('--query-point', type=list, default=(67, 78, 175), help='query point')
    parser.add_argument('--spacing', type=list, default=(2, 2, 2), help='spacing')
    parser.add_argument('--save-dir', type=str, 
                        default='work_dirs/visual_results/', 
                        help='save visualize results directory path')
    args = parser.parse_args()
    return args


def load_image(img_file, spacing=(2, 2, 2)):
    load_transforms = Compose(
        [
            LoadImaged(keys=["img"]),
            EnsureChannelFirstd(keys=["img"]),
            EnsureTyped(keys=["img"]),
            Orientationd(keys=["img"], axcodes="RAS"),
            Spacingd(keys=["img"], pixdim=spacing, mode=("bilinear")),
            # Rescale to [0, 1]:
            ScaleIntensityRanged(keys=["img"], a_min=-175.0, a_max=250.0, b_min=0.0, b_max=1.0, clip=True),
        ]
    )

    data = {"img": img_file, "img_metas": {"filename": os.path.basename(img_file)}}
    data = load_transforms(data)
    
    batch_data = {"img": [data["img"].unsqueeze(0)],
                  "img_metas": [[data["img_metas"]]]}

    return batch_data

def get_query_point(img, query_point):
    p = np.array(query_point)
    print('query point', p, ', intensity', img['img'][0][0, 0, p[0], p[1], p[2]])
    return p

def get_sim_embed_semantic_loc(query_img_info, key_img_info, query_point, imshape, use_sim_coarse=True):
    query_point = np.array(query_point)
    query_point_fine_re = np.floor(query_point / 2.).astype(int)
    
    fine_query = query_img_info[0]
    coarse_query = query_img_info[1]
    seman_query = query_img_info[2]
    coarse_query = F.interpolate(coarse_query, fine_query.shape[2:], mode='trilinear', align_corners=False)
    coarse_query = F.normalize(coarse_query, dim=1)

    fine_key = key_img_info[0]
    coarse_key = key_img_info[1]
    seman_key = key_img_info[2]
    coarse_key = F.interpolate(coarse_key, fine_key.shape[2:], mode='trilinear', align_corners=False)
    coarse_key = F.normalize(coarse_key, dim=1)

    query_fine = fine_query[0, :, query_point_fine_re[0], query_point_fine_re[1], query_point_fine_re[2]].view(-1, 128)
    query_sem = seman_query[0, :, query_point_fine_re[0], query_point_fine_re[1], query_point_fine_re[2]].view(-1, 128)
    key_fine = fine_key[0, :, :, :, :].reshape(128, -1)
    key_sem = seman_key[0, :, :, :, :].reshape(128, -1)

    query_coarse = coarse_query[0, :, query_point_fine_re[0], query_point_fine_re[1], query_point_fine_re[2]].view(-1, 128)
    key_coarse = coarse_key[0, :, :, :, :].reshape(128, -1)

    sim_fine = torch.einsum("nc,ck->nk", query_fine, key_fine)
    sim_sem = torch.einsum("nc,ck->nk", query_sem, key_sem)
    sim_coarse = torch.einsum("nc,ck->nk", query_coarse, key_coarse)

    sim_fine = sim_fine.reshape(fine_key.shape[2:])
    sim_sem = sim_sem.reshape(fine_key.shape[2:])
    sim_coarse = sim_coarse.reshape(coarse_key.shape[2:])
    sim_fine = sim_fine.view(1, 1, sim_fine.shape[0], sim_fine.shape[1], sim_fine.shape[2])
    sim_sem = sim_sem.view(1, 1, sim_sem.shape[0], sim_sem.shape[1], sim_sem.shape[2])
    sim_coarse = sim_coarse.view(1, 1, sim_coarse.shape[0], sim_coarse.shape[1], sim_coarse.shape[2])
    if use_sim_coarse:
        sim = (sim_fine + sim_coarse + sim_sem) / 3
    else:
        sim = (sim_sem + sim_fine) / 2

    sim = F.interpolate(sim, imshape, mode='trilinear', align_corners=False)
    sim = sim[0, 0, :, :, :]
    ind = torch.where(sim == sim.max())
    x = int(ind[0][0].data.cpu().numpy())
    y = int(ind[1][0].data.cpu().numpy())
    z = int(ind[2][0].data.cpu().numpy())
    
    return [x, y, z], sim.max().data.cpu().numpy()


def visualize(im1, im2, pt1, pt2, score, save_folder='', savename=None):
    print('visualizing ...')
    markersize = 10

    fig, ax = plt.subplots(3, 2, figsize=(10, 16))
    q_img = im1.transpose(2, 1, 0)
    q_img = q_img.astype(np.float32)

    slice = q_img[pt1[2], :, :]
    slice = slice[::-1, :]
    ax[0, 0].set_title('query')
    ax[0, 0].imshow(slice, cmap='gray')
    ax[0, 0].plot((pt1[0]), (q_img.shape[1] - pt1[1] - 1), 'o', markerfacecolor='none',
                  markeredgecolor="red",
                  markersize=markersize, markeredgewidth=2)
    
    slice = q_img[:, pt1[1], :]
    slice = slice[::-1, :]
    ax[1, 0].set_title('query')
    ax[1, 0].imshow(slice, cmap='gray')
    ax[1, 0].plot((pt1[0]), (q_img.shape[0] - pt1[2] - 1), 'o',
                  markerfacecolor='none', markeredgecolor="red",
                  markersize=markersize, markeredgewidth=2)

    slice = q_img[:, :, pt1[0]]
    slice = slice[::-1, :]
    ax[2, 0].set_title('query')
    ax[2, 0].imshow(slice, cmap='gray')
    ax[2, 0].plot((pt1[1]), (q_img.shape[0] - pt1[2] - 1), 'o',
                  markerfacecolor='none', markeredgecolor="red",
                  markersize=markersize, markeredgewidth=2)
    
    k_img = im2.transpose(2, 1, 0)
    k_img = k_img.astype(np.float32)

    slice = k_img[pt2[2], :, :]
    slice = slice[::-1, :]
    ax[0, 1].set_title('key')
    ax[0, 1].imshow(slice, cmap='gray')
    ax[0, 1].plot((pt2[0]), (k_img.shape[1] - pt2[1] - 1), 'o', markerfacecolor='none',
                  markeredgecolor="red",
                  markersize=markersize, markeredgewidth=2)

    slice = k_img[:, pt2[1], :]
    slice = slice[::-1, :]
    ax[1, 1].set_title('key')
    ax[1, 1].imshow(slice, cmap='gray')
    ax[1, 1].plot((pt2[0]), (k_img.shape[0] - pt2[2] - 1), 'o',
                  markerfacecolor='none',
                  markeredgecolor="red",
                  markersize=markersize, markeredgewidth=2)
    
    slice = k_img[:, :, pt2[0]]
    slice = slice[::-1, :]
    ax[2, 1].set_title('key')
    ax[2, 1].imshow(slice, cmap='gray')
    ax[2, 1].plot((pt2[1]), (k_img.shape[0] - pt2[2] - 1), 'o',
                  markerfacecolor='none',
                  markeredgecolor="red",
                  markersize=markersize, markeredgewidth=2)
    
    plt.suptitle(f'score:{score}')
    plt.tight_layout()
    if not savename is None:
        os.makedirs(save_folder, exist_ok=True)
        plt.savefig(os.path.join(save_folder,
            f'{savename}_{pt1[0]}_{pt1[1]}_{pt1[2]}_{pt2[0]}_{pt2[1]}_{pt2[2]}.png'), 
            dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
        plt.close()

def main():
    args = parse_args()
    
    # load model
    model = init(args.config, args.checkpoint)

    # load image
    img1 = load_image(args.img1_file, args.spacing)
    img2 = load_image(args.img2_file, args.spacing)

    pt1 = get_query_point(img1, args.query_point)

    emb1 = get_embedding(img1, model)
    emb2 = get_embedding(img2, model)

    pt2, score = get_sim_embed_semantic_loc(emb1, emb2, pt1,
                                            img2["img"][0].shape[2:])
    pt2 = np.array(pt2).astype(int)
    print(pt2, score)

    save_folder = os.path.join(args.save_dir, '_'.join(os.path.splitext(args.checkpoint)[0].split('/')[2:]),
                               os.path.basename(args.img1_file).split('.')[0] + '_' + os.path.basename(args.img2_file).split('.')[0])
    os.makedirs(save_folder, exist_ok=True)

    visualize(img1["img"][0][0, 0].cpu().numpy().astype(np.float32), 
              img2["img"][0][0, 0].cpu().numpy().astype(np.float32), 
              pt1, pt2, score,
              save_folder=save_folder, savename='demo')

    

if __name__ == '__main__':
    main()