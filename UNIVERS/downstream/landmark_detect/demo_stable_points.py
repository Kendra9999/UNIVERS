import argparse
import os
import sys
sys.path.append('..')
sys.path.append('../..')

import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    EnsureTyped,
    Orientationd,
    Spacingd,
    ScaleIntensityRanged
)

from demo import visualize
from fixed_point_iter import fixed_point_iterations
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
                        default='/mnt/sdb/drong/Project_adrenal/Data/ag/image/0004580443_325_0000.nii.gz', 
                        help='query image file path')
    parser.add_argument('--img2-file', type=str, 
                        default='/mnt/sdb/drong/Project_adrenal/Data/ag/image/21_021_0000.nii.gz',
                        help='key image file path')
    parser.add_argument('--query-point', type=list, default=(91, 78, 172), help='query point')
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

    pt2, score = fixed_point_iterations(emb1, emb2, pt1,
                                        img2["img"][0].shape[2:])
    pt2 = np.array(pt2).astype(int)
    print(pt2, score)

    save_folder = os.path.join(args.save_dir, '_'.join(os.path.splitext(args.checkpoint)[0].split('/')[2:]),
                               os.path.basename(args.img1_file).split('.')[0] + '_' + os.path.basename(args.img2_file).split('.')[0])
    os.makedirs(save_folder, exist_ok=True)

    visualize(img1["img"][0][0, 0].cpu().numpy().astype(np.float32), 
              img2["img"][0][0, 0].cpu().numpy().astype(np.float32), 
              pt1, pt2, score,
              save_folder=save_folder, savename='demo_stable_points')


if __name__ == '__main__':
    main()