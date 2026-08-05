import argparse
import os
import sys
sys.path.append('..')
sys.path.append('../..')
import json

import numpy as np
import torchio as tio
from torch.utils.data import Dataset, DataLoader

from demo import get_sim_embed_semantic_loc
from fixed_point_iter import fixed_point_iterations
from utils import *

os.chdir(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))  # go to root dir of this project

def parse_args():
    parser = argparse.ArgumentParser(description='Contrastive PreTrain')
    parser.add_argument('--config', type=str, 
                        default='configs/univers/univers.py', 
                        help='train config file path')
    parser.add_argument('--checkpoint', type=str, 
                        default='work_dirs/works/univers/20260629_155555/iter_35000.pth', 
                        help='pretrain checkpoint file path')
    parser.add_argument('--data-path', type=str, 
                        default='/mnt/sdb/drong/Project_representation/Data_preprocess/DeepLesion/Images_nifti_test/', 
                        help='test data directory path')
    parser.add_argument('--anno-path', type=str, 
                        default='/mnt/sdb/drong/Project_representation/Data/DLT-main/data/', 
                        help='test data annotation directory path')
    parser.add_argument('--spacing', type=list, default=(2, 2, 2), help='spacing')
    parser.add_argument('--save-dir', type=str, 
                        default='work_dirs/dlt_results/', 
                        help='save DLT test results directory path')
    args = parser.parse_args()
    return args


def load_image(img_path, re_orient=True, spacing=(2., 2., 2.)):
    img_tio = tio.ScalarImage(img_path)
    ToCanonical = tio.ToCanonical()
    img_tio = ToCanonical(img_tio)
    img_tio.data = torch.flip(img_tio.data, (1, 2))
    img_tio.affine = np.array(
        [-img_tio.affine[0, :], -img_tio.affine[1, :], img_tio.affine[2, :], img_tio.affine[3, :]])
    
    if re_orient:
        img_data = img_tio.data
        img_tio.data = img_data.permute(0, 2, 1, 3)
        img_tio.affine = np.array(
            [img_tio.affine[1, :], img_tio.affine[0, :], img_tio.affine[2, :], img_tio.affine[3, :]])
    
    subject = tio.Subject(image=img_tio)
    resample_transform = tio.Resample(target=spacing)
    subject = resample_transform(subject)

    # intensity scale
    img = subject.image.data.type(torch.FloatTensor)
    img = img.clamp(min=-175.0, max=250.0)
    img = (img - img.min()) / (img.max() - img.min())
    
    img = img.permute(0, 3, 1, 2)

    return img        



class DLTTestDataset(Dataset):
    def __init__(self, data_path, anno_path, spacing):
        self.data_path = data_path
        self.gths = json.load(open(os.path.join(anno_path, 'test.json'), 'r'))
        self.all_keys = list(self.gths.keys())

        self.spacing = spacing
    
    def __len__(self):
        return len(self.all_keys)
    
    def __getitem__(self, index):
        element = self.gths[self.all_keys[index]]
        query_img_name = element['source']
        key_img_name = element['target']

        query_bbx = element['source box']  # [x1,y1,z1,x2-x1,y2-y1,z2-z1] z index is the inverse of the sever deeplesion dataset
        query_center = element['source center']
        key_bbx = element['target box']
        key_center = element['target center']

        query_img = load_image(os.path.join(self.data_path, query_img_name), spacing=self.spacing)
        key_img = load_image(os.path.join(self.data_path, key_img_name), spacing=self.spacing)
        
        origin_query_img = tio.ScalarImage(os.path.join(self.data_path, query_img_name))
        query_img_origin_spacing = origin_query_img.spacing
        query_img_origin_shape = origin_query_img.shape
        query_spacing_norm_ratio = np.array(query_img_origin_spacing) / np.array(self.spacing)
        query_bbx_reshaped = np.array(query_bbx).reshape(2, 3)
        query_bbx_reshaped[1, :] = query_bbx_reshaped[1, :] + query_bbx_reshaped[0, :]
        query_bbx_reshaped[:, 2] = query_img_origin_shape[3] - query_bbx_reshaped[:, 2]
        query_bbx_reshaped[:, 2] = query_bbx_reshaped[::-1, 2]
        query_bbx_normed = query_bbx_reshaped * query_spacing_norm_ratio
        query_center[2] = query_img_origin_shape[3] - query_center[2] - 1

        query_point_fine = (query_bbx_reshaped[1, :] + query_bbx_reshaped[0, :]) / 2 * query_spacing_norm_ratio
        query_center_fine = query_center * query_spacing_norm_ratio
        query_point_fine = query_center_fine

        
        origin_key_img = tio.ScalarImage(os.path.join(self.data_path, key_img_name))
        key_img_origin_spacing = origin_key_img.spacing
        key_img_origin_shape = origin_key_img.shape
        key_spacing_norm_ratio = np.array(key_img_origin_spacing) / np.array(self.spacing)
        key_bbx_reshaped = np.array(key_bbx).reshape(2, 3)
        key_bbx_reshaped[1, :] = key_bbx_reshaped[1, :] + key_bbx_reshaped[0, :]
        key_bbx_reshaped[:, 2] = key_img_origin_shape[3] - key_bbx_reshaped[:, 2]
        key_bbx_reshaped[:, 2] = key_bbx_reshaped[::-1, 2]
        key_bbx_normed = key_bbx_reshaped * key_spacing_norm_ratio
        key_center[2] = key_img_origin_shape[3] - key_center[2] - 1

        key_point_fine_gt = (key_bbx_reshaped[1, :] + key_bbx_reshaped[0, :]) / 2 * key_spacing_norm_ratio
        key_center_fine = key_center * key_spacing_norm_ratio
        key_point_fine_gt = key_center_fine
        
        return query_img, key_img, \
            query_bbx_reshaped, query_bbx_normed, query_point_fine, \
            key_bbx_reshaped, key_bbx_normed, key_point_fine_gt, \
            query_spacing_norm_ratio, key_spacing_norm_ratio,\
            query_img_name, key_img_name



def main():
    args = parse_args()
    
    # load model
    model = init(args.config, args.checkpoint)

    # load test data
    test_dataset = DLTTestDataset(args.data_path, args.anno_path, args.spacing)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=12)

    all_dis = []
    all_dis_x = []
    all_dis_y = []
    all_dis_z = []

    count = 0
    hit = 0
    miss = 0
    hit_10 = 0
    miss_10 = 0

    for batch_data in test_dataloader:
        count += 1

        query_img, key_img, \
        query_bbx_reshaped, query_bbx_normed, query_point_fine, \
        key_bbx_reshaped, key_bbx_normed, key_point_fine_gt, \
        query_spacing_norm_ratio, key_spacing_norm_ratio,\
        query_img_name, key_img_name = batch_data

        query_bbx_reshaped, query_bbx_normed, query_point_fine, \
        key_bbx_reshaped, key_bbx_normed, key_point_fine_gt, \
        query_spacing_norm_ratio, key_spacing_norm_ratio,\
        query_img_name, key_img_name = query_bbx_reshaped[0].numpy(), query_bbx_normed[0].numpy(), query_point_fine[0].numpy(), \
                            key_bbx_reshaped[0].numpy(), key_bbx_normed[0].numpy(), key_point_fine_gt[0].numpy(), \
                            query_spacing_norm_ratio[0].numpy(), key_spacing_norm_ratio[0].numpy(), \
                            query_img_name[0], key_img_name[0]

        query_img_batch = {"img": [query_img],
                  "img_metas": [[{"filename": query_img_name}]]}
        key_img_batch = {"img": [key_img],
                  "img_metas": [[{"filename": key_img_name}]]}
        
        query_img_info = get_embedding(query_img_batch, model)
        key_img_info = get_embedding(key_img_batch, model)

        pt1 = np.array([query_point_fine[2], query_point_fine[1], query_point_fine[0]])
        # pt2, score = get_sim_embed_semantic_loc(query_img_info, key_img_info, pt1,
        #                                         key_img_batch["img"][0].shape[2:])
        pt2, score = fixed_point_iterations(query_img_info, key_img_info, pt1,
                                            key_img_batch["img"][0].shape[2:])
        pt2 = np.array([pt2[2],pt2[1],pt2[0]])
        
        
        # Spacing 2x2x2 mm
        dis = np.linalg.norm((key_point_fine_gt - pt2)) * 2.
        dis_x = np.linalg.norm((key_point_fine_gt[0] - pt2[0])) * 2.
        dis_y = np.linalg.norm((key_point_fine_gt[1] - pt2[1])) * 2.
        dis_z = np.linalg.norm((key_point_fine_gt[2] - pt2[2])) * 2.
        all_dis.append(dis)
        all_dis_x.append(dis_x)
        all_dis_y.append(dis_y)
        all_dis_z.append(dis_z)

        pt2_remap = pt2 / key_spacing_norm_ratio
        final = key_bbx_reshaped - pt2_remap
        if (final[0, :] <= 0).all() and (final[1, :] > -0).all():
            hit = hit + 1
        else:
            miss = miss + 1
        
        if dis <= 10:
            hit_10 = hit_10 + 1
        else:
            miss_10 = miss_10 + 1

        print (f"query: {query_img_name}, key: {key_img_name}")
        print (f"query point: {query_point_fine}, key point: {key_point_fine_gt}")
        print (f"dis: {dis}, dis_x: {dis_x}, dis_y: {dis_y}, dis_z: {dis_z}")

    print (f"total case: {count}, number of matching within the bboxes: {hit}, number of failed cases: {miss}")
    print(f"CPM@Radius: {hit / count}")
    print (f"total case: {count}, number of matching within 10: {hit_10}, number of failed cases: {miss_10}")
    print(f"CPM@10MM: {hit_10 / count}")
    print(f"MED: {np.array(all_dis).mean()}, {np.array(all_dis).std()}")
    print(f"MED_X: {np.array(all_dis_x).mean()}, {np.array(all_dis_x).std()}")
    print(f"MED_Y: {np.array(all_dis_y).mean()}, {np.array(all_dis_y).std()}")
    print(f"MED_Z: {np.array(all_dis_z).mean()}, {np.array(all_dis_z).std()}")
    
    save_folder = os.path.join(args.save_dir, '_'.join(os.path.splitext(args.checkpoint)[0].split('/')[2:]))
    os.makedirs(save_folder, exist_ok=True)

    with open(os.path.join(save_folder, 'results_stable.txt'), 'w') as f:
        f.write(f"total case: {count}, number of matching within the bboxes: {hit}, number of failed cases: {miss}\n")
        f.write(f"CPM@Radius: {hit / count}\n")
        f.write(f"total case: {count}, number of matching within 10: {hit_10}, number of failed cases: {miss_10}\n")
        f.write(f"CPM@10MM: {hit_10 / count}\n")
        f.write(f"MED: {np.array(all_dis).mean()}, {np.array(all_dis).std()}\n")
        f.write(f"MED_X: {np.array(all_dis_x).mean()}, {np.array(all_dis_x).std()}\n")
        f.write(f"MED_Y: {np.array(all_dis_y).mean()}, {np.array(all_dis_y).std()}\n")
        f.write(f"MED_Z: {np.array(all_dis_z).mean()}, {np.array(all_dis_z).std()}\n")



if __name__ == '__main__':
    main()