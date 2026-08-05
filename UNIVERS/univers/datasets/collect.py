from collections.abc import Mapping, Sequence

import torch

def collate(batch):
    """
    Puts each data field into a dict of
       Keys: "img": a dict of keys:
                    "overlap_patches": a Tensor of shape (N, C, D, H, W)
                    "overlap_patches_girds": a Tensor of shape (N, 3, D, H, W)
                    "whole_images": a Tensor of shape (N, C, D, H, W)
                    "whole_images_labels": a Tensor of shape (N, 1, D, H, W)
             "img_metas": a list of dict of keys:
                    "style": "overlap" or "whole"
    """
    if not isinstance(batch, Sequence):
        raise TypeError(f"{batch.dtype} is not supported.")
    
    r_batch = {"img": {}, "img_metas": []}

    all_overlap_patches = []
    all_overlap_patches_girds = []
    all_whole_images = []
    all_whole_images_labels = []

    for data in batch:
        if "view1_p1" in data:
            r_batch["img_metas"].append({"style": "overlap"})

            all_overlap_patches.append(data["view1_p1"])
            all_overlap_patches_girds.append(data["view1_p1_mesh"])
            all_overlap_patches.append(data["view1_p2"])
            all_overlap_patches_girds.append(data["view1_p2_mesh"])
            all_overlap_patches.append(data["view2_p1"])
            all_overlap_patches_girds.append(data["view2_p1_mesh"])
            all_overlap_patches.append(data["view2_p2"])
            all_overlap_patches_girds.append(data["view2_p2_mesh"])
        
        else:
            r_batch["img_metas"].append({"style": "whole"})

            if "label1" in data:
                all_whole_images.append(data["view1"])
                all_whole_images_labels.append(data["label1"])
                all_whole_images.append(data["view2"])
                all_whole_images_labels.append(data["label2"])
            else:
                all_whole_images.append(data["view1"])
                all_whole_images_labels.append(data["label"])
                all_whole_images.append(data["view2"])
                all_whole_images_labels.append(data["label"])
    
    r_batch["img"]["overlap_patches"] = torch.stack(all_overlap_patches, dim=0)
    r_batch["img"]["overlap_patches_girds"] = torch.stack(all_overlap_patches_girds, dim=0)
    r_batch["img"]["whole_images"] = torch.stack(all_whole_images, dim=0)
    r_batch["img"]["whole_images_labels"] = torch.stack(all_whole_images_labels, dim=0)

    return r_batch
            
