import copy
import numpy as np
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Transform
from monai.transforms import (
    Compose,
    SpatialCropd,
    Resized,
    RandGaussianNoised,
    ScaleIntensityd
)

def meshgrid3d(shape, spacing):
    y_ = np.linspace(0., (shape[1] - 1) * spacing[0], shape[1])
    x_ = np.linspace(0., (shape[2] - 1) * spacing[1], shape[2])
    z_ = np.linspace(0., (shape[3] - 1) * spacing[2], shape[3])

    x, y, z = np.meshgrid(x_, y_, z_, indexing='xy')
    mesh = np.stack([y, x, z], axis=0)
    return mesh

class Create3DMeshd(Transform):
    """
    Create 3d mesh grid for the volumes
    """
    def __init__(self, keys: list[str]):
        self.keys = keys

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            mesh = meshgrid3d(img.shape, spacing=[1., 1., 1.]) # synthetic data, set spacing to 1 mm
            d[key + "_mesh"] = MetaTensor.ensure_torch_and_prune_meta(
                                    mesh, copy.deepcopy(img.meta))
        return d
    
class Crop3DPatch(Transform):
    """
    Crop two overlap patches from a 3D volume.
    """
    def __init__(self, keys: list[str], scale=(0.8, 1.2), patch_size=(96, 96, 96)):
        self.keys = keys
        self.scale = scale
        self.patch_size = patch_size

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            img = d[key]
            mesh = d[key + "_mesh"]
            
            img_p1 = copy.deepcopy(img)
            img_p2 = copy.deepcopy(img)
            mesh_p1 = copy.deepcopy(mesh)
            mesh_p2 = copy.deepcopy(mesh)
            d[key + "_p1"] = img_p1
            d[key + "_p2"] = img_p2
            d[key + "_p1_mesh"] = mesh_p1
            d[key + "_p2_mesh"] = mesh_p2


            p1_scale = np.random.uniform(self.scale[0], self.scale[1])
            p2_scale = np.random.uniform(self.scale[0], self.scale[1])
            origin_size = img.shape[1:]

            p1_crop_size = np.ceil(np.array(self.patch_size) * p1_scale).astype(int)
            p2_crop_size = np.ceil(np.array(self.patch_size) * p2_scale).astype(int)
            
            margin_x = 0
            margin_y = 0
            margin_z = 0
            rangex_min = margin_x
            rangex_max = origin_size[1] - margin_x + 1
            rangey_min = margin_y
            rangey_max = origin_size[0] - margin_y + 1
            rangez_min = margin_z
            rangez_max = origin_size[2] - margin_z + 1

            p1_left_up = (np.random.randint(rangey_min, rangey_max - p1_crop_size[0]),
                          np.random.randint(rangex_min, rangex_max - p1_crop_size[1]),
                          np.random.randint(rangez_min, rangez_max - p1_crop_size[2]))
            p1_left_up = np.asarray(p1_left_up)
            p2_left_up = (np.random.randint(rangey_min, rangey_max - p2_crop_size[0]),
                          np.random.randint(rangex_min, rangex_max - p2_crop_size[1]),
                          np.random.randint(rangez_min, rangez_max - p2_crop_size[2]))
            p2_left_up = np.asarray(p2_left_up)

            p1_y1x1z1_crop = p1_left_up
            p1_y2x2z2_crop = p1_left_up + p1_crop_size
            p2_y1x1z1_crop = p2_left_up
            p2_y2x2z2_crop = p2_left_up + p2_crop_size

            p1_crop_transform = SpatialCropd(
                    keys=[key + "_p1", key + "_p1_mesh"],
                    roi_start=(p1_y1x1z1_crop[0], p1_y1x1z1_crop[1], p1_y1x1z1_crop[2]),
                    roi_end=(p1_y2x2z2_crop[0], p1_y2x2z2_crop[1], p1_y2x2z2_crop[2]),
                    )
            p2_crop_transform = SpatialCropd(
                    keys=[key + "_p2", key + "_p2_mesh"],
                    roi_start=(p2_y1x1z1_crop[0], p2_y1x1z1_crop[1], p2_y1x1z1_crop[2]),
                    roi_end=(p2_y2x2z2_crop[0], p2_y2x2z2_crop[1], p2_y2x2z2_crop[2]),
                    )
            d = p1_crop_transform(d)
            d = p2_crop_transform(d)

            resize_transform = Resized(
                    keys=[key + "_p1", key + "_p1_mesh", key + "_p2", key + "_p2_mesh"],
                    spatial_size=self.patch_size,
                    mode=("area", "bilinear", "area", "bilinear"),
                )
            d = resize_transform(d)
            
        return d

def get_crop_transforms(crop_size = (96, 96, 96)):
    """
    Crop two overlap patches from a 3D volume.
    """
    train_transforms = Compose(
        [
            Create3DMeshd(keys=["view1", "view2"]),
            Crop3DPatch(keys=["view1", "view2"], patch_size=crop_size),
            # Apply Gaussian noise:
            RandGaussianNoised(
                keys=["view1_p1"],
                prob=0.5,
                mean=0.0,
                std=0.001,
            ),
            RandGaussianNoised(
                keys=["view1_p2"],
                prob=0.5,
                mean=0.0,
                std=0.001,
            ),
            RandGaussianNoised(
                keys=["view2_p1"],
                prob=0.5,
                mean=0.0,
                std=0.001,
            ),
            RandGaussianNoised(
                keys=["view2_p2"],
                prob=0.5,
                mean=0.0,
                std=0.001,
            ),
            ScaleIntensityd(keys=["view1_p1", "view1_p2", "view2_p1", "view2_p2"]),
        ]
    )

    return train_transforms


if __name__ == "__main__":
    import os
    import matplotlib.pyplot as plt
    from transforms import get_online_transforms

    # image_fn = "foreground_masked_shapes31_ZYZ45ZR.nii.gz"
    # image_fn = "unconstrained_shapes38_QXN2T0J.nii.gz"
    image_fn = "foreground_masked_enveloped_shapes34_59WSV6Z.nii.gz"
    label_dir = "/mnt/sdb/drong/Project_representation/Data_gen/Totalsegmentator_gen_v1/label_ensembles"
    image_view1_dir = "/mnt/sdb/drong/Project_representation/Data_gen/Totalsegmentator_gen_v1/synthesized_views/view1"
    image_view2_dir = "/mnt/sdb/drong/Project_representation/Data_gen/Totalsegmentator_gen_v1/synthesized_views/view2"

    data = {"view1": os.path.join(image_view1_dir, "view1_" + image_fn),
            "view2": os.path.join(image_view2_dir, "view2_" + image_fn),
            "label": os.path.join(label_dir, image_fn)}
    
    transform_data = get_online_transforms()(data)
    
    crop_data = get_crop_transforms()(transform_data)

    fig, ax = plt.subplots(1, 3, figsize=(30, 10))
    ax[0].set_title("view1")
    ax[0].imshow(transform_data["view1"][0, :, :, 60], cmap="gray")
    ax[1].set_title("view2")
    ax[1].imshow(transform_data["view2"][0, :, :, 60], cmap="gray")
    ax[2].set_title("label")
    ax[2].imshow(transform_data["label"][0, :, :, 60], cmap="gray")
    plt.savefig("tmp_after_transform.png")

    
    fig, ax = plt.subplots(3, 4, figsize=(40, 30))
    ax[0, 0].set_title("view1")
    ax[0, 0].imshow(transform_data["view1"][0, :, :, 60], cmap="gray")
    ax[1, 0].set_title("view1_p1")
    ax[1, 0].imshow(crop_data["view1_p1"][0, :, :, 60], cmap="gray")
    ax[2, 0].set_title("view1_p2")
    ax[2, 0].imshow(crop_data["view1_p2"][0, :, :, 60], cmap="gray")
    ax[0, 1].set_title("view1 mesh")
    ax[0, 1].imshow(crop_data["view1_mesh"][:, :, :, 60].permute(1, 2, 0) / 128.0)
    ax[1, 1].set_title("view1_p1 mesh")
    ax[1, 1].imshow(crop_data["view1_p1_mesh"][:, :, :, 60].permute(1, 2, 0) / 128.0)
    ax[2, 1].set_title("view1_p2 mesh")
    ax[2, 1].imshow(crop_data["view1_p2_mesh"][:, :, :, 60].permute(1, 2, 0) / 128.0)
    ax[0, 2].set_title("view2")
    ax[0, 2].imshow(transform_data["view2"][0, :, :, 60], cmap="gray")
    ax[1, 2].set_title("view2_p1")
    ax[1, 2].imshow(crop_data["view2_p1"][0, :, :, 60], cmap="gray")
    ax[2, 2].set_title("view2_p2")
    ax[2, 2].imshow(crop_data["view2_p2"][0, :, :, 60], cmap="gray")
    ax[0, 3].set_title("view2 mesh")
    ax[0, 3].imshow(crop_data["view2_mesh"][:, :, :, 60].permute(1, 2, 0) / 128.0)
    ax[1, 3].set_title("view2_p1 mesh")
    ax[1, 3].imshow(crop_data["view2_p1_mesh"][:, :, :, 60].permute(1, 2, 0) / 128.0)
    ax[2, 3].set_title("view2_p2 mesh")
    ax[2, 3].imshow(crop_data["view2_p2_mesh"][:, :, :, 60].permute(1, 2, 0) / 128.0)
    plt.savefig("tmp_after_crop.png")


    import torch

    qt = (40, 40, 40)
    query_mesh = crop_data["view1_p1_mesh"][:, qt[0], qt[1], qt[2]]
    print (query_mesh)
    
    view1_p2_kt = torch.norm(crop_data["view1_p2_mesh"] - query_mesh[:, None, None, None], dim=0)
    view1_p2_kt = torch.where(view1_p2_kt == view1_p2_kt.min())
    print (view1_p2_kt)
    view2_p1_kt = torch.norm(crop_data["view2_p1_mesh"] - query_mesh[:, None, None, None], dim=0)
    view2_p1_kt = torch.where(view2_p1_kt == view2_p1_kt.min())
    print (view2_p1_kt)
    view2_p2_kt = torch.norm(crop_data["view2_p2_mesh"] - query_mesh[:, None, None, None], dim=0)
    view2_p2_kt = torch.where(view2_p2_kt == view2_p2_kt.min())
    print (view2_p2_kt)

    fig, ax = plt.subplots(1, 4, figsize=(40, 10))
    ax[0].set_title("view1_p1")
    ax[0].imshow(crop_data["view1_p1"][0, qt[0]-20:qt[0]+20, qt[1]-20:qt[1]+20, qt[2]], cmap="gray")
    ax[1].set_title("view1_p2")
    ax[1].imshow(crop_data["view1_p2"][0, view1_p2_kt[0][0]-20:view1_p2_kt[0][0]+20, 
                                       view1_p2_kt[1][0]-20:view1_p2_kt[1][0]+20, view1_p2_kt[2][0]], cmap="gray")
    ax[2].set_title("view2_p1")
    ax[2].imshow(crop_data["view2_p1"][0, view2_p1_kt[0][0]-20:view2_p1_kt[0][0]+20,
                                       view2_p1_kt[1][0]-20:view2_p1_kt[1][0]+20, view2_p1_kt[2][0]], cmap="gray")
    ax[3].set_title("view2_p2")
    ax[3].imshow(crop_data["view2_p2"][0, view2_p2_kt[0][0]-20:view2_p2_kt[0][0]+20,
                                       view2_p2_kt[1][0]-20:view2_p2_kt[1][0]+20, view2_p2_kt[2][0]], cmap="gray")
    plt.savefig("tmp_same_point_from_mesh.png")