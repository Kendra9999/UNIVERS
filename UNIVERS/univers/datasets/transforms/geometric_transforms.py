import numpy as np
import copy
from monai.transforms import Transform
from monai.transforms import (
    Compose,
    CropForegroundd,
    RandAxisFlipd,
    RandAffined,
    ScaleIntensityd
)
from .crop_transforms import *

class CreateKeysbyCopyd(Transform):
    """
    Create new keys by copying the values of existing keys.
    """
    def __init__(self, old_key: str, new_keys: list[str]):
        self.old_key = old_key
        self.new_keys = new_keys

    def __call__(self, data):
        d = dict(data)
        for key in self.new_keys:
            d[key] = copy.deepcopy(d[self.old_key])
        del d[self.old_key]
        return d
    

def get_geometric_transforms(crop_transforms=False, crop_transform_kwargs=None):
    if not crop_transforms:
        geo_transforms = Compose(
            [
                CreateKeysbyCopyd(
                    old_key="label",
                    new_keys=["label1", "label2"],
                ),
                RandAffined(
                    keys=["view2", "label2"],
                    prob=0.98,
                    mode=['bilinear', 'nearest'],
                    translate_range=(10, 10, 10),
                    rotate_range=(np.pi/6, np.pi/6, np.pi/6),
                    scale_range=(0.25, 0.25, 0.25),
                ),
                # Rescale to [0, 1]:
                ScaleIntensityd(keys=["view1", "view2"]),
            ]
        )
    else:
        geo_transforms = Compose(
            [
                CropForegroundd(
                    keys=["view1", "view2", "label"], 
                    source_key="label",
                    select_fn=lambda x: x > 0,
                ),
                Resized(
                    keys=["view1", "view2", "label"],
                    spatial_size=crop_transform_kwargs["crop_size"],
                    mode=("area", "area", "nearest"),
                ),
                Create3DMeshd(keys=["view1", "view2"]),
                CreateKeysbyCopyd(
                    old_key="view1",
                    new_keys=["view1_p1", "view1_p2"],
                ),
                CreateKeysbyCopyd(
                    old_key="view1_mesh",
                    new_keys=["view1_p1_mesh", "view1_p2_mesh"],
                ),
                CreateKeysbyCopyd(
                    old_key="view2",
                    new_keys=["view2_p1", "view2_p2"],
                ),
                CreateKeysbyCopyd(
                    old_key="view2_mesh",
                    new_keys=["view2_p1_mesh", "view2_p2_mesh"],
                ),
                RandAffined(
                    keys=["view1_p2", "view1_p2_mesh"],
                    prob=0.98,
                    mode=['bilinear', 'bilinear'],
                    translate_range=(10, 10, 10),
                    rotate_range=(np.pi/6, np.pi/6, np.pi/6),
                    scale_range=(0.25, 0.25, 0.25),
                ),
                RandAffined(
                    keys=["view2_p1", "view2_p1_mesh"],
                    prob=0.98,
                    mode=['bilinear', 'bilinear'],
                    translate_range=(10, 10, 10),
                    rotate_range=(np.pi/6, np.pi/6, np.pi/6),
                    scale_range=(0.25, 0.25, 0.25),
                ),
                RandAffined(
                    keys=["view2_p2", "view2_p2_mesh"],
                    prob=0.98,
                    mode=['bilinear', 'bilinear'],
                    translate_range=(10, 10, 10),
                    rotate_range=(np.pi/6, np.pi/6, np.pi/6),
                    scale_range=(0.25, 0.25, 0.25),
                ),
                # Rescale to [0, 1]:
                ScaleIntensityd(keys=["view1_p1", "view1_p2", "view2_p1", "view2_p2"]),
            ]
        )

    return geo_transforms


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
    
    geo_data = get_geometric_transforms()(transform_data)

    fig, ax = plt.subplots(1, 4, figsize=(40, 10))
    ax[0].set_title("view1")
    ax[0].imshow(geo_data["view1"][0, :, :, 60], cmap="gray")
    ax[1].set_title("view2")
    ax[1].imshow(geo_data["view2"][0, :, :, 60], cmap="gray")
    ax[2].set_title("label1")
    ax[2].imshow(geo_data["label1"][0, :, :, 60], cmap="gray")
    ax[3].set_title("label2")
    ax[3].imshow(geo_data["label2"][0, :, :, 60], cmap="gray")
    plt.savefig("tmp_after_geo_transform.png")
