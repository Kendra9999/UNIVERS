_base_ = '../_base_/default_runtime.py'

model = dict(
    type='UNIVERS',
    backbone=dict(
        type='PlainConvEncoder',
        input_channels=1,
        n_stages=6,
        features_per_stage=[32, 64, 128, 256, 320, 320],
        kernel_sizes=[3, 3, 3, 3, 3, 3],
        strides=[1, 2, 2, 2, 2, 2],
        n_conv_per_stage=[2, 2, 2, 2, 2, 2],
        return_skips=True),
    neck=dict(
        type='FPN3d',
        start_level=0,
        end_level=3,
        in_channels=[64, 128, 256],
        out_channels=128,
        num_outs=3,
        conv_cfg=dict(type='Conv3d')),
    read_out_head=dict(
        type='FPN3d',
        start_level=0,
        end_level=2,
        in_channels=[320, 320],
        out_channels=128,
        num_outs=2,
        conv_cfg=dict(type='Conv3d')),
    superloss=dict(type='SupConMeanLoss',
                   temperature=0.1,
                   base_temperature=1.0,
                   ),
    # model training and testing settings
    train_cfg=dict(
        intra_cfg=dict(
            pre_select_pos_number=2000,
            after_select_pos_number=100,
            pre_select_neg_number=2000,
            after_select_neg_number=500,
            positive_distance=2./2.,
            ignore_distance=20./2.,
            coarse_positive_distance=25./2.,
            coarse_ignore_distance=5./2.,
            coarse_z_thres=6.,
            coarse_pre_select_neg_number=250,
            coarse_after_select_neg_number=200,
            coarse_global_select_number=1000,
            temperature=0.5),
    ),
    test_cfg=dict(
        save_path='/data1/ydr/Project_representation/SSL_contrastive/work_dirs/results/',
        output_embedding=True
    ))


synthetic_data_root = "/mnt/sdb/drong/Project_representation/Data_gen/Totalsegmentator_gen_v1/"

overlap_patch_set = dict(
    type = 'Dataset3dSynthetic',
    data_dir = synthetic_data_root,
    crop_transform = True,
    crop_transform_kwargs = dict(
        crop_size = (96, 96, 96),
    ),
    geometric_transform = True,
    geometric_transform_prob = 0.5
)

whole_image_set = dict(
    type = 'Dataset3dSynthetic',
    data_dir = synthetic_data_root,
    geometric_transform = False,
    geometric_transform_prob = 0.5
)

data = dict(
    samples_per_gpu=3,
    workers_per_gpu=24,
    train=dict(
        type='ConcatDataset',
        datasets=[overlap_patch_set, whole_image_set]
    ),
    val=dict(),
    test=dict(), 
)


find_unused_parameters = True

# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=1000,
    gamma=0.95)
runner = dict(type="IterBasedRunner", max_iters=50000)
fp16 = dict(loss_scale="dynamic")


checkpoint_config = dict(by_epoch=False, interval=5000, max_keep_ckpts=5)
log_config = dict(
    interval=10,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        dict(
            type='WandbLoggerHook',
            init_kwargs=dict(
                project = "SSL_contrastive",
                entity = "1820037839-shanghai-jiao-tong-university",
                name = "univers",
            ),
        )
    ])