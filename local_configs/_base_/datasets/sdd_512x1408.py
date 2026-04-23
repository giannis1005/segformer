# dataset settings
dataset_type = 'SDDDataset'
data_root = None
sdd_root = r'C:\Users\giann\OneDrive\Desktop\SDD'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
img_scale = (1408, 512)

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='BinaryMask'),
    dict(type='Resize', img_scale=img_scale, keep_ratio=False),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=False),
            dict(type='SetOriShapeToImgShape'),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=sdd_root,
        ann_dir=sdd_root,
        split='data/sdd/splits/train_2x_pos.txt',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=sdd_root,
        ann_dir=sdd_root,
        split='data/sdd/splits/val_pos11_neg70.txt',
        eval_shape=(512, 1408),
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir=sdd_root,
        ann_dir=sdd_root,
        split='data/sdd/splits/val_pos11_neg70.txt',
        eval_shape=(512, 1408),
        pipeline=test_pipeline))
