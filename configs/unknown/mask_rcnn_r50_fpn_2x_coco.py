_base_ = [
    '../_base_/models/mask_rcnn_r50_fpn.py',
    # '../_base_/datasets/coco_instance.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

test_cfg = dict(
    rpn=dict(
        nms_across_levels=False,
        nms_pre=1000,
        nms_post=1000,
        max_num=1000,
        nms_thr=0.7,
        min_bbox_size=0),
    rcnn=dict(
        score_thr=0.05,
        nms=dict(type='nms', iou_thr=0.5),
        max_per_img=100,
        mask_thr_binary=0.5))


dataset_type = 'ADE20kCOCODataset'
data_root = 'data/ADE20K/'
img_norm_cfg = dict(
    # refer https://github.com/CSAILVision/semantic-segmentation-pytorch/blob/master/mit_semseg/dataset.py
    # get the same mean / std values.
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instance_training_gts.json',
        img_prefix=data_root + '/images/training/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instance_validation_gts.json',
        ann_file=data_root + 'annotations/instance_validation_unknown_23classes.json',
        img_prefix=data_root + '/images/validation/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        # ann_file=data_root + 'annotations/instance_validation_gts.json',
        ann_file=data_root + 'annotations/instance_validation_unknown_23classes.json',
        img_prefix=data_root + '/images/validation/',
        pipeline=test_pipeline))
evaluation = dict(metric=['segm'])
work_dir = './work_dirs/mask_rcnn_r50_fpn_2x_ade20k'
