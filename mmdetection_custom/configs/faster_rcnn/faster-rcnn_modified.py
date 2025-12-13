_base_ = [
    '../../../../mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection_modified.py',
    '../_base_/schedules/schedule_1x_modified.py', 
    '../../../../mmdetection/configs/_base_/default_runtime.py'
]

custom_imports = dict(
    imports = _base_.custom_imports.imports + ['mmdetection_custom.hooks.moving_avg_hook'],
    allow_failed_imports=False
)

custom_hooks = [
    dict(type='MovingAvgLossHook', window_size=1000, key='train/loss', out_key='train/loss_avg_1K'),
    dict(type='MovingAvgLossHook', window_size=10000, key='train/loss', out_key='train/loss_avg_10K'),
    dict(type='MovingAvgLossHook', window_size=10000, key='train/acc', out_key='train/acc_avg_10K'),
]

visualizer_config = dict(
    project_name = 'EugeneSandbox',
    task_name = 'faster-rcnn_r50',
    tags=[""]
)


visualizer = dict(
    vis_backends=[
        dict(type='LocalVisBackend'),
        dict(type='ClearMLVisBackend',
            init_kwargs = visualizer_config,
            artifact_suffix = ('.py', 'yaml', '.pth', 'jarvis')
        )
    ]
)
