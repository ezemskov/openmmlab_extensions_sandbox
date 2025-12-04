_base_ = [
    '../../../../mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection_modified.py',
    '../_base_/schedules/schedule_1x_modified.py', 
    '../../../../mmdetection/configs/_base_/default_runtime.py'
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
