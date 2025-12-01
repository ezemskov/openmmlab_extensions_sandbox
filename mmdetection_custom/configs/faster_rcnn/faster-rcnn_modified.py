_base_ = [
    '../../../../mmdetection/configs/_base_/models/faster-rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection_modified.py',
    '../_base_/schedules/schedule_1x.py', 
    '../../../../mmdetection/configs/_base_/default_runtime.py'
]
