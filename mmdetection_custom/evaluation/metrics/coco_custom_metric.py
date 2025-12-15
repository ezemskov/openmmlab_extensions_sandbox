import numpy as np
from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS


@METRICS.register_module()
class CocoMetric90(CocoMetric):
    """Extension of CocoMetric that adds bbox_mAP_90 (AP at IoU=0.90)."""

    def results2dict(self, coco_eval, classwise=False):
        # Get the standard evaluation dictionary
        result_dict = super().results2dict(coco_eval, classwise)

        # Extract precision matrix: [IoU, Recall, Class, Area, MaxDet]
        precisions = coco_eval.eval['precision']
        if precisions is None:
            # No detections => AP_90 = 0
            result_dict['bbox_mAP_90'] = 0.0
            return result_dict

        # Find the IoU threshold index that corresponds to 0.90
        iou_thresholds = coco_eval.params.iouThrs
        if 0.90 not in iou_thresholds:
            # Should never happen, but safe fallback
            result_dict['bbox_mAP_90'] = 0.0
            return result_dict

        idx_90 = np.where(iou_thresholds == 0.90)[0][0]

        # According to COCOEval, AP is mean over:
        # recalls, categories, areas, maxDets
        ap_90 = np.mean(precisions[idx_90])

        # Store AP@90
        result_dict['bbox_mAP_90'] = float(ap_90)

        return result_dict