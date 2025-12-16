import numpy as np
from mmdet.evaluation.metrics import CocoMetric
from mmdet.registry import METRICS

"""Extension of CocoMetric that adds bbox_mAP_90 (AP at IoU=0.90)."""
@METRICS.register_module()
class CocoMetric90(CocoMetric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Register our custom metric so MMDet will print it
        self.metrics.append('bbox_mAP_90')

    def evaluate(self, size):
        """Override evaluate so we can inject bbox_mAP_90 reliably."""
        # Run standard COCO evaluation first
        results = super().evaluate(size)

        print(f"CocoMetric results : '{results}'" )
        # Access the internal COCOEval object
        coco_eval = self.coco_eval['bbox']
        precisions = coco_eval.eval['precision']

        if precisions is None:
            results['bbox_mAP_90'] = 0.0
            print(f"CocoMetric90 results : '{results}'" )
            return results

        # COCO IoU thresholds
        iou_thrs = coco_eval.params.iouThrs
        if 0.90 not in iou_thrs:
            results['bbox_mAP_90'] = 0.0
            print(f"CocoMetric90 results : '{results}'" )
            return results

        idx_90 = np.where(iou_thrs == 0.90)[0][0]

        # precision shape: [IoU, Recall, Class, Area, MaxDet]
        ap_90 = float(np.mean(precisions[idx_90]))

        # Inject custom metric
        results['bbox_mAP_90'] = ap_90

        print(f"CocoMetric90 results : '{results}'" )
        return results