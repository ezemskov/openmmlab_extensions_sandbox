import numpy as np
from mmdet.datasets.api_wrappers import COCOeval, COCOevalMP
from mmdet.evaluation.metrics import CocoMetric
from mmengine.fileio import load
from mmdet.registry import METRICS
from typing import Dict
import os.path as osp
import tempfile

"""Extension of CocoMetric that adds bbox_mAP_90 (AP at IoU=0.90)."""
@METRICS.register_module()
class CocoMetric90(CocoMetric):

    def compute_metrics(self, results: list) -> Dict[str, float]:
        res_metrics = super().compute_metrics(results)
        
        print("CocoMetric results calculated")

        # split gt and prediction list
        gts, preds = zip(*results)
        tmp_dir = tempfile.TemporaryDirectory()
        outfile_prefix = osp.join(tmp_dir.name, 'results')
        # convert predictions to coco format and dump to json file
        result_files = self.results2json(preds, outfile_prefix)

        metric = 'bbox'
        if metric not in result_files:
            raise KeyError(f'{metric} is not in result_files')

        predictions = load(result_files[metric])
        coco_dt = self._coco_api.loadRes(predictions)

        if self.use_mp_eval:
            coco_eval = COCOevalMP(self._coco_api, coco_dt, metric)
        else:
            coco_eval = COCOeval(self._coco_api, coco_dt, metric)

        coco_eval.params.catIds = self.cat_ids
        coco_eval.params.imgIds = self.img_ids
        coco_eval.params.maxDets = list(self.proposal_nums)
        coco_eval.params.iouThrs = self.iou_thrs

        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        precisions = coco_eval.eval['precision']
        if precisions is None:
            print(f"CocoMetric90 parent results : eval['precision'] is None" )
            res_metrics['bbox_mAP_90'] = 0.0
            return res_metrics

        if 0.90 not in self.iou_thrs:
            # Should not happen, as COCO eval uses 0.50:0.05:0.95, but safe fallback
            print(f"CocoMetric90 parent results : iou_thrs {self.iou_thrs}" )
            res_metrics['bbox_mAP_90'] = 0.0
            return res_metrics

        idx_90 = np.where(self.iou_thrs == 0.90)[0][0]
        ap_90 = float(np.mean(precisions[idx_90]))
        res_metrics['bbox_mAP_90'] = ap_90

        return res_metrics