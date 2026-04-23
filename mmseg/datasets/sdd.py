import os
import os.path as osp

import mmcv
import numpy as np
from mmcv.utils import print_log

from mmseg.utils import get_root_logger
from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class SDDDataset(CustomDataset):
    """SDD binary segmentation dataset.

    The source masks use 0 for background and 255 for positive/defect pixels.
    This dataset maps every non-zero mask value to class 1 for both training
    and evaluation.
    """

    CLASSES = ('background', 'positive')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, eval_shape=None, **kwargs):
        self.eval_shape = eval_shape
        super(SDDDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='_label.bmp', **kwargs)

    def get_gt_seg_maps(self, efficient_test=False):
        if efficient_test:
            raise NotImplementedError(
                'SDDDataset does not support efficient_test because masks '
                'must be binarized before metric calculation.')

        gt_seg_maps = []
        for img_info in self.img_infos:
            seg_map = osp.join(self.ann_dir, img_info['ann']['seg_map'])
            gt_seg_map = mmcv.imread(
                seg_map, flag='unchanged', backend='pillow')
            gt_seg_map = (gt_seg_map > 0).astype(np.uint8)
            if self.eval_shape is not None:
                height, width = self.eval_shape
                gt_seg_map = mmcv.imresize(
                    gt_seg_map, (width, height), interpolation='nearest')
            gt_seg_maps.append(gt_seg_map)
        return gt_seg_maps

    def evaluate(self,
                 results,
                 metric='sdd',
                 logger=None,
                 efficient_test=False,
                 lambda_fp=0.10,
                 **kwargs):
        if efficient_test:
            raise NotImplementedError(
                'SDDDataset does not support efficient_test because masks '
                'must be binarized before metric calculation.')

        gt_seg_maps = self.get_gt_seg_maps(efficient_test=False)
        assert len(results) == len(gt_seg_maps)

        tp = fp = fn = 0
        neg_fp_images = 0
        neg_fp_pixels = 0
        neg_total_pixels = 0
        neg_images = 0

        for pred, gt in zip(results, gt_seg_maps):
            if isinstance(pred, str):
                pred = np.load(pred)
            pred = (pred > 0).astype(np.uint8)
            gt = (gt > 0).astype(np.uint8)

            pred_pos = pred == 1
            gt_pos = gt == 1

            tp += int(np.logical_and(pred_pos, gt_pos).sum())
            fp += int(np.logical_and(pred_pos, np.logical_not(gt_pos)).sum())
            fn += int(np.logical_and(np.logical_not(pred_pos), gt_pos).sum())

            if not gt_pos.any():
                neg_images += 1
                neg_fp_images += int(pred_pos.any())
                neg_fp_pixels += int(pred_pos.sum())
                neg_total_pixels += int(gt.size)

        pos_dice = 2.0 * tp / (2.0 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
        pos_iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0
        neg_fp_rate = neg_fp_images / neg_images if neg_images > 0 else 0.0
        neg_fp_pixel_rate = (
            neg_fp_pixels / neg_total_pixels if neg_total_pixels > 0 else 0.0)
        sdd_score = pos_dice - lambda_fp * neg_fp_rate

        eval_results = {
            'pos_dice': pos_dice,
            'pos_iou': pos_iou,
            'neg_fp_rate': neg_fp_rate,
            'neg_fp_pixel_rate': neg_fp_pixel_rate,
            'sdd_score': sdd_score,
        }

        print_log(
            'SDD metrics: '
            f'pos_dice={pos_dice:.6f}, '
            f'pos_iou={pos_iou:.6f}, '
            f'neg_fp_rate={neg_fp_rate:.6f}, '
            f'neg_fp_pixel_rate={neg_fp_pixel_rate:.6f}, '
            f'sdd_score={sdd_score:.6f} '
            f'(lambda_fp={lambda_fp}, negative_images={neg_images})',
            logger=logger or get_root_logger())

        if mmcv.is_list_of(results, str):
            for file_name in results:
                os.remove(file_name)
        return eval_results
