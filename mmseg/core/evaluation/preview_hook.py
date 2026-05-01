import os.path as osp

import mmcv
import numpy as np
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import Hook

from mmseg.datasets.pipelines import Compose


class _LoadImage(object):

    def __call__(self, results):
        if isinstance(results['img'], str):
            results['filename'] = results['img']
            results['ori_filename'] = results['img']
        else:
            results['filename'] = None
            results['ori_filename'] = None
        img = mmcv.imread(results['img'])
        results['img'] = img
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        return results


class PreviewHook(Hook):
    """Save fixed-sample prediction previews during training."""

    def __init__(self,
                 interval=500,
                 out_dir='previews',
                 data_root=None,
                 positive_split=None,
                 negative_split=None,
                 max_positive=1,
                 max_negative=1):
        self.interval = interval
        self.out_dir = out_dir
        self.data_root = data_root
        self.positive_split = positive_split
        self.negative_split = negative_split
        self.max_positive = max_positive
        self.max_negative = max_negative
        self.samples = None

    def after_train_iter(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        if self.samples is None:
            self.samples = self._load_samples()
        if not self.samples:
            runner.logger.warning('PreviewHook found no preview samples.')
            return

        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        device = next(model.parameters()).device
        was_training = model.training
        model.eval()

        out_dir = osp.join(runner.work_dir, self.out_dir, f'iter_{runner.iter + 1:06d}')
        mmcv.mkdir_or_exist(out_dir)

        try:
            for sample in self.samples:
                img_path = osp.join(self.data_root, sample + '.jpg')
                gt_path = osp.join(self.data_root, sample + '_label.bmp')
                result = self._inference(model, img_path, device)
                pred = result[0]
                img = mmcv.imread(img_path)
                pred_overlay = self._overlay(model, img, pred)
                gt_mask = (mmcv.imread(gt_path, flag='unchanged', backend='pillow') > 0).astype(np.uint8)
                gt_overlay = self._overlay(model, img, gt_mask)

                sample_name = sample.replace('/', '__')
                mmcv.imwrite(img, osp.join(out_dir, f'{sample_name}_image.jpg'))
                mmcv.imwrite(pred_overlay, osp.join(out_dir, f'{sample_name}_pred.jpg'))
                mmcv.imwrite(gt_overlay, osp.join(out_dir, f'{sample_name}_gt.jpg'))
                mmcv.imwrite((pred * 255).astype(np.uint8), osp.join(out_dir, f'{sample_name}_pred_mask.png'))
                mmcv.imwrite((gt_mask * 255).astype(np.uint8), osp.join(out_dir, f'{sample_name}_gt_mask.png'))
        finally:
            if was_training:
                model.train()

    def _load_samples(self):
        samples = []
        if self.positive_split and osp.exists(self.positive_split):
            samples.extend(self._read_split(self.positive_split)[:self.max_positive])
        if self.negative_split and osp.exists(self.negative_split):
            samples.extend(self._read_split(self.negative_split)[:self.max_negative])
        return samples

    @staticmethod
    def _read_split(path):
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    @staticmethod
    def _overlay(model, img, seg):
        if img.shape[:2] != seg.shape[:2]:
            img = mmcv.imresize(img, (seg.shape[1], seg.shape[0]))
        palette = np.array(model.PALETTE, dtype=np.uint8)
        color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
        for label, color in enumerate(palette):
            color_seg[seg == label, :] = color
        color_seg = color_seg[..., ::-1]
        return (img * 0.5 + color_seg * 0.5).astype(np.uint8)

    @staticmethod
    def _unwrap_meta(data):
        return [item.data[0] for item in data['img_metas']]

    def _inference(self, model, img_path, device):
        cfg = model.cfg
        test_pipeline = [_LoadImage()] + cfg.data.test.pipeline[1:]
        test_pipeline = Compose(test_pipeline)
        data = test_pipeline(dict(img=img_path))
        data = collate([data], samples_per_gpu=1)

        if next(model.parameters()).is_cuda:
            data = scatter(data, [device])[0]
        else:
            data['img_metas'] = self._unwrap_meta(data)

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
        return result
