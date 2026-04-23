import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class BinaryMask(object):
    """Map every non-zero segmentation value to class 1."""

    def __call__(self, results):
        for key in results.get('seg_fields', []):
            results[key] = (results[key] > 0).astype(np.uint8)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class SetOriShapeToImgShape(object):
    """Keep test predictions at the resized shape instead of original shape."""

    def __call__(self, results):
        results['ori_shape'] = results['img_shape']
        return results

    def __repr__(self):
        return self.__class__.__name__
