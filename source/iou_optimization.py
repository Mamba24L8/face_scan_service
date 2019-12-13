# -*- coding: utf-8 -*-
"""
Created on 12/5/19 10:53 AM

@author: mamba

@purpose：
"""
import numpy as np


def bbox_iou(bbox_a, bbox_b, offset=0):
    """Calculate Intersection-Over-Union(IOU) of two bounding boxes.

    Parameters
    ----------
    bbox_a : numpy.ndarray
        An ndarray with shape :math:`(N, 4)`.
    bbox_b : numpy.ndarray
        An ndarray with shape :math:`(M, 4)`.
    offset : float or int, default is 0
        The ``offset`` is used to control the whether the width(or height) is computed as
        (right - left + ``offset``).
        Note that the offset must be 0 for normalized bboxes, whose ranges are in ``[0, 1]``.

    Returns
    -------
    numpy.ndarray
        An ndarray with shape :math:`(N, M)` indicates IOU between each pairs of
        bounding boxes in `bbox_a` and `bbox_b`.

    Examples
    --------
    >>> import numpy as np
    >>> bbox_a = np.array([[0, 0, 0.8, 0.8]])
    >>> bbox_b = np.array([[0.4, 0.4, 1, 1]])
    >>> iou = bbox_iou(bbox_a, bbox_b)
    """
    if isinstance(bbox_a, list) and isinstance(bbox_b, list):
        bbox_a, bbox_b = np.array([bbox_a]), np.array([bbox_b])

    if bbox_a.shape[1] < 4 or bbox_b.shape[1] < 4:
        raise IndexError("Bounding boxes axis 1 must have at least length 4")

    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    br = np.minimum(bbox_a[:, None, 2:4], bbox_b[:, 2:4])

    area_i = np.prod(br - tl + offset, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:4] - bbox_a[:, :2] + offset, axis=1)
    area_b = np.prod(bbox_b[:, 2:4] - bbox_b[:, :2] + offset, axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


class IouProcess:
    pass


if __name__ == '__main__':
    a = np.array([[1, 1, 3, 3]])
    b = np.array([[2, 2, 4, 4]])
    print(bbox_iou(a, b))
