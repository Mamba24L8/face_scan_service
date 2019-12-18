# -*- coding: utf-8 -*-
"""
Created on 12/5/19 10:53 AM

@author: mamba

@purpose：
"""
import os
import datetime
import numpy as np
import pandas as pd

from pathlib import Path


def calculate_current_time(date: str, time_dot) -> datetime.datetime:
    """ 用于计算图片出现的时间点

    Parameters
    ----------
    date : str, 日期字符串， fmt="%Y-%m-%d %H:%M:%S", "2019-09-10 10:00:00"
    time_dot : Optional[int, float]

    Returns
    -------
    current_time : datetime.datetime, 图片出现的时间点
    """
    if isinstance(time_dot, str):
        time_dot = float(time_dot)
    date = datetime.datetime.strptime(date, "%Y-%m-%d %H:%M:%S")
    time_dot = datetime.timedelta(seconds=time_dot)
    return date + time_dot


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

    def __init__(self, threshold, fps):
        """人脸去重

        Parameters
        ----------
        threshold : float
        fps : float or int
        """
        self.threshold = threshold
        self.interval = fps * 1800

    def process(self, df):
        n_row = len(df)
        if n_row == 0:
            return pd.DataFrame()

        df["tag"] = [[] for _ in range(n_row)]
        if n_row == 1:
            return df

        bbox_a, id_a = df.loc[0, "bbox"], df.loc[0, "id"]
        new_df = pd.DataFrame()
        new_df = new_df.append(df.loc[0])
        for index, row in df.iterrows():
            bbox_b, id_b = row["bbox"], row["id"]
            if (bbox_iou(bbox_a, bbox_b) < self.threshold) or (
                    id_b - id_a > self.interval):
                new_df = new_df.append(df.loc[index], ignore_index=True)
                bbox_a, id_a = bbox_b, id_b
            else:
                new_df.loc[len(new_df), "tag"].append(id_b)
        return new_df

    def runner(self, df, message, web_db, tool):
        if not df:
            return
        names = set(df["who"])
        result = []
        for name in names:
            tmp_df = df[df["who"] == name]
            result.append(self.process(tmp_df))
        result = pd.concat(result)
        for index, row in result.iterrows():
            dct = {
                "date": calculate_current_time(message["date"], row["time"]),
                "video_path": message["video_path"],
                "data_source": message["data_source"],
                "chan_num": message["chan_num"],
                "chan_name": message["chan_name"],
                "frame_path": os.path.join(tool.suspicion_face_dir, row["who"],
                                           Path(row["frame_path"]).name),
                "person_id": row["wid"],
                "people_name": row["who"],
                "sim": row["sim"],
                "time_dot": row["time"]
            }
            web_db.insert2face_details(dct)


if __name__ == '__main__':
    a = np.array([[1, 1, 3, 3]])
    b = np.array([[2, 2, 4, 4]])
    print(bbox_iou(a, b))
