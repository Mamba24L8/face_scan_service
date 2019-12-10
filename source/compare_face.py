# -*- coding: utf-8 -*-
"""
Created on 12/5/19 10:52 AM

@author: mamba

@purpose：
"""

import json
import numpy as np
import pandas as pd

from scipy.spatial import cKDTree
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional, Union, Generator, Set


def format_data(data: List[Dict]) -> Tuple[List, np.array, List]:
    """对数据库中读取的数据，进行格式化.数据库读出来的数据是[{name:_name, feature:_feature, id:_id}
    其中_feature是4个向量，格式化成，每个向量对应一个_name和_id.

    Parameters
    ----------
    data : List[Dict], such as [{}, {}], 每个人物的feature是4个向量.

    Returns
    -------
    format_data : Tuple, (name, feature, id),
                  ([张, 张, 张, 张],[feature[0],...,  feature[4]], [1,1,1,1])

    Notes
    -----
    数据库中, 一个目标人物人包含4个人脸特征, 4个人脸特征分别来自4张的图片.所有我们需要将
    feature分开作为单独的一个feature，但其name和id不变. 在和一张图片比对时, 这四个特征
    如果某个相似度大于阈值, 那就认为这个是图片上的人是目标人物.
    """

    names, features, ids = [], [], []
    for one_person_info in data:
        feature = json.loads(one_person_info["feature"])
        feature_len = len(feature)
        names.extend([one_person_info["person_name"]] * feature_len)
        features.extend(feature)
        ids.extend([one_person_info["id"]] * feature_len)
    features = np.array(features)
    len1, _, len3 = features.shape
    features = features.reshape(len1, len3)

    return names, features, ids


def get_df(face_infos_list):
    """

    Parameters
    ----------
    face_infos_list : list
        一张图片的grpc返回结果为list的一个元素

    Returns
    -------
    df : pd.DataFrame
    """
    lst = []
    for idx, feat_infos in enumerate(face_infos_list):
        if feat_infos:
            for feat_info in feat_infos:
                lst.append([idx, *feat_info])
    df = pd.DataFrame(lst, columns=["idx", "feature", "bb", "landmark"])
    return df


class CompareFace:

    def __init__(self, target_person_info, threshold):
        self.names, self.features, self.ids = target_person_info
        self.threshold = threshold

    def similarity(self, matrix) -> np.ndarray:
        """两个矩阵之间的余弦相似度

        Parameters
        ----------
        matrix : numpy.ndarray,
            with shape: (n, 512), n = 人脸个数 - 1

        Returns
        -------
            余弦相似度
        """
        return cosine_similarity(matrix, self.features)

    @staticmethod
    def get_df(face_infos_list):
        """

        Parameters
        ----------
        face_infos_list : list
            一张图片的grpc返回结果为list的一个元素

        Returns
        -------
        df : pd.DataFrame
        """
        lst = []
        for idx, feat_infos in enumerate(face_infos_list):
            if feat_infos:
                for feat_info in feat_infos:
                    lst.append([idx, *feat_info])
        df = pd.DataFrame(lst, columns=["idx", "feature", "bb", "landmark"])
        return df

    def compare_face(self, df: pd.DataFrame):
        """

        Parameters
        ----------
        df : pd.DataFrame,
             with columns=["idx", "feature", "bb", "landmark"]

        Returns
        -------
        df : pd.DataFrame
        """
        features = np.vstack(df["feature"].values)
        dist = cosine_similarity(features, self.features)
        coordinate = np.where(dist > self.threshold)
        # df["max_ind"], df["max_dist"] = dist.argmax(axis=1), dist.max(axis=1)
        df = df.loc[coordinate[0]]
        df["who"] = [self.names[x.item()] for x in coordinate[1]]
        df["wid"] = [self.ids[x.item()] for x in coordinate[1]]
        df["sim"] = [dist[x, y] for x, y in zip(*coordinate)]
        # 排重，一张图片一个人只会出现一次， 保留sim最大的
        df.sort_values(by=["sim"], inplace=True)
        df.drop_duplicates(subset=['idx', 'who'], keep='last', inplace=True)
        return df
