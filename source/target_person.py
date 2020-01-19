# -*- coding: utf-8 -*-
"""
Created on 2020/1/8 14:06

@author: mamba

@purpose：习大大出现的视频
"""
import os
import json
import requests
import pandas as pd
import numpy as np

from retry import retry
from loguru import logger
from datetime import datetime
from pathlib import Path
from numbers import Number
from typing import Dict
from functools import lru_cache
from abc import ABCMeta, abstractmethod
from moviepy.editor import VideoFileClip
from source.utils import json_load
from source.utils import save_image
from source.processor import SavePath
from source.compare_face import format_data, CompareFace


class TargetPerson(metaclass=ABCMeta):
    """人物对象的处理基类"""

    @abstractmethod
    def get_info(self, *args, **kwargs):
        """从数据库中加载人物信息并进行格式化

        See Also
        --------
            format_data
        """
        pass

    @abstractmethod
    def compare(self, *args, **kwargs):
        """针对不同人物进行处理过程（人脸对比过程）"""
        pass

    @abstractmethod
    def runner(self, *args, **kwargs):
        """主程序"""
        pass


class KeyPerson(TargetPerson):
    """正面人物查找，需要保存图片"""

    def __init__(self, database, similarity):
        """

        Parameters
        ----------
        database : 数据库实例, 用于获取key person 以及保存结果
        similarity : float, 相似度
        """
        self.database = database
        self.similarity = similarity

    @lru_cache(maxsize=2048)
    def get_info(self):  # todo 是否需要新建一个表，将习近平加进去
        """获取4个特征"""
        pass

    def compare(self):  # 这里和落马官员对比不一样
        pass

    def runner(self):
        pass


class SackedOfficial(TargetPerson):
    """落马官员"""

    def __init__(self, database, similarity):
        """
        Parameters
        ----------
        database : instance os mysql, 用于查找特定人物
        similarity ： float, 相似度阈值
        """
        self.database = database
        self.similarity = similarity
        self.target_person_info = self.get_info()

    @lru_cache(maxsize=2048)
    def get_info(self, field="use_face", status=1):
        return format_data(self.database.query_person_list(field, status))

    def compare(self, df: pd.DataFrame):
        cmp = CompareFace(self.target_person_info, self.similarity)
        return cmp.compare_face(df)

    def runner(self, df: pd.DataFrame, images: np.ndarray,
               save_path: SavePath):
        """ 画框、标注名字、保存"""
        df = self.compare(df)
        for _, row in df.iterrows():
            idx, bbox, name = row["idx"], row["bbox"], row["who"]

            person_folder = Path(save_path.suspicion_dir, name)
            if not person_folder.exists():
                person_folder.mkdir()

            filename = os.fspath(
                Path(person_folder, Path(row["frame_path"]).name)
            )
            save_image(images[:, :, :, idx], bbox, name, filename)
        return df


class SpecialPerson(TargetPerson):
    """AI搜索"""

    def __init__(self, database, similarity, es):
        """

        Parameters
        ----------
        database : instance os mysql, 用于查找特定人物
        sim ： float, 相似度阈值
        es : instance of elasticsearch, 数据输出到es中
        """
        self.database = database
        self.similarity = similarity
        self.es = es
        self.target_person_info = self.get_info()

    @lru_cache(maxsize=2048)
    def get_info(self, field="use_ai", status=1):
        return format_data(self.database.query_person_list(field, status))

    def compare(self, df: pd.DataFrame):
        cmp = CompareFace(self.target_person_info, self.similarity)
        return cmp.compare_face(df)

    @staticmethod
    def get_hash_map(video_url, row, msg):
        hash_map = {
            "frame_url": msg.get("frame_path"),
            "video_name": os.path.basename(video_url),
            "video_url": video_url,
            "channel_id": msg.get("chan_num"),
            "channel_name": msg.get("chan_name"),
            "create_time": str(datetime.now()).split('.')[0],
            "time": msg.get("time"),
            "date": msg.get("date"),
            "face_name": row["who"],
            "source": msg.get("data_source")
        }
        return hash_map

    def runner(self, msg: Dict, df: pd.DataFrame):
        elastic_search = []
        df = self.compare(df)

        if df.empty:
            return elastic_search

        video_url = '/'.join(msg['video_path'].split('/')[-5:])
        for _, row in df.iterrows():
            if row["wid"] is not None:
                hash_map = self.get_hash_map(video_url, row, msg)
                elastic_search.append(hash_map)
        # todo 在这里将数据保存到es中
        return elastic_search


class ViolentSearch:
    """暴力搜索"""

    @staticmethod
    def runner(msg: dict, df: pd.DataFrame):
        violent_search = []
        for _, row in df.iterrows():
            info = {
                **msg,
                "feature": json.dumps(row["feature"].tolist()),
                "bbox": json.dumps(row["bbox"][0].tolist()),
                "landmark": json.dumps(row["landmark"].tolist())
            }
            violent_search.append(info)
        return violent_search


class Download:
    """ 下载视频，用于视频截片段"""

    def __init__(self, url, save_path, chunk_size=1024 * 512):
        """

        Parameters
        ----------
        url : str, 下载地址
        save_path : str, 保存地址
        chunk_size : 单次下载大小
        """
        self.url = url
        self.save_path = save_path
        self.chunk_size = chunk_size

    @retry(tries=4)
    def download(self):
        try:
            # time_out: 请求用时20秒，下载用时200秒, 否则超时
            req = requests.get(url=self.url, stream=True, timeout=(20, 200))
            with open(self.save_path, "wb") as file:
                for chunk in req.iter_content(chunk_size=self.chunk_size):
                    if chunk:
                        file.write(chunk)
            logger.success(f"[下载{self.url}成功]\n保存路径: {self.save_path}")
            return self.save_path
        except Exception as e:
            logger.error(f"[下载{self.url}失败]\n失败原因: {e}")


class GetVideoClip:
    """ 截取视频片段"""

    def __init__(self, filename):
        """

        Parameters
        ----------
        filename : List[dict], 保存每张图片的信息的路径
        """
        self.msg = json_load(filename)

    def clip_video(self, video_path):
        """ 截取视频列表"""
        self.msg.sort(key=lambda hash_map: hash_map["date"])
        pass

    def get_time_quantum(self, interval_time):
        """获得时间段

        Parameters
        ----------
        interval_time : Number

        Returns
        -------

        """

        pass

    @staticmethod
    def get_video_clip(video, time_pairs, save_path):
        """获得视频时间段

        Parameters
        ----------
        video : str or VideoFileClip, 视频路径或者视频已经被打开
        time_pairs : List[Tuple[start, stop]]
        save_path : str, 保存路径

        Returns
        -------

        """
        if isinstance(video, str):
            video = VideoFileClip(video)
        for index, time_pair in enumerate(time_pairs):
            sub_video = video.subclip(t_start=time_pair[0], t_end=time_pair[1])
            save_name = os.path.join(save_path, str(index), ".mp4")
            sub_video.write_videofile(save_name)
