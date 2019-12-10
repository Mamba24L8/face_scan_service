# -*- coding: utf-8 -*-
"""
Created on 12/5/19 10:53 AM

@author: mamba

@purpose：
"""
import os
import time
import json
import numpy as np

from pathlib import Path
from datetime import datetime

from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple
from loguru import logger
from source.compare_face import format_data, CompareFace, get_df
from source.grpc_client import GetFaceFeature


class Tool:
    """创建识别出的目标人物的图片和信息"""

    def __init__(self, message: Dict, key: str = "frames_dir"):
        self.path = message.get(key)
        self.txt_dir = self.make_log_dir()
        self.suspicion_dir = self.make_suspicion_dir()

    def make_log_dir(self) -> str:
        """ 用于保存视频识别到目标人物的备份，如果需要暴力检索和AI检索，也会备份到该文件夹
        该文件夹下有log.txt, 有可能有es.txt, vio.txt

        Returns
        -------
        txt_path : str, 识别到的目标人物
        """
        txt_dir = Path(self.path.replace("frames", "txt"))
        if not txt_dir.exists():
            txt_dir.mkdir()
        return os.fspath(Path(txt_dir, "log.txt"))

    def make_suspicion_dir(self) -> str:
        """将识别到的目标人物保存到该文件夹下, 假如识别到一个人，那么会以该人物的名字创建
        一个新的文件夹（该文件夹不存在），这个文件夹下保存保存该人物的图片.

        Returns
        -------
        suspicion_dir ： str, 将识别到的目标人物保存到该文件夹下
        """
        suspicion_dir = Path(self.path.replace("frames", "suspicion_face"))
        if not suspicion_dir.exists():
            suspicion_dir.mkdir()
        return os.fspath(suspicion_dir)


class IouTask:
    """获取Iou任务, 用于去重

    IoU： 交并比, intersection over union. 通常称为Jaccard系数, 描述两个边框的相似度.
    """

    def __init__(self, message: dict, tool: Tool):
        if isinstance(message, dict) and isinstance(tool, Tool):
            dct = {"txt_path": tool.txt_dir,
                   "suspicion_dir": tool.suspicion_dir}
            self.message = message.update(dct)
        else:
            raise ValueError("任务消息不正确")


class TargetPerson(metaclass=ABCMeta):
    """人物对象的处理基类"""

    @abstractmethod
    def person_info_loader(self, *args, **kwargs):
        """从数据库中加载人物信息并进行格式化

        See Also
        --------
            format_data
        """
        pass

    @abstractmethod
    def process(self, *args, **kwargs):
        """针对不同人物进行处理过程的主程序"""
        pass

    @abstractmethod
    def runner(self, *args, **kwargs):
        pass


class SackedOfficials(TargetPerson):
    _field = 'use_face'
    _status = 1

    def __init__(self, db, sim):
        self.db = db
        self.sim = sim
        self.target_person_info = self.person_info_loader()

    def person_info_loader(self) -> Tuple[List, np.array, List]:
        sacked_officials_list = self.db.query_person_list('use_face', 1)
        return format_data(sacked_officials_list)

    def process(self    , df):
        # df = get_df(face_infos_list)
        return CompareFace(self.target_person_info, self.sim).compare_face(df)

    def runner(self, single_picture_feature: List[np.ndarray], message: Dict):
        pass


class SpecialPerson:
    _field = 'use_ai'
    _status = 1

    def __init__(self, db, sim, es):
        self.db = db
        self.sim = sim
        self.es = es
        self.target_person_info = self.person_info_loader()

    def person_info_loader(self) -> Tuple[List, np.array, List]:
        special_person_list = self.db.query_person_list('use_ai', 1)
        return format_data(special_person_list)

    def process(self, df):
        # df = get_df(face_infos_list)
        return CompareFace(self.target_person_info, self.sim).compare_face(df)

    def runner(self, single_picture_feature: List[np.ndarray], message: Dict):
        es_saver = []
        vio_saver = []

        for feature, bbox, landmark in single_picture_feature:
            msg = {
                **message,
                "feature": json.dumps(feature.tolist()),
                "bbox": json.dumps(bbox.tolist()),
                "landmark": json.dumps(landmark.tolist())
            }
            vio_saver.append(msg)

        boxes = self.process(single_picture_feature)

        if boxes:
            video_url = '/'.join(message['video_path'].split('/')[-5:])

            for box in boxes:
                if box.get("wid", None):
                    face_info = {
                        "frame_url": message.get("frame_path"),
                        "video_name": os.path.basename(video_url),
                        "video_url": video_url,
                        "channel_id": message.get("chan_num"),
                        "channel_name": message.get("chan_name"),
                        "create_time": str(datetime.now()).split('.')[0],
                        "time": message.get("time"),
                        "date": message.get("date"),
                        "face_name": box.get("who"),
                        "source": message.get("data_source")}
                    es_saver.append(face_info)
        return es_saver, vio_saver


class FaceProcess:

    def __init__(self, message, address, frame_rate):
        self.message = message
        self.address = address
        self.frame_rate = frame_rate
        self.path = self.sort_filter_filename

    def get_time_dot(self, filename):
        return int(Path(filename).stem) / self.frame_rate

    def sort_filter_filename(self):
        frame_dir = self.message["frame_dir"]
        interval = self.message["interval"]
        frame_path_list = list(Path(frame_dir).glob("*.jpg"))
        frame_path_list = list(filter(lambda x: int(x.stem) % interval != 0,
                                      frame_path_list))
        frame_path_list.sort(key=lambda x: int(x.stem))
        return list(map(os.fspath, frame_path_list))

    def runner(self):
        logger.info("开始处理任务 {}".format(self.message['video_path']))
        tic = time.time()
        with GetFaceFeature(self.address) as gff:
            for face_infos_list, images, filenames in gff.images_feature(
                    self.path):
                df = get_df(face_infos_list)
                df["frame_path"] = [filenames[i] for i in df["idx"]]
                df["time"] = df["frame_path"].apply(
                    lambda x: int(Path(x).stem) / self.frame_rate)

            if str(self.message["is_search"]) == 1:
                pass

        toc = time.time()
