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
import pandas as pd

from pathlib import Path
from datetime import datetime

from abc import ABCMeta, abstractmethod
from typing import Dict, List, Tuple
from loguru import logger
from source.compare_face import format_data, CompareFace, get_df
from source.grpc_client import GetFaceFeature
from source.utils import save_image, json_dump


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
    """获取Iou任务, 用于去重 [弃用] 直接在faceworker中去重

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
        """针对不同人物进行处理过程（人脸对比过程）"""
        pass

    @abstractmethod
    def runner(self, *args, **kwargs):
        """主程序"""
        pass


def log_face(df, suspicion_face_dir):
    idx_unique = df["idx"].unique()
    lst = []
    for i in idx_unique:
        tmp_df = df[df["idx"] == i]
        tmp_df.reset_index(drop=True, inplace=True)

        frame_log = {"res": [], "id": tmp_df["frame_id"][0],
                     "time": tmp_df["time_dot"][0]}
        for _, row in tmp_df.iterrows():
            frame_log["res"].append({
                "personId": row["wid"], "personName": row["who"],
                "sim": row["sim"], "bb": row["bbox"][0].tolist(),
                "lm": row["landmark"].tolist(),
                "frameUrl": os.path.join(suspicion_face_dir, row["who"],
                                         Path(row["frame_path"]).name),
                "tag": ""
            })
        lst.append(frame_log)
    return lst


class SackedOfficials(TargetPerson):
    """落马官员"""
    _field = 'use_face'
    _status = 1

    def __init__(self, db, sim):
        self.db = db
        self.sim = sim
        self.target_person_info = self.person_info_loader()

    def person_info_loader(self) -> Tuple[List, np.array, List]:
        sacked_officials_list = self.db.query_person_list('use_face', 1)
        return format_data(sacked_officials_list)

    def process(self, df: pd.DataFrame):
        # df = get_df(face_infos_list)
        return CompareFace(self.target_person_info, self.sim).compare_face(df)

    def runner(self, message: dict, df: pd.DataFrame, images: np.ndarray,
               tool: Tool):
        """ 画框、标注名字、保存"""
        df = self.process(df)
        for index, row in df.iterrows():
            idx, bbox, name = row["idx"], row["bbox"], row["who"]
            person_folder = Path(tool.suspicion_dir, name)
            if not person_folder.exists():
                person_folder.mkdir()
            save_path = os.fspath(
                Path(person_folder, Path(row["frame_path"]).name))
            save_image(images[:, :, :, idx], bbox, name, save_path)
        return df


class SpecialPerson:
    """AI搜索"""
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

    def process(self, df: pd.DataFrame):
        # df = get_df(face_infos_list)
        return CompareFace(self.target_person_info, self.sim).compare_face(df)

    def runner(self, message: Dict, df: pd.DataFrame):
        elastic_search = []
        df = self.process(df)

        if df:
            video_url = '/'.join(message['video_path'].split('/')[-5:])

            for _, row in df.iterrows():
                if row["wid"] is None:
                    continue
                info = {
                    "frame_url": message.get("frame_path"),
                    "video_name": os.path.basename(video_url),
                    "video_url": video_url,
                    "channel_id": message.get("chan_num"),
                    "channel_name": message.get("chan_name"),
                    "create_time": str(datetime.now()).split('.')[0],
                    "time": message.get("time"),
                    "date": message.get("date"),
                    "face_name": row["who"],
                    "source": message.get("data_source")
                }
                elastic_search.append(info)
        return elastic_search


class ViolentSearch:
    """暴力搜索"""

    @staticmethod
    def runner(message: dict, df: pd.DataFrame):
        violent_search = []
        for _, row in df.iterrows():
            info = {
                **message,
                "feature": json.dumps(row["feature"].tolist()),
                "bbox": json.dumps(row["bbox"][0].tolist()),
                "landmark": json.dumps(row["landmark"].tolist())
            }
            violent_search.append(info)
        return violent_search


class FaceProcess:

    def __init__(self, message, address, frame_rate):
        self.message = message
        self.address = address
        self.frame_rate = frame_rate
        self.path = self.sort_filter_filename()

    def get_time_dot(self, filename):
        """获得时间戳"""
        return int(Path(filename).stem) / self.frame_rate

    def sort_filter_filename(self):
        """对图片名字按照数字大小进行排序，并进行滤掉(按照电视台的等级，多少帧取一张图片)."""
        frame_dir = self.message["frame_dir"]
        interval = self.message["interval"]
        frame_path_list = list(Path(frame_dir).glob("*.jpg"))
        frame_path_list = list(filter(lambda x: int(x.stem) % interval != 0,
                                      frame_path_list))
        frame_path_list.sort(key=lambda x: int(x.stem))
        return list(map(os.fspath, frame_path_list))

    def runner(self, sacked_officials, special_person=None,
               violent_search=None, tool=None):
        logger.info("开始处理任务 {}".format(self.message['video_path']))
        tic = time.time()
        with GetFaceFeature(self.address) as gff:
            df_list, violent_search_list, es_list = [], [], []
            for face_infos_list, images, filenames in gff.images_feature(
                    self.path):
                df = get_df(face_infos_list)

                df["frame_path"] = [filenames[i] for i in df["idx"]]
                df["frame_id"] = df["frame_path"].apply(
                    lambda x: int(Path(x).stem))
                df["time"] = df["frame_id"] / self.frame_rate

                df_list.append(
                    sacked_officials.runner(self.message, df, images, tool))

                if isinstance(special_person, SpecialPerson):
                    es_list += special_person.runner(self.message, df)
                if isinstance(violent_search, ViolentSearch):
                    violent_search_list += violent_search.runner(self.message,
                                                                 df)

        if es_list:
            special_person.es.bulk(es_list)
            es_backup_file = tool.txt_dir.replace("log", "es")
            json_dump(es_backup_file, es_list)
        if violent_search_list:
            violent_search_backup_file = tool.txt_dir.replace("log", "vio")
            json_dump(violent_search_backup_file, violent_search_list)
        if df_list:
            df_list = pd.concat(df_list)
            df_list.to_json(orient="records")
        toc = time.time()
        logger.success(
            f"人脸识别完成{self.message}\t识别人脸{len(df_list)}个\t用时{toc - tic}秒")
