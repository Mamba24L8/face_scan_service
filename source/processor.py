# -*- coding: utf-8 -*-
"""
Created on 12/5/19 10:53 AM

@author: mamba

@purpose：
"""
import os
import time
import pandas as pd

from random import sample
from pathlib import Path
from typing import Dict, List
from loguru import logger
from source.compare_face import get_df
from source.grpc_client import GetFaceFeature
from source.utils import json_dump, get_gcd
from source.target_person import SackedOfficial, SpecialPerson, ViolentSearch

try:
    from more_itertools import chunked
except ImportError:
    logger.debug("The package that is more_itertools is not found")
    from source.utils import chunked


class SavePath:
    """创建识别出的目标人物的图片和信息路径

    Examples
    --------
    >>> msg = {'zip_file_path': 'http://10.242.189.120:8080/frames_zip/yxsz/201909/10/122139/640100_122139_20190910100000_20190910110000.ts.tar',
              'frames_dir': '/data/frames/201806/25/1126/20180625-181559_20180625-201605.mp4',
              'audio_path': None,
              'id': '47032',
              'date': '2019-09-10 10:00:00',
              'end_date': '2019-09-10 11:00:00',
              'chan_num': '122139',
              'chan_name': 'None',
              'video_path': '/media/yzfbscc/yxsz/yxsz/201909/10/122139/640100_122139_20190910100000_20190910110000.ts',
              'data_source': 'youxian',
              'video_location': '/media/yzfbscc/yxsz/yxsz/201909/10/122139/640100_122139_20190910100000_20190910110000.ts',
              'start_time': '2019-09-10 10:00:00',
              'video_url': 'http://10.242.189.224:8080/data/yxsz/yxsz/201909/10/122139/640100_122139_20190910100000_20190910110000.ts',
              'video_copy_path': '/data/videos/yxsz/yxsz/201909/10/122139/640100_122139_20190910100000_20190910110000.ts',
              'fps': 25, 'resolution': [352, 288], 'task_id': 'a10aea6e-60c5-43d7-8c8f-cb697085ed4d', 'is_search': 1, 'grade': 3}
    >>> tool = SavePath(msg=msg)
    >>> tool.suspicion_dir
    '/data/suspicion_face/201806/25/1126/20180625-181559_20180625-201605.mp4'
    >>> tool.txt_dir
    '/data/txt/201806/25/1126/20180625-181559_20180625-201605.mp4/log.txt'
    """

    def __init__(self, msg: Dict, key: str = "frames_dir"):
        self.path = msg.get(key)
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

    def __init__(self, msg: dict, save_path: SavePath):
        if isinstance(msg, dict) and isinstance(save_path, SavePath):
            dct = {"txt_path": save_path.txt_dir,
                   "suspicion_dir": save_path.suspicion_dir}
            self.msg = msg.update(dct)
        else:
            raise ValueError("任务消息不正确")


# def log_face(df, suspicion_face_dir):
#     idx_unique = df["idx"].unique()
#     lst = []
#     for i in idx_unique:
#         tmp_df = df[df["idx"] == i]
#         tmp_df.reset_index(drop=True, inplace=True)
#
#         frame_log = {"res": [], "id": tmp_df["frame_id"][0],
#                      "time": tmp_df["time_dot"][0]}
#         for _, row in tmp_df.iterrows():
#             frame_log["res"].append({
#                 "personId": row["wid"], "personName": row["who"],
#                 "sim": row["sim"], "bb": row["bbox"][0].tolist(),
#                 "lm": row["landmark"].tolist(),
#                 "frameUrl": os.path.join(suspicion_face_dir, row["who"],
#                                          Path(row["frame_path"]).name),
#                 "tag": ""
#             })
#         lst.append(frame_log)
#     return lst


class FaceProcess:

    def __init__(self, msg, address, frame_rate, pattern="*.jpg"):
        """

        Parameters
        ----------
        msg : dict
        address : str, facecpp's ip
        frame_rate : int or float, 一秒分帧数
        """
        self.msg = msg
        self.address = address
        self.frame_rate = frame_rate
        self.path = self.sort_filter_filename(pattern)

    def get_time_dot(self, filename):
        """获得时间戳"""

        return int(Path(filename).stem) / self.frame_rate

    def sort_filter_filename(self, pattern="*.jpg") -> List:
        """对图片名字按照数字大小进行排序，并进行滤掉(按照电视台的等级，多少帧取一张图片).
        """
        frame_dir = self.msg["frame_dir"]
        interval = self.msg["interval"]
        frame_paths = list(Path(frame_dir).glob(pattern))
        frame_paths.sort(key=lambda x: int(x.stem))

        # 对于interval是整数时
        if interval == 1:
            return list(map(os.fspath, frame_paths))
        if isinstance(interval, int):
            frame_paths = list(filter(lambda x: int(x.stem) % interval == 0,
                                      frame_paths))
            return list(map(os.fspath, frame_paths))

        # 对于interval是分数时
        frame_paths_chosen = []
        num_image = int(self.frame_rate * interval)
        for frame_path in chunked(frame_paths, num_image):
            gcd = get_gcd(self.frame_rate, num_image)  # 求公约数
            rate, num = int(self.frame_rate / gcd), int(num_image / gcd)

            for frames in chunked(frame_path, num):
                size = len(frames)

                if size >= rate:
                    frame_paths_chosen.extend(sample(frames, rate))
                else:
                    frame_paths_chosen.extend(sample(frames, size))

        frame_paths_chosen.sort(key=lambda x: int(x.stem))
        return list(map(os.fspath, frame_paths_chosen))

    def runner(self, save_path: SavePath,
               sacked_official: SackedOfficial,
               special_person=None,
               violent_search=None,
               **kwargs):
        """ 人脸识别、结果数据存储

        Parameters
        ----------
        save_path : instances of Tool
        sacked_official : instances of SackedOfficials, 落马官员
        special_person : instances of SpecialPerson,  AI搜索
        violent_search : instances of ViolentSearch, 暴力检索

        """
        tic = time.time()
        num_face = 0  # 人脸数量
        num_img = 0  # 图片数量

        logger.info("开始处理任务 {}".format(self.msg['video_path']))
        with GetFaceFeature(self.address, max_workers=None) as gff:
            df_list, violent_search_list, save2es_list = [], [], []
            for face_list, images, files in gff.images_feature(self.path):
                df = get_df(face_list)
                num_img += len(files)
                num_face += len(df)

                if df.empty:  # pandas不能直接用 if df
                    continue

                df["frame_path"] = [files[i] for i in df["idx"]]
                df["frame_id"] = df["frame_path"].apply(
                    lambda x: int(Path(x).stem))
                df["time"] = df["frame_id"] / self.frame_rate

                df_list.append(sacked_official.runner(df, images, save_path))

                if isinstance(special_person, SpecialPerson):
                    save2es_list += special_person.runner(self.msg, df)
                if isinstance(violent_search, ViolentSearch):
                    violent_search_list += violent_search.runner(self.msg, df)

                for instance in kwargs:
                    if hasattr(instance, "runner"):
                        instance.runner(self.msg, df)
                    else:
                        raise ValueError("该对象没有‘runner’方法")

        if save2es_list:  # todo 本地备份
            special_person.es.bulk(save2es_list)
            es_backup_file = save_path.txt_dir.replace("log", "es")
            json_dump(es_backup_file, save2es_list)

        if violent_search_list:  # 暴力检索本地备份
            violent_search_file = save_path.txt_dir.replace("log", "vio")
            json_dump(violent_search_file, violent_search_list)

        if df_list:
            df_list = pd.concat(df_list)
            df_list.to_json(orient="records")

        toc = time.time()
        logger.success(
            f"人脸识别完成{self.msg}\t识别人脸{len(df_list)}个\t用时{toc - tic}秒")
        return df_list
