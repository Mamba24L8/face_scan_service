# -*- coding: utf-8 -*-
"""
Created on 2020/1/8 14:06

@author: mamba

@purpose：习大大出现的视频
"""
import os
import requests
import tarfile
tarfile.open

from numbers import Number
from retry import retry
from loguru import logger
from source.utils import json_load
from iou_optimization import bbox_iou


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
            # time_out: 请求用时20秒，下载用时200秒
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
        self.message = json_load(filename)

    def clip_video(self, input_video):
        """ 截取视频"""
        self.message.sort(key=lambda hash_map: hash_map["date"])
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

    def get_video_clip(self):
        """获得视频时间段"""
        pass
