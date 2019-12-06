# -*- coding: utf-8 -*-
"""
Created on 12/5/19 10:53 AM

@author: mamba

@purpose：  1. 从redis得到任务
            2. 下载压缩包
            3. 解压
            4. 整理信息（判断是否是AI搜索的台， 判断每秒处理多少帧）
            5. 为落马官员识别推送任务（存放到redis）
            6.
"""
import os
import time
import json
import requests
import tarfile

from retry import retry
from copy import deepcopy
from typing import Dict, Tuple
from pathlib import Path
from configparser import ConfigParser
from abc import ABCMeta, abstractmethod

from loguru import logger

config = ConfigParser()
config.read('/face_scan/conf/unzip.cfg')


class OrganizeInfo:

    def __init__(self, data: Dict):
        self.data = data
        self.url = self.data["zip_file_path"]
        self._name, self.filename = self.get_filename()
        self.extract_dir = self.get_extract_dir()

    def get_filename(self) -> Tuple[str, str]:
        _name = os.fspath(Path(self.data["zip_file_path"]).name)
        filename = os.fspath(Path("/data/zip", self.data["task_id"], _name))
        return _name, filename

    def get_extract_dir(self) -> str:
        ymd = "/".join(self.data["video_path"].split('/')[-4:-1])
        extract_dir = Path("/data", "frames", ymd)
        if not extract_dir.exists():
            extract_dir.mkdir()
        return os.fspath(extract_dir)

    def get_download_input(self) -> Tuple[str, str]:
        """获得下载输入"""
        return self.url, self.filename

    def get_unpack_input(self) -> Tuple[str, str]:
        """解包输入"""
        return self.filename, self.extract_dir

    def push_info(self, dct) -> Dict:
        info = deepcopy(self.data)
        info["frames_dir"] = os.fspath(Path(self.extract_dir, self._name))
        if not isinstance(dct, dict):
            dct = {}
        info["is_search"] = dct.get("search_tv", 0)
        info["grade"] = dct.get("grade", 1)
        return info


class Command(metaclass=ABCMeta):
    """命令的抽象类，两个命令： 1. 解压， 2. 下载"""

    @abstractmethod
    def execute(self):
        pass


class DownloadPack(Command):
    """下载压缩包"""
    CHUNK_SIZE = 1024 * 512  # 每次写文件的缓冲大小

    def __init__(self, url, filename):
        self.url = url
        self.filename = filename

    def execute(self):
        filename = self._download(self.url, self.filename)
        return filename

    @retry(tries=8)
    def _download(self, url, filename):
        try:
            req = requests.get(url=url, stream=True, timeout=(20, 200))
            total_size = int(req.headers.get("Content-Length"))
            with open(filename, "wb") as file:
                if total_size is None:
                    for chunk in req.iter_content(chunk_size=self.CHUNK_SIZE):
                        if chunk:
                            file.write(chunk)
            logger.success(f"[下载{url}成功] 文件大小: {total_size} 保存路径: {filename}")
            return filename
        except requests.exceptions.ReadTimeout:
            logger.info(f"[下载文件超时] 文件{url}")
            raise ValueError(f"下载失败, 文件{url}")
        except requests.exceptions.ConnectTimeout:
            logger.info(f"[网络连接超时] 文件{url}")
            raise ValueError(f"下载失败, 文件{url}")

    def undo(self):
        if Path(self.filename).exists():
            os.remove(self.filename)


class UnPackTarFile(Command):
    """解包"""

    def __init__(self, filename, extract_dir):
        self.filename = filename
        self.extract_dir = extract_dir

    def execute(self):
        filename = self._unpack_tarfile(self.filename, self.extract_dir)
        return filename

    @staticmethod
    def _unpack_tarfile(filename, extract_dir):
        try:
            with tarfile.open(filename, "r|") as z:
                z.extractall(path=extract_dir)
            logger.success(f"[解包{filename}] 成功")
            return extract_dir + "/" + filename.split("/")[-1][:-4]
        except Exception as e:
            logger.info(f"[解包{filename}] 报错 {e}")
            raise Exception(e)

    def undo(self):
        path = Path(self.extract_dir, Path(self.filename).stem)
        if path.exists():
            os.removedirs(path)


def push_info2redis(message, redis_client, key, queue_size):
    """解压后的任务存放到redis里

    Parameters
    ----------
    message : Dict, push message to redis
    redis_client : instance of redis client
    key : str, queue name
    queue_size : int, queue size
    """
    while True:
        task_len = redis_client.llen(key)
        if task_len < int(queue_size):
            redis_client.lpush(key, json.dumps(message))
            logger.info('向队列中添加任务成功{}'.format(message))
            break
        else:
            logger.info('队列已满,等待15秒')
            time.sleep(15)
