# -*- coding: utf-8 -*-
"""
Created on 12/5/19 10:48 AM

@author: mamba

@purpose：
"""
import time
import json

from loguru import logger
from easydict import EasyDict as edict
from config import DEFAULT, MYSQL, REDIS, DOCKER, ELASTICSEARCH


class FaceWorkerRunner:

    def __init__(self, redis_conn, faceworker_db, front_db, es_client=""):
        self.redis_conn = redis_conn
        self.faceworker_db = faceworker_db
        self.front_db = front_db
        self.es_client = es_client
        self.queue = REDIS["object"]["face_tasks"]

    def get_grade_and_search_tv(self, channel_number):
        """获取电视台等级

        Parameters
        ----------
        channel_number ： str

        Returns
        -------
        grade :
        is_search :
        """
        dct = self.front_db.query_grade(channel_number)
        if not isinstance(dct, dict):
            dct = {}
        return dct.get("grade", 1), dct.get("search_tv", 0)

    def runner(self):
        while True:
            message = self.redis_conn.rpop(self.queue)
            if not message:
                logger.info("暂时没有任务，等待10s")
                time.sleep(10)
                continue
            message = json.loads(message)
            logger.info(f"取到一条人脸识别任务 {message}")
            # 开始执行任务，更改状态为1
            self.faceworker_db.set_status(message["id"], "face_status", 1)
            message["grade"], message[
                "is_search"] = self.get_grade_and_search_tv(message["chan_num"])
            message = {
                "interval": DEFAULT["frame_rate_grade"][str(message["grade"])],
            }
            pass
