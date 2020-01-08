# -*- coding: utf-8 -*-
"""
Created on 12/5/19 10:48 AM

@author: mamba

@purpose：
"""
import time
import json
import config

from loguru import logger
from easydict import EasyDict as edict
from source.processor import Tool, SackedOfficials, SpecialPerson, \
    ViolentSearch, FaceProcess
from config import MYSQL, REDIS, DOCKER, ELASTICSEARCH, \
    frame_rate_grade


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
        hash_map = self.front_db.query_grade(channel_number)
        if not isinstance(hash_map, dict):
            hash_map = {}
        return hash_map.get("grade", 1), hash_map.get("search_tv", 0)

    def runner(self):

        while True:
            message = self.redis_conn.rpop(self.queue)

            if not message:
                logger.info("暂时没有任务，等待10s")
                time.sleep(10)
                continue

            message = json.loads(message)
            logger.info(f"取到一条人脸识别任务 {message}")
            tic = time.time()
            # 开始执行任务，更改状态为1
            self.faceworker_db.set_status(message["id"], "face_status", 1)

            grade, is_search = self.get_grade_and_search_tv(message["chan_num"])
            message.update({
                "grade": grade,
                "is_search": is_search,
                "interval": frame_rate_grade[str(grade)]
            })

            if int(is_search) == 1:
                violent_search = ViolentSearch()
                special_person = SpecialPerson(self.front_db, config.sim2search,
                                               self.es_client)
            else:
                violent_search, special_person = None, None

            tool = Tool(message)
            sacked_officials = SackedOfficials(self.front_db, config.sim2fall)
            processor = FaceProcess(message, config.address, config.frame_rate)

            df_list = processor.runner(sacked_officials=sacked_officials,
                                       special_person=special_person,
                                       violent_search=violent_search,
                                       tool=tool
                                       )
            # 　todo IoU排重、文件删除
            logger.success(f"识别成功， 用时{time.time() - tic}秒")


def runner():
    def call():
        return

    return call


if __name__ == '__main__':
    runner()
