# -*- coding: utf-8 -*-
"""
Created on 12/5/19 10:54 AM

@author: mamba

@purpose：
"""
import os

from typing import Dict, List
from loguru import logger
from elasticsearch import Elasticsearch


class ElasticsearchClient:
    """elasticsearch客户端的封装

    Examples
    --------
    """

    def __init__(self, host, port, _index, _type):
        self.es = Elasticsearch(
            host=os.environ.get('ELASTICSEARCH_ADDR') or host,
            port=port, retry_on_timeout=True
        )
        self._index = _index  # 索引名字 取名随意但要和树欣沟通好
        self._type = _type  # type   取名随意但要和树欣沟通好

        if not self.es.indices.exists(index=self._index):
            self.set_mapping()

    def set_mapping(self):
        body = {
            "mappings": {
                self._type: {
                    "properties": {
                        "frame_url": {"type": "keyword"},
                        "source": {"type": "keyword"},
                        "video_name": {"type": "keyword"},
                        "video_url": {"type": "keyword"},
                        "channel_id": {"type": "keyword"},
                        "channel_name": {"type": "keyword"},
                        "time": {"type": "keyword"},
                        "face_name": {"type": "keyword"},
                        "date": {"type": "date",
                                 "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"},
                        "create_time": {"type": "date",
                                        "format": "yyyy-MM-dd HH:mm:ss||yyyy-MM-dd||epoch_millis"}
                        # 存进来的格式　类似与　‘2018-01-01 12:00:00’
                    }
                }
            }
        }
        return self.es.indices.create(index=self._index, body=body)

    def insert(self, data: Dict):
        """插入单条数据
        """
        try:
            res = self.es.index(index=self._index, doc_type=self._type,
                                body=data)
            return res
        except Exception as e:
            logger.error(f"插入数据到es出错{str(e)}")

    def bulk(self, data: List[Dict]):
        """插入多条数据
        """
        body = []
        for d in data:
            body.append({
                "index": {
                    "index": self._index,
                    "_type": self._type
                }
            })
            body.append(d)

        try:
            res = self.es.bulk(body=body)
            logger.debug("es client build 完毕")
            return res
        except Exception as e:
            logger.error(f"bulk 存储报错{str(e)}")
