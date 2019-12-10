# -*- coding: utf-8 -*-
"""
Created on 12/5/19 10:54 AM

@author: mamba

@purpose：与数据库交互部分，主要是监控的状态修改
"""

import pymysql

from loguru import logger
from DBUtils.PersistentDB import PersistentDB
from pymysql.cursors import DictCursor


class MysqlUtil:
    """数据库连接池，为每一个线程单独创建一个线程，连接关闭不会真正关闭，只有等程序关闭之时，才
    会关闭连接。参数中的ping方法默认会在请求之时，探测服务器是否能连接成功

    """

    def __init__(self, **kwargs):
        pool = PersistentDB(creator=pymysql, **kwargs)
        self.connect = pool.connection()


class FrontDb(MysqlUtil):
    """
    Examples
    --------
    >>> config = {"host":"192.168.43.159","user":"root","password":"123456",
    ... "database":"web_system"}
    >>> front = FrontDb(**config)
    >>> print(front.query_grade(172))
    >>> print(front.query_person_list("use_ai", 0))
    """

    def query_grade(self, tv_num):
        """获得等级"""
        sql = ("SELECT `grade`, `search_tv` "
               "FROM `tv_table` "
               "WHERE `tv_num`={}".format(tv_num)
               )
        try:
            with self.connect.cursor(DictCursor) as cursor:
                cursor.execute(sql)
                tv_dic = cursor.fetchone()
            self.connect.commit()
            if tv_dic:
                logger.info(f"查询台的级别和是否用于ai搜索成功{tv_num} :{tv_dic}")
                return tv_dic
        except Exception as e:
            logger.info(f"\n ----- 查询台号报错 {e} ----- ")
        finally:
            self.connect.close()

    def query_person_list(self, field, status):
        """获得目标人物的信息"""
        sql = ("SELECT `person_id`, `person_name`, `feature` "
               "FROM `target_info` "
               "WHERE {}={}".format(field, status)
               )
        try:
            with self.connect.cursor(DictCursor) as cursor:
                cursor.execute(sql)
                person_dic = cursor.fetchall()
            self.connect.commit()
            if person_dic:
                logger.info(f"在数据库中查询目标人物 {len(person_dic)} 个")
                return person_dic
        except Exception as e:
            logger.error(f"\n ----- 查询目标人物报错 {e} ----- ")
            return
        finally:
            self.connect.close()


class FaceWorkerDB:
    """用来设置任务状态

    Examples
    --------
    >>> config = {"host":"192.168.43.159","user":"root","password":"123456",
    ... "database":"web_system", "table":"video_face_status"}
    """

    def __init__(self, **kwargs):
        if "table" in kwargs:
            self.table = kwargs["table"]
            del kwargs["table"]
        pool = PersistentDB(creator=pymysql, **kwargs)
        self.connect = pool.connection()

    def set_status(self, _id, field, status):
        """任务状态"""
        sql = ("UPDATE {} SET {}={} "
               "WHERE task_id={}".format(self.table, field, status, _id)
               )
        try:
            with self.connect.cursor() as cursor:
                cursor.execute(sql)
            self.connect.commit()
            logger.info("置 {} 视频状态成功".format(_id))
        except Exception as e:
            logger.error("设置视频状态报错：\n{}".format(repr(e)))
        finally:
            self.connect.close()

    def set_data_error(self, _id):
        """报错之后，进行状态更新"""
        sql = ("UPDATE {} SET data_err=data + 1 "
               "WHERE task_id={}".format(self.table, _id)
               )
        logger.info("SQL: {}".format(sql))

        try:
            with self.connect.cursor() as cursor:
                cursor.execute(sql)
            self.connect.commit()
            logger.info("data_err加1成功： {} ".format(_id))
        except Exception as e:
            logger.error("更新 data_error 状态报错 {}".format(e))
        finally:
            self.connect.close()

    def query_data_err_by_id(self, _id):
        sql = ("SELECT data_err FROM {} "
               "WHERE task_id={}".format(self.table, _id)
               )
        try:
            with self.connect.cursor() as cursor:
                cursor.execute(sql)
            data_err = cursor.fetchone()
            if data_err:
                self.connect.commit()
                return data_err

        except Exception as e:
            logger.error(f"\n ----- 查询id为 {_id} 的data_err报错 {e} ----- ")
        finally:
            self.connect.close()
