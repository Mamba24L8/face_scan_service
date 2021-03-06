# -*- coding: utf-8 -*-
"""
Created on 12/5/19 10:49 AM

@author: mamba

@purpose：
"""

sim2fall = 0.5
sim2search = 0.6
key_person_similarity = 0.5

whether_search = "yes"
frame_rate = 6
frame_rate_grade: {"1": 1, "2": 2/3, "3": 3, "4": 6}

# MYSQL
MYSQL = {
    "host": "10.97.34.206",
    "port": 3306,
    "user": "root",
    "password": "123456",
    "db": {
        "video_db": "video_data_info",
        "search_db": "face",
        "front_db": "web_system"
    },
    "tb": {
        "video_tb": "video_face_status",
        "search_tb": "face_feature_violent",
        "tv_tb": "tv_table",
        "target_tb": "target_info"
    },
    "charset": "utf8"
}

# redis配置文件
REDIS = {
    "host": "localhost",
    "port": 6379,
    "user": None,
    "password": None,
    "db": 0,
    "object": {
        "face_tasks": "face_framing_task",
        "iou_tasks": "iou_tasks"
    }
}

# face_cpp docker
address = ""
DOCKER = {"docker-name": "10.107.193.162",
          "host": "10.107.193.162",
          "port": 8001
          }

ELASTICSEARCH = {
    "host": "10.242.189.206",
    "port": 30920,
    "index": "face_list_test",
    "type": "txt"
}

if __name__ == '__main__':
    from easydict import EasyDict as edict

    d = edict(MYSQL)
    print(d.db.video_db)
