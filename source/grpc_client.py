# -*- coding: utf-8 -*-
"""
Created on 12/5/19 10:53 AM

@author: mamba

@purpose： grpc客服端
"""
import os
import cv2
import grpc
import numpy as np

from base64 import b64encode
from pathlib import Path
from functools import singledispatch
from more_itertools import chunked
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from typing import Tuple, List, Optional, Sequence, Generator, Iterator
from source.lib import face_grpc_pb2, face_grpc_pb2_grpc, facerecog_pb2

MAX_WORKERS = 8


def sort_filename(frame_dir: str) -> List[str]:
    """将一个文件夹下的文件，按照帧排序, 返回获得图片的绝对路径"""
    frame_path_list = list(Path(frame_dir).glob("*.jpg"))
    frame_path_list.sort(key=lambda x: int(x.stem))
    return list(map(os.fspath, frame_path_list))


class ImageIter:
    """图片迭代器"""

    def __init__(self, image_files: List, shape: Tuple = (389, 500, 3),
                 max_workers: int = MAX_WORKERS):
        """

        Parameters
        ----------
        image_files : list,
            image files
        shape : tuple of int, default (389, 500, 3)
            image with shape: default (389, 500, 3), single image's shape
        max_workers : int, default is None
            number of thread pool
        """
        self.image_files = image_files
        self.shape = (*shape, len(self.image_files))
        self.max_workers = max_workers
        self.mat = np.zeros(shape=self.shape, dtype=np.uint8)

    def __iter__(self):
        mat = self.image_loader()
        size = self.shape[3]
        for i in range(size):
            yield mat[:, :, :, i]

    def __len__(self):
        return len(self.image_files)

    def image_loader(self):
        """ load image with thread pool
        thread pool is faster than multiprocess pool

        Returns
        -------
        mat : numpy.ndarray,
            with shape: (389, 500, 3, len(image_files))
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future = executor.map(cv2.imread, self.image_files)
        for idx, img in enumerate(future):
            self.mat[:, :, :, idx] = img
        return self.mat


def image_loader(image_files: list, shape=(389, 500, 3), max_workers=None):
    """ load images by thread pool

    Parameters
    ----------
    image_files : list,
        image files
    shape : tuple of int, default (389, 500, 3)
        image with shape: default (389, 500, 3), single image's shape
    max_workers : int, default is None
        number of thread pool

    Returns
    -------
    mat : numpy.ndarray,
        with shape: (389, 500, 3, len(image_files))
    """
    shape = (*shape, len(image_files))
    mat = np.zeros(shape=shape, dtype=np.uint8)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        res = executor.map(cv2.imread, image_files)
    for idx, img in enumerate(res):
        mat[:, :, :, idx] = img
    return mat


class GenerateRequest:
    """Grpc requests of a chunk size image"""

    def __init__(self, image_iter, max_workers=MAX_WORKERS):
        """

        Parameters
        ----------
        image_iter : Iterator or ImageIter
        max_workers : int, default is 4
            number of thread pool
        """
        self.image_iter = image_iter
        self.max_workers = max_workers

    @staticmethod
    def encode_image(image: np.ndarray):
        image = cv2.imencode(".jpg", image)[1]
        return b64encode(image)

    def generate_requests(self):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future = executor.map(self.encode_image, self.image_iter)
        return list(future)


def get_request(filename: str) -> Tuple[str, np.ndarray]:
    """使用base64对图片转换

    Parameters
    ----------
    filename : str, 单个图片路径

    Returns
    -------
    req : str,
          Encode the bytes-like object s using Base64 and return a bytes object
    img : np.ndarray,
          brief Loads an image from a file
    """
    img = cv2.imread(filename)
    img_cv_encode = cv2.imencode(".jpg", img)[1]
    return b64encode(img_cv_encode), img


class GetFaceFeature:
    """

    Examples
    --------
    >>> from glob import glob
    >>> import pickle
    >>> path = glob("/home/mamba/20180625/*.jpg")[:400]
    >>> with GetFaceFeature(address="223.71.97.245:8001") as gff:
    ...     for x in gff.images_feature(path=path, chunk_size=20):
    >>>         with open("/home/mamba/Projects/face_scan_service/data/r.pkl", "wb") as f:
    >>>             pickle.dump(f)
    """
    def __init__(self, address, timeout=10):
        self.address = address
        self.timeout = timeout
        self.channel = grpc.insecure_channel(self.address)
        self.stub = face_grpc_pb2_grpc.face_grpcServiceClsStub(self.channel)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self.channel:
            return
        try:
            self.channel.close()
            logger.debug("Grpc Channel Have Closed")
        except Exception as e:
            raise Exception(e)

    def single_pic_feature(self, req):
        """

        Parameters
        ----------
        req : str, the request of grpc request
        time_out : int

        Returns
        -------
        face_infos : List[Tuple],
            the feature with shape (1, 512),
            bounding_box with shape (1, 4)
            landmark of face with shape (5, 2)
        """
        try:
            response = self.stub.face_grpc(
                face_grpc_pb2.face_grpcRequest(req=req),
                timeout=self.timeout)
            response_facerecog = facerecog_pb2.FaceRecogResponse()
            response_facerecog.ParseFromString(response.res)

            face_infos = []
            for face in response_facerecog.faces:
                feature = np.array(
                    [[face_feature for face_feature in face.face_feature]])
                bounding_box = np.array(
                    [[face.rect.x, face.rect.y, face.rect.x + face.rect.width,
                      face.rect.y + face.rect.height]])
                landmarks = np.array([(lm.x, lm.y) for lm in face.landmarks])

                face_infos.append((feature, bounding_box, landmarks))
            return face_infos
        except Exception as e:
            logger.error(f"人脸特征获取错误，请检查人脸docker服务是否正常，错误信息 {e}")
            return None

    def images_feature(self, path, chunk_size=2000, max_workers=None):
        for filenames in chunked(path, chunk_size):
            image_iter = ImageIter(filenames)
            _requests = GenerateRequest(image_iter).generate_requests()
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                face_infos_list = list(
                    executor.map(self.single_pic_feature, _requests))
            yield face_infos_list, image_iter.mat, filenames
