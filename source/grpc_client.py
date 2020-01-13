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
from concurrent.futures import ThreadPoolExecutor
from loguru import logger
from typing import Tuple, List, Optional, Sequence, Generator, Iterator
from source.lib import face_grpc_pb2, face_grpc_pb2_grpc, facerecog_pb2

try:
    from more_itertools import chunked
except ImportError:
    logger.debug("The package that is more_itertools is not found")
    from source.utils import chunked

MAX_WORKERS = 8


def sort_filename(frame_dir: str) -> List[str]:
    """将一个文件夹下的文件，按照帧排序, 返回获得图片的绝对路径"""
    frame_path_list = list(Path(frame_dir).glob("*.jpg"))
    frame_path_list.sort(key=lambda x: int(x.stem))
    return list(map(os.fspath, frame_path_list))


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


class ImageIter:
    """图片迭代器"""

    def __init__(self, image_files: List, max_workers: int = 8):
        """

        Parameters
        ----------
        image_files : list,
            image files
        shape : tuple of int, default None
            image with shape: default (389, 500, 3), single image's shape
        max_workers : int, default is None
            number of thread pool
        """
        self.image_files = image_files
        size = len(self.image_files)
        self.shape = (*self.get_shape(), size)
        self.max_workers = min(max_workers, size)
        self.mat = np.zeros(shape=self.shape, dtype=np.uint8)

    def __iter__(self):
        mat = self.image_loader()
        size = self.shape[3]
        for i in range(size):
            yield mat[:, :, :, i]

    def __len__(self):
        return len(self.image_files)

    def get_shape(self):
        image_path = self.image_files[0]
        shape = cv2.imread(image_path).shape
        if shape[2] != 3:
            raise ValueError("图片不是三维")
        return shape

    def image_loader(self):
        """ load image with thread pool
        thread pool is faster than multiprocess pool

        Returns
        -------
        mat : numpy.ndarray,
            with shape: (xxx, 500, 3, len(image_files))
        """
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future = executor.map(cv2.imread, self.image_files)
        for idx, img in enumerate(future):
            self.mat[:, :, :, idx] = img
        return self.mat


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
        self.max_workers = min(max_workers, len(image_iter))

    @staticmethod
    def encode_image(image: np.ndarray):
        image = cv2.imencode(".jpg", image)[1]
        return b64encode(image)

    def generate_requests(self):
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future = executor.map(self.encode_image, self.image_iter)
        return list(future)


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

    def __init__(self, address, timeout=10, max_workers=None):
        """

        Parameters
        ----------
        address : str, facecpp地址
        timeout ： int, 网络请求超时
        max_workers ： int, 线程数
        """
        self.address = address
        self.timeout = timeout
        self.max_workers = max_workers
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

    def images_feature(self, path, chunk_size=2000):
        """

        Parameters
        ----------
        path : list,
            存放图片的位置列表
        chunk_size : int,
            一次处理图片的数量
        """
        for filenames in chunked(path, chunk_size):
            image_iter = ImageIter(filenames)
            workers = min(self.max_workers, len(image_iter))
            requests = GenerateRequest(image_iter, workers).generate_requests()
            with ThreadPoolExecutor(max_workers=workers) as executor:
                face_infos_list = list(
                    executor.map(self.single_pic_feature, requests))
            # image_iter中的mat已经被GenerateRequest修改, 所以可以直接调用image_iter.mat
            # 注意python中的函数参数传递
            yield face_infos_list, image_iter.mat, filenames
