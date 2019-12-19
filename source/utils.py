# -*- coding: utf-8 -*-
"""
Created on 12/5/19 10:54 AM

@author: mamba

@purpose：
"""
import cv2
import json
import time
import numpy as np

from loguru import logger
from functools import wraps
from PIL import Image, ImageDraw, ImageFont


def execution_time(func):
    """ Decorator that reports the execution time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        logger.debug("{}:\t{}".format(func.__name__, end - start))
        return result

    return wrapper


def get_host_ip():
    """在docker中，获得宿主机的ip"""
    pass


def json_dump(filename, data):
    """

    Parameters
    ----------
    filename : str, 路径
    data : dict， 数据
    """
    with open(filename, "w") as f:
        json.dump(data, f)


def json_load(filename):
    """

    Parameters
    ----------
    filename : str, 路径
    data : dict， 数据
    """
    with open(filename, "r") as f:
        data = json.load(f)
    return data


def draw_box(image, bbox):
    """ 画边框

    Parameters
    ----------
    image : numpy.ndarray
         Image with shape `H, W, 3`
    bbox : numpy.ndarray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.

    Returns
    -------
    image : numpy.ndarray
    """
    bbox = bbox[0]
    cv2.rectangle(image, (int(bbox[0]), int(bbox[1])),
                  (int(bbox[2]), int(bbox[3])),
                  (0, 0, 255), 3)
    return image


def draw_box_text(image, bbox, text):
    """添加文字
    cv2.putText不能写中文，所以这儿先把图片转换成PIL格式，用PIL的方法写入中文，然后再转回cv2的格式

    Parameters
    ----------
    image : numpy.ndarray
         Image with shape `H, W, 3`
    bbox : numpy.ndarray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    text : str

    Returns
    -------
    image : numpy.ndarray
    """
    x, y, _, _ = [int(i) for i in bbox[0]]
    # 图像从OpenCV格式转换成PIL格式
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 字体  字体*.ttc的存放路径一般是： /usr/share/fonts/opentype/noto/ 查找指令locate *.ttc
    font = ImageFont.truetype('NotoSansCJK-Black.ttc', 20)
    draw = ImageDraw.Draw(image)
    draw.text((int(x), int(y - 30)), text=text, font=font, fill=(255, 255, 0))
    # 转换回OpenCV格式
    image = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return image


def save_image(image, bbox, text, filename):
    """画框、标注名字、保存

    Parameters
    ----------
    image : numpy.ndarray
         Image with shape `H, W, 3`
    bbox : numpy.ndarray
        Bounding boxes with shape `N, 4`. Where `N` is the number of boxes.
    text : str
    filename : str
        save image to filename, absolute path
    
    Examples
    --------
    >>> img_path = "/home/mamba/20180625/0028828.jpg"
    >>> save_path = "/home/mamba/bb.jpg"
    >>> import numpy as np
    >>> img = cv2.imread(img_path)
    >>> save_image(img, np.array([[136, 89, 198, 184]]), "科比", save_path)
    True
    """
    try:
        image = draw_box(image, bbox)
        image = draw_box_text(image, bbox, text)
        cv2.imwrite(filename, image)
        return True
    except Exception as e:
        logger.error(f"保存图片失败， 错误原因{e}")
        return False


if __name__ == '__main__':
    import doctest

    doctest.testmod()
    # img = draw_box(img, np.array([[136, 89, 198, 184]]))
    # img = draw_box_text(img, np.array([[136, 89, 198, 184]]), "科比")
    # cv2.imwrite("/home/mamba/bb.jpg", img)
