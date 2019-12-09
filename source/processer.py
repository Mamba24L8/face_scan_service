# -*- coding: utf-8 -*-
"""
Created on 12/5/19 10:53 AM

@author: mamba

@purpose：
"""
import os
import json
import numpy as np

from pathlib import Path
from datetime import datetime

from abc import ABCMeta, abstractmethod
from typing import Dict, Callable, List, Tuple


class MessageProcess:
    pass


class IouTask:
    pass


class TargetPerson(metaclass=ABCMeta):
    """人物对象的处理基类"""

    @abstractmethod
    def person_info_loader(self, *args, **kwargs):
        """从数据库中加载人物信息并进行格式化

        See Also
        --------
            format_data
        """
        pass

    @abstractmethod
    def process(self, *args, **kwargs):
        """针对不同人物进行处理过程的主程序"""
        pass

    @abstractmethod
    def runner(self, *args, **kwargs):
        pass


class SackedOfficials(TargetPerson):
    pass


class SpecialPerson(TargetPerson):
    pass
