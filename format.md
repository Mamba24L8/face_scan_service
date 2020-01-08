# 项目代码格式说明

## 模块导入

在导入模块时主要分为三部分， 一种时`import`类型，另外是`from-import`类型，`from-import`类型又包含两种，一种是导入自己项目中的模块，另外一种就是导入非自己项目中的模块。这三种必须要分组分类进行代码组织，如下：

```python
import os
import json
import numpy as np
import pandas as pd

from .package import module
from .package.module import class

from datetime import date, datetime
from typing import Iterable, Union, Dict
```

## 注释

对部分**关键变量**需要进行注释，说明。对复杂或稍微复杂的函数需要进行注释说明其功能，参数及返回结果。另外，尽可能的使用`type hint`的形式来完成代码示例如下：

```python
var: int = 10


def function(a: float, b: int=20) -> float:
    """实现两个变量的加法
    parameter
    ---------
        a: float, 参数a的说明
        b: int, 参数b的说明, 默认值为20
    return
    ------
        float, 说明
    """
    return a + b
```

## 代码块

在实现过程中如果有代码的公用的部分，需要将公用的部分进行封装，放入到项目的`common.py`或者`tools.py`下。不同开发人员都在公用，如果要修改该公用的模块，修改后需要保持其他开发人员能继续正常的使用。

代码块中的相关命名，全局变量必须全大写和下划线，函数命名必须小写和下划线，类命名时首个单词字母为大写。如：`LOCAL_VARIABLES`，`function_name`，`ClassNameFormat`。

## 配置

在项目开发能配置的，均放入到`config.py`中。配置中不能和一些全局变量混淆。配置文件中的名称需要自己定义明确，不能在其他的配置项中添加自己的配置（除非一起讨论确定后）。

## 项目结构

```
├── data  # 数据
├── doc   # 项目文档
├── README.md  # 项目说明
├── requirement.txt  # 项目中使用的python包
├── scripts  # 可执行文件
├── source   # (1) 源代码中的所有模块、包都应该放在此目录。不要置于顶层目录。#(2)                 其子目录tests/存放单元测试代码； (3) 程序的入口最好命名main.py。

```
