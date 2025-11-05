# FlowerClassification

## python版本
* python3.8

## 1.创建环境
### 创建conda环境
* 创建环境：```conda create -n "虚拟环境名称" python=3.8```

* 激活环境：```conda activate "虚拟环境名称"```
>
* 安装依赖库：进入项目目录后```pip install -r requirements.txt```

### 创建venv环境
* 创建环境：```py -3.8 -m venv "虚拟环境名称"```
>
>执行后，项目目录下会生成一个 venv 文件夹，包含独立的 Python 解释器和依赖库。
>
* 激活环境：```venv\Scripts\activate.bat```
>
* 安装依赖库：进入项目目录后```pip install -r requirements.txt```

## 2.运行代码
### 训练代码
* ```cd ./code```进入代码目录后,在终端中输入```python train.py```进行模型训练

### 预测代码
* 在终端中输入```python predict.py```进行批量预测

> 结果保存在results目录中
