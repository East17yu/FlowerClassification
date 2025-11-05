import pandas as pd
import os
from pathlib import Path
import shutil
import random
from tqdm import tqdm
from keras.preprocessing.image import ImageDataGenerator
from dataclasses import dataclass
from PIL import Image

def get_numClasses():
    """
    获取图像种类数量
    """
    df = pd.read_excel('../data/train_labels.xlsx')
    return df['category_id'].nunique()

def check_image(image_path):
    """
    检查图片文件是否有效
    """  
    try:
        with Image.open(image_path) as img:
            # 尝试加载图像数据
            img.load()
            # 尝试转换为RGB模式（检测格式问题）
            if img.mode not in ('RGB', 'L'):
                img = img.convert('RGB')
            return True, "有效文件"
    except Exception as e:
        return False, str(e)

def data_process():
    """
    将原数据集中的图片按照表中的类别ID分类保存
    """
    
    dataLabel_path = '../data/train_labels.xlsx'
    data_path = '../data/train'
    target_dir = '../data/newData'

    if os.path.exists(target_dir):
        print("按类别分类的数据集已经存在!")
        return 
    

    print("-----将数据集按类别分类-----")
    df = pd.read_excel(dataLabel_path)

    Path(target_dir).mkdir(exist_ok=True)
    copied_files = 0
    invalid_files = 0 
    invalid_records = []  

    num_classes = get_numClasses()

    for index, row in df.iterrows():
        filename = row['filename']
        category_id = row['category_id']
        image_path = os.path.join(data_path, filename)

        is_valid, reason = check_image(image_path)
        if not is_valid:
            invalid_files += 1
            invalid_records.append(f"文件: {filename}, 原因: {reason}")
            continue

        output_dir = os.path.join(target_dir, str(category_id))
        Path(output_dir).mkdir(exist_ok=True)

        output_path = os.path.join(output_dir, filename)
        shutil.copyfile(image_path, output_path)
        copied_files += 1

    print(f"共复制了{copied_files}个有效文件, 种类数: {num_classes}")
    print(f"发现{invalid_files}个无效文件:")
    for record in invalid_records:
        print(f"{record}")


def split_train_val():
    """
    划分训练集和验证集
    """
    data_dir = '../data/newData'
    target_dir = '../data/dataSet'
    train_ratio = 0.8

    if os.path.exists(target_dir):
        print("训练集和验证集已经存在!")
        return 

    categories = [cls for cls in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, cls))]

    total_files = 0
    all_files = []
    print("-----划分数据集中-----")
    for category in categories:
        cls_origin_path = os.path.join(data_dir, category)
        image_files = [f for f in os.listdir(cls_origin_path)]
        random.shuffle(image_files)
        train_len = int(len(image_files) * train_ratio)
        all_files.append((category, image_files, train_len))
        total_files += len(image_files)

    with tqdm(total=total_files, desc='数据集分类进度') as pbar:
        for category, image_files, train_len in all_files:
            cls_origin_path = os.path.join(data_dir, category)
            train_path = os.path.join(target_dir, "train", category)
            val_path = os.path.join(target_dir, "val", category)
            os.makedirs(train_path, exist_ok=True)
            os.makedirs(val_path, exist_ok=True)

            train_files = image_files[:train_len]
            val_files = image_files[train_len:]

            for img in train_files:
                src = os.path.join(cls_origin_path, img)
                dst = os.path.join(train_path, img)
                shutil.copyfile(src, dst)
                pbar.update(1)

            for img in val_files:
                src = os.path.join(cls_origin_path, img)
                dst = os.path.join(val_path, img)
                shutil.copyfile(src, dst)
                pbar.update(1)

    print("-----数据集划分完成-----")

def get_data_generator(train_path, val_path):
    data_process()
    split_train_val()

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=20,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        zoom_range=0.2,
        shear_range=0.2,
        fill_mode='nearest'
    )
    train_generator = train_datagen.flow_from_directory(
        train_path,
        target_size=(224, 224), 
        batch_size=32,
        class_mode='categorical',
        shuffle=True
    )

    val_datagen = ImageDataGenerator(rescale=1. / 255)
    val_generator = val_datagen.flow_from_directory(
        val_path,
        target_size=(224, 224),
        batch_size=32,
        class_mode='categorical',
        shuffle=False
    )

    return train_generator, val_generator

def classes_indices(path):
    classes = [cla for cla in os.listdir(path) if os.path.isdir(os.path.join(path, cla))]
    classes.sort()
    class_indices = dict((k, v) for v, k in enumerate(classes))

    return class_indices

@dataclass
class Config:
    """
    模型参数配置
    """
    data_root: str = "../data/newData"
    train_path: str = "../data/dataSet/train"
    val_path: str = "../data/dataSet/val"
    model_path: str = "../model/best.h5"
    img_size: int = 224
    batch_size: int = 32
    optimizer: str = 'Adam'
    lr: float = 1e-3
    epochs: int = 100
    num_classes: int = get_numClasses()

