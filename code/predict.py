import os
import csv
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model
import argparse

def preidct(model_path, image_dir, output_dir, image_shape=(224, 224, 3)):
    """
    model_path: 训练好的模型的路径
    image_dir: 图片所在目录
    output_dir: 输出结果的路径
    image_shape: 加载图片的形状,与模型训练时一致
    """
    os.makedirs(output_dir, exist_ok=True)

    model = load_model(model_path)
    class_name = [
    164, 165, 166, 167, 169, 171, 172, 173, 174, 176, 177, 178, 179, 180,
    183, 184, 185, 186, 188, 189, 190, 192, 193, 194, 195, 197, 198, 199,
    200, 201, 202, 203, 204, 205, 206, 207, 208, 209, 210, 211, 212, 213,
    214, 215, 216, 217, 218, 220, 221, 222, 223, 224, 225, 226, 227, 228,
    229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242,
    243, 244, 245, 1734, 1743, 1747, 1749, 1750, 1751, 1759, 1765, 1770, 
    1772, 1774, 1776, 1777, 1780, 1784, 1785, 1786, 1789, 1796, 1797,
    1801, 1805, 1806, 1808, 1818, 1827, 1833
    ]

    image_files = [f for f in os.listdir(image_dir)]
    image_files.sort()

    with open(output_dir, 'w', newline='', encoding='utf-8') as csvfile:
        fieldname = ['img_name', 'predicted_class', 'confidence']
        writer = csv.DictWriter(csvfile, fieldnames=fieldname)
        writer.writeheader()

        for image_file in image_files:
            image_path = os.path.join(image_dir, image_file)

            image = load_img(image_path, target_size=image_shape)
            image_array = img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = image_array / 255.0

            predictions = model.predict(image_array)
            predict_class_index = np.argmax(predictions[0])
            predict_class_name = class_name[predict_class_index]

            confidence = round(float(predictions[0][predict_class_index]), 2)

            writer.writerow({
                'img_name': image_file,
                'predicted_class': predict_class_name,
                'confidence': confidence
            })
    
    print(f"预测完成,结果已保存至{output_dir}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='花卉分类模型预测')
    parser.add_argument('test_dir', type=str, help='测试图片目录')
    parser.add_argument('output_path', type=str, help='预测结果输出路径(CSV文件)')
    args = parser.parse_args()

    model_path = "../model/best_model.h5"
    test_dir = args.test_dir
    output_path = args.output_path

    preidct(model_path, test_dir, output_path)