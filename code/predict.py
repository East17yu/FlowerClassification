import os
import csv
import numpy as np
from keras.preprocessing.image import load_img, img_to_array
from keras.models import load_model

def preidct(model_path, image_dir, output_dir, image_shape=(224, 224, 3)):
    """
    model_path: 训练好的模型的路径
    image_dir: 图片所在目录
    output_dir: 输出结果的路径
    image_shape: 加载图片的形状,与模型训练时一致
    """

    model = load_model(model_path)
    class_name = model.class_name
    image_files = [f for f in os.listdir(image_dir)]

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
    
    model_path = "../model/best.h5"
    image_dir = "../data/test"
    output_dir = "../results/submission.csv"

    preidct(model_path, image_dir, output_dir)