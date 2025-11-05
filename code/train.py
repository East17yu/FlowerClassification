from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.optimizers import Adam
from model import build_model
from utils import  get_data_generator, get_numClasses, Config
import os
from ezmodel import build_model_lightweight
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def train():
    # 获取模型配置信息
    config = Config()
    # 获取数据生成器和图像种类数量
    num_classes = get_numClasses()
    train_generator, val_generator = get_data_generator(config.train_path, config.val_path)
    
    model = build_model(num_classes)
    # 回调函数
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5),
        ModelCheckpoint(
            os.path.join('../model', 'best.h5'),
            save_best_only=True,
            monitor='val_accuracy',
            mode='max'
        )
    ]

    history = model.fit(
        train_generator,
        epochs=config.epochs,
        validation_data=val_generator,
        callbacks=callbacks
    )

    model.save(config.model_path)

if __name__ == '__main__':
    train()