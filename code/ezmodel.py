from keras import Input, Model  # 函数式API核心依赖
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense
from keras.optimizers import Adam

def build_model_lightweight(num_classes, image_shape=(224, 224, 3)):
    # 1. 显式定义输入张量（函数式API入口）
    input_tensor = Input(shape=image_shape)
    
    # 2. 构建特征提取网络（链式连接，前一层输出作为后一层输入）
    # Block1
    x = Conv2D(16, (3, 3), activation='relu')(input_tensor)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block2
    x = Conv2D(32, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block3
    x = Conv2D(64, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # Block4
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    # 3. 分类器头部
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)  # 输出张量
    
    # 4. 定义完整模型（绑定输入和输出）
    model = Model(inputs=input_tensor, outputs=output_tensor)
    
    # 5. 保持原训练配置不变
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model