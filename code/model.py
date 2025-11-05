from keras import Input, Model
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, Add
from keras.optimizers import Adam
from keras.metrics import Precision

def build_residual_block(x, filters):
    """新增残差块：解决深层网络梯度消失问题"""
    residual = x
   
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    if residual.shape[-1] != filters:
        residual = Conv2D(filters, (1, 1), padding='same')(residual)
        residual = BatchNormalization()(residual)
    x = Add()([x, residual]) 
    return x

def build_model(num_classes=100, image_shape=(224, 224, 3)):
    input_tensor = Input(shape=image_shape)
    
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = MaxPooling2D((2, 2))(x)
    
    x = build_residual_block(x, 64)
    x = MaxPooling2D((2, 2))(x)
    
    x = build_residual_block(x, 128)
    x = MaxPooling2D((2, 2))(x)
    
    x = build_residual_block(x, 256) 
    x = MaxPooling2D((2, 2))(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.45)(x)  
    x = Dense(256, activation='relu')(x)  
    x = BatchNormalization()(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=input_tensor, outputs=output_tensor)
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(name='precision')]
    )
    return model