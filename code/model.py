from keras import Input, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, Dropout, Dense, Add
from tensorflow.keras.layers import Reshape, Multiply, Activation
from tensorflow.keras.optimizers import Adam
from keras.metrics import Precision

def se_block(x, filters, ratio=16):

    se = GlobalAveragePooling2D()(x)
    se = Dense(filters//ratio, activation='relu')(se)
    se = Dense(filters, activation='sigmoid')(se)
    se = Reshape((1, 1, filters))(se)
    return Multiply()([x, se])

def build_residual_block(x, filters):
    residual = x

    
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters, (3, 3), activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = se_block(x, filters)
    
    if residual.shape[-1] != filters:
        residual = Conv2D(filters, (1, 1), padding='same')(residual)
        residual = BatchNormalization()(residual)
    
    x = Add()([x, residual])
    x = Activation('relu')(x)  
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