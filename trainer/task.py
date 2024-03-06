from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Dense,
    Flatten,
    Dropout,
    Input,
    BatchNormalization,
)


def get_model_definition(x_train):
    inputs = Input(shape=x_train.shape[1:])
    x = Conv2D(filters=64, kernel_size=(5, 5), activation="relu")(inputs)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.5)(x)
    x = BatchNormalization()(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    x = Dropout(rate=0.5)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(rate=0.5)(x)
    outputs = Dense(10, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)