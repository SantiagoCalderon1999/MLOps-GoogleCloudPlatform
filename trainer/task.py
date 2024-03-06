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

from tensorflow.keras.datasets import mnist
from scripts.helper import preprocess_data

(x_train, labels_train), (x_test, labels_test) = mnist.load_data()
x_train, y_train, x_test, y_test = preprocess_data(x_train, labels_train, x_test, labels_test)

model = get_model_definition(x_train)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(
    x_train, y_train, validation_data=(x_test, y_test), epochs=20, batch_size=256
)
BUCKET_ROOT='gcs/models-mnist-training'
model.save(f'{BUCKET_ROOT}/model_output')