import numpy as np
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.preprocessing.image import ImageDataGenerator
from keras.constraints import maxnorm
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import tensorflow_datasets as tfds
import os

# Гиперпараметры
batch_size = 64
# 10 категорий для изображений  (CIFAR-10)
num_classes = 10
# количество эпох
epochs = 30


def load_data():
    def preprocess_image(image, label):
        # преобразование целочисленного диапазона [0, 255] в диапазон действительных чисел [0, 1]
        image = tf.image.convert_image_dtype(image, tf.float32)
        return image, label

    # загрузка датасета CIFAR-10, его разбиение на обучающий и тестовый набор
    ds_train, info = tfds.load("cifar10", with_info=True, split="train", as_supervised=True)
    ds_test = tfds.load("cifar10", split="test", as_supervised=True)
    # перемешиваем набор данных, предварительно обрабатывая, и разделяем по пакетам
    ds_train = ds_train.shuffle(1024).map(preprocess_image).batch(batch_size)
    ds_test = ds_test.shuffle(1024).map(preprocess_image).batch(batch_size)
    return ds_train, ds_test, info


def create_model(input_shape):
    # построение модели
    model = Sequential()
    model.add(Conv2D(filters=32, kernel_size=(3, 3), padding="same", input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # сглаживание
    model.add(Flatten())
    model.add(Dense(1024, activation="relu"))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation="softmax"))
    # печатаем итоговую архитектуру модели
    model.summary()
    # компиляция модели с помощью оптимизатора Адама
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


if __name__ == "__main__":
    ds_train, ds_test, info = load_data()
    # конструирование модели
    model = create_model(input_shape=info.features["image"].shape)
    # обучение модели
    model.fit(ds_train, validation_data=ds_test, epochs=epochs, batch_size=batch_size)
    # сохраняем модель на диске
    model.save("model-cifar10-v1.h5")
