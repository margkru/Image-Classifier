from flask import Flask
from train import load_data
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np

# CIFAR-10 classes
categories = ['Самолет', 'Автомобиль', 'Птица', 'Кошка', 'Олень', 'Собака', 'Лягушка', 'Лошадь', 'Корабль', 'Грузовик']


def test_model():
    # загрузка тестового набор
    ds_train, ds_test, info = load_data()
    # загрузка сохраненной модели
    model = load_model("model-cifar10-v1.h5")
    return ds_test, model


def predict(ds_test, model):
    # получить прогноз для случайных изображений
    data_sample = next(iter(ds_test))
    predict_lable = []
    sample_label = []
    for i in range(4):
        sample_image = data_sample[0].numpy()[i]
        sample_label.append(categories[data_sample[1].numpy()[i]])
        prediction = np.argmax(model.predict(sample_image.reshape(-1, *sample_image.shape))[0])
        predict_lable.append(categories[prediction])
        plt.imsave("static/img/" + str(i) + ".jpg", sample_image)
    return sample_label, predict_lable
