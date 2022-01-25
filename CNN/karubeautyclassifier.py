import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import pathlib
x_train =[]
y_train = []
x_test=[]
y_test =[]
train_dir = pathlib.Path('/media/onesmus/dev/dev/machine learning/datasets/Datasets/beautytest/train')
test_dir = pathlib.Path('/media/onesmus/dev/dev/machine learning/datasets/Datasets/beautytest/test')
labels_dic ={
    'average':0,
    'beautiful':1
}
train_images_dic={
    'average':list(train_dir.glob('average/*')),
    'beautiful':list(train_dir.glob('beautiful/*'))
}
test_images_dic={
    'average':list(test_dir.glob('average/*')),
    'beautiful':list(test_dir.glob('beautiful/*'))
}

k = list(train_dir.glob('average/*'))
print(str(k))
print(str(k[0]))
image = plt.imread(k[0])
print(image.shape)
plt.imshow(image)
plt.show()


def read_train_images(data_dic_category):
    for labels,images in data_dic_category.items():
        for image in images:
            img = plt.imread(str(image))
            x_train.append(img)
            y_train.append(labels_dic[labels])
read_train_images(train_images_dic)


def read_test_images(data_dic_category):
    for labels,images in data_dic_category.items():
        for image in images:
            img = plt.imread(str(image))
            x_test.append(img)
            y_test.append(labels_dic[labels])
read_test_images(test_images_dic)

model =keras.Sequential([
    layers.Conv2D(64,(3,3),activation='relu',input_shape=(224,224,3)),
    layers.MaxPool2D(2,2),
    layers.Flatten(),
    layers.Dense(128,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.Adam,loss=tf.keras.losses.BinaryCrossentropy,metrics=tf.keras.metrics.Accuracy)

model.fit(x_train,y_train,epochs=30)
model.evaluate(x_test,y_test)

