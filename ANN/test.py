import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split


df = pd.read_csv("/home/onesmus/dev/machine_learning/datasets/cleanedDataset/Churn_Modelling.csv")
print(df.head(10))
x_train = df.iloc[:, 3:13].values
y_train = df.iloc[:, 13].values
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=0)

model = keras.models.Sequential([
    keras.layers.Dense(6, input_shape=(13,), activation="relu"),
    keras.layers.Dense(6, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])


model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=100)
model.evaluate(x_test, y_test)
