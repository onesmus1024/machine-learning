
import  tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import  layers

x=np.arange(12)
y = np.arange(12)*2
lo=layers.Dense(units=1,input_shape=[1])
model = keras.Sequential([
    lo
])
model.compile(optimizer='sgd',loss='mean_squared_error')
model.fit(x,y,epochs=100)
print(model.predict([6]))
print("#"*50)
print(lo.get_weights())
