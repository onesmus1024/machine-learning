import tensorflow as tf
import seaborn as sn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split


insurance_df = pd.read_csv('datasets/cleanedDataset/cleanedinsurancedata.csv',index_col=0)
print(insurance_df.head())
x = insurance_df[['age','bmi','sex_num','smoker_num']]
y = insurance_df[['charges']]
k =np.array(y)
l=np.array(x)

# sn.pairplot(x,diag_kind='kde')
# plt.show()
print(insurance_df.describe())
print(insurance_df.isna().sum())
y=y/1000
print(x.shape)
print(y.shape)

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)

model =keras.Sequential([
    layers.Dense(512,input_shape=[4]),
    layers.Dropout(0.1),
    layers.Dense(1)
])

model.compile(optimizer='adam',loss='mae',metrics=['mae'])
history = model.fit(x_train,y_train,epochs=10,validation_split=0.2,batch_size=100)

pred = model.predict(x_train)
print('predicted')
print(pred[0])
print('actual value is ')
print(y_train)


plt.plot(history.history['loss'],label='loss',c='r')
plt.plot(history.history['val_loss'],label='val_loss',c='b')
# plt.plot(history.history['mae'],range(0,1000))
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()
plt.show()
