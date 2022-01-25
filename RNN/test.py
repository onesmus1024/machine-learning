# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing import sequence

import numpy as np

x = np.arange(start=10,step=10,stop=70)
x=x.reshape([2,3])
print(x)
print(x.shape)
print("#"*100)
print(x[:,:-1])
print("#"*100)
print(x[:,-1])
