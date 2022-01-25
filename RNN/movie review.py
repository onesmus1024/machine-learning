import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import imdb
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#loading the dataset
(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)
#padding to make all reviews of the same length
train_data = pad_sequences(train_data,padding='post',maxlen=256)
test_data = pad_sequences(test_data,padding='post',maxlen=256)

vocal_size=10000
model = keras.Sequential([
    layers.Embedding(vocal_size, 16,input_length=256),
    layers.Bidirectional(layers.LSTM(16)),
    
    layers.Dense(512,activation='relu'),
    layers.Dense(1,activation='sigmoid')
])
#sicing traing data to have validation data
validation_data=train_data[:10000]
validation_label=train_labels[:10000]
partial_train_data=train_data[10000:]
partial_train_labels=train_labels[10000:]

model.compile(optimizer=keras.optimizers.RMSprop(),loss=keras.losses.BinaryCrossentropy(),metrics=keras.metrics.Accuracy())
history=model.fit(partial_train_data,partial_train_labels,batch_size=100,epochs=20, validation_data=(validation_data,validation_label))

#evaluate the model accuracy
results = model.evaluate(test_data,test_labels)
print(results)

"""print(train_data[0])
print("#"*100)
print(train_labels[0])
print("#"*100)
print(test_data[0])
print("#"*100)
print(test_labels[0])"""
history_dict = history.history
print(history_dict.keys())
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(loss_values) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
