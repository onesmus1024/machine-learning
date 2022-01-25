import random
import json
from playsound import playsound
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
from sklearn.preprocessing import LabelEncoder
from gtts import gTTS
import os

vocal_size=2000
tokenizer = Tokenizer(num_words=vocal_size,oov_token='<oov>')
le=LabelEncoder()

with open("datasets/Datasets/json/Intent.json") as content:
    data = json.load(content)

tag=[]
questions=[]
responses={}

for intent in data['intents']:
    responses[intent['intent']]=intent['responses']
    for question in intent['text']:
        tag.append(intent['intent'])
        questions.append(question)
    
df = pd.DataFrame({'intent':tag,'question':questions})

y_train=df['intent']
x_train=df['question']
y_train=le.fit_transform(y_train)
tokenizer.fit_on_texts(df['question'])
x_train=tokenizer.texts_to_sequences(x_train)
x_train=sequence.pad_sequences(x_train,maxlen=10)
max_len=max([len(k) for k in x_train])
print(max_len)
label_len=len(set(y_train))
print(label_len)


model = keras.Sequential([
    layers.Input(max_len),
    layers.Embedding(vocal_size,10),
    layers.Bidirectional(layers.LSTM(64,return_sequences=True)),
    layers.Bidirectional(layers.LSTM(64)),
    layers.Flatten(),
    layers.Dense(label_len,activation='softmax')
])

model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=400)

model.save("/media/onesmus/dev/dev/machine_learning/savedmodel/chatBotModel")

# converter=tf.lite.TFLiteConverter.from_saved_model("/media/onesmus/dev/dev/machine_learning/savedmodel/chatBotModel")
# tflitemodel=converter.convert()

# with open("/media/onesmus/dev/dev/machine_learning/tflitemodel/chatBotmodelLite.tflite",'wb') as f:
#     f.write(tflitemodel)

# plt.plot(history.history['loss'],label='loss')
# plt.plot(history.history['accuracy'],label='accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('loss')
# plt.legend()
# plt.show()
print("#I am your helper in this session\n #May we can start by knowing each other(who developed me,my name ) \n #If you get bored as for a joke\n #To refresh you can ask for a gossip")
while True:
    question=input("enter message>>")
    predict=[]
    predict.append(question)
    predict=tokenizer.texts_to_sequences(predict)
    predict=sequence.pad_sequences(predict,maxlen=max_len)
    predicted=model.predict(predict)
    predicted =predicted.argmax()
    tag = le.inverse_transform([predicted])[0]
    if tag =="GoodBye":
        mytext=random.choice(responses[tag])
        language = 'en'
        myobj = gTTS(text=mytext, lang=language, slow=False)
        myobj.save("chatbot.mp3")
        playsound("chatbot.mp3")
        print('bot reply>>',random.choice(responses[tag]))
        break
    mytext=random.choice(responses[tag])
    print("bot reply>>",mytext)
    language = 'en'
    try:
        myobj = gTTS(text=mytext, lang=language, slow=False)
        myobj.save("chatbot.mp3")
        playsound("chatbot.mp3")
    except :
        print("<unable to say it due to some errors>")
