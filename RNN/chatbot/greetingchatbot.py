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
import time
import tensorflow_hub as hub
import speech_recognition as sr



vocal_size=20000
tag=[]
questions=[]
responses={}
"""log dir to be used by tensorflowboard"""
root_logdir = os.path.join(os.curdir,"my_logs")



tokenizer = Tokenizer(num_words=vocal_size,oov_token='<oov>')
le=LabelEncoder()

with open("datasets/Datasets/json/Intent.json") as content:
    data = json.load(content)


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
label_len=len(set(y_train))


"""TO always return unique dir every run"""
def get_run_logdir():
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir,run_id)


""" callback to be used by tensorboard"""
tensorboard_cb= keras.callbacks.TensorBoard(get_run_logdir())







model = keras.Sequential([
    layers.Embedding(vocal_size, 64, input_length=max_len),
    layers.Dropout(0.2),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.Dropout(0.2),
    layers.Bidirectional(layers.LSTM(32)),
    layers.Dropout(0.2),
    layers.Dense(16, activation='relu'),
    layers.Dense(label_len,activation='softmax')
])
model.summary()
model.compile(loss='sparse_categorical_crossentropy',optimizer='adam' ,metrics=['accuracy'])


history = model.fit(x_train,y_train,epochs=400,validation_split=0.1,callbacks=[tensorboard_cb])

def speech_to_text():
    r = sr.Recognizer()
    m = sr.Microphone()
    try:
        print("A moment of silence, please...")
        with m as source: r.adjust_for_ambient_noise(source)
        print("Set minimum energy threshold to {}".format(r.energy_threshold))
        while True:
            print("Say something!")
            with m as source: audio = r.listen(source)
            print("Got it! Now to recognize it...")
            try:
                # recognize speech using Google Speech Recognition
                value = r.recognize_google(audio)

                # we need some special handling here to correctly print unicode characters to standard output
                if str is bytes:  # this version of Python uses bytes for strings (Python 2)
                    print(u"You said {}".format(value).encode("utf-8"))
                    return value.encode("utf-8")
                else:  # this version of Python uses unicode for strings (Python 3+)
                    print("You said {}".format(value))
                    return value
            except sr.UnknownValueError:
                print("Oops! Didn't catch that")
            except sr.RequestError as e:
                print("Uh oh! Couldn't request results from Google Speech Recognition service; {0}".format(e))
    except KeyboardInterrupt:
        pass


print("#I am your helper in this session\n #May be we can start by knowing each other(who developed me,my name ) \n #If you get bored as for a joke\n #To refresh you can ask for a gossip")
while True:
    question=speech_to_text()
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
