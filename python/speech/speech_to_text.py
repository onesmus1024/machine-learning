import speech_recognition as sr

# initialize the recognizer
r = sr.Recognizer()
with sr.AudioFile("./maybe-next-time.wav") as source:
    # listen for the data (load audio to memory)
    audio_data = r.listen(source)
    # recognize (convert from speech to text)
    text = r.recognize_google(audio_data)
    print(text)