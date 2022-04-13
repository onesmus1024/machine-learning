import tensorflow as tf
import tensorflow_io as tfio
import matplotlib.pyplot as plt
from IPython.display import Audio


audio = tfio.audio.AudioIOTensor('chatbot.mp3')

audio_slice = audio[100:]

# remove last dimension
audio_tensor = tf.squeeze(audio_slice, axis=[-1])

tensor = tf.cast(audio_tensor, tf.float32) / 32768.0


plt.figure()
plt.plot(tensor.numpy())
plt.show()

print(audio_tensor)
print(audio)
Audio(audio_tensor.numpy(), rate=audio.rate.numpy())