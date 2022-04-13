import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1/255)
train_generator = train_datagen.flow_from_directory(
'/media/onesmus/dev/dev/machine_learning/datasets/Datasets/beautytest/train',
target_size=(300, 300),
class_mode='binary'
)
validate_datagen = ImageDataGenerator(rescale=1/255)
validate_generator = train_datagen.flow_from_directory(
'/media/onesmus/dev/dev/machine_learning/datasets/Datasets/beautytest/valid',
target_size=(300, 300),
class_mode='binary'
)

model = keras.Sequential([ 
    layers.Conv2D(64, (3, 3), activation='relu', input_shape=(300, 300, 3)),
    # layers.MaxPooling2D(2, 2),
    # layers.Conv2D(64, (3, 3), activation='relu'),
    # layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='RMSprop',loss='binary_crossentropy',metrics=['accuracy'])
history = model.fit_generator(train_generator, epochs=5, validation_data=validate_generator)