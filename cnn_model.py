import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


from keras_vggface.vggface import VGGFace

train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input)

train_generator = \
    train_datagen.flow_from_directory(
'./processed_img',
target_size=(224,224),
color_mode='rgb',
batch_size=32,
class_mode='categorical',
shuffle=True)

train_generator.class_indices.values()
# dict_values([0, 1, 2])
NO_CLASSES = len(train_generator.class_indices.values())


# base_model = VGGFace(include_top=True,
#     model='vgg16',
#     input_shape=(224, 224, 3))
# base_model.summary()

base_model = VGGFace(include_top=False,
model='vgg16',
input_shape=(224, 224, 3))
# base_model.summary()
# print(len(base_model.layers))


x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
preds = Dense(NO_CLASSES, activation='softmax')(x)


model = Model(inputs = base_model.input, outputs = preds)
model.summary()

#excluding first 19 layers from training
for layer in model.layers[:19]:
    layer.trainable = False

# train the rest of the layers - 19 onwards as these are our added
# layers
for layer in model.layers[19:]:
    layer.trainable = True

model.compile(optimizer='Adam',
    loss='categorical_crossentropy',
    metrics=['accuracy'])

model.fit(train_generator,
  batch_size = 1,
  verbose = 1,
  epochs = 20)

# creates a HDF5 file, saving model
model.save(
    'transfer_learning_trained' +
    '_face_cnn_model.h5')

class_dictionary = train_generator.class_indices
class_dictionary = {
    value:key for key, value in class_dictionary.items()
}
print(class_dictionary)
face_label_filename = 'face-labels.pickle'
with open(face_label_filename, 'wb') as f: pickle.dump(class_dictionary, f)