import tensorflow as tf
from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.models import model_from_json
import  glob, os
from imutils import paths
import pandas as pd
# classification metrics imports
from sklearn.metrics import confusion_matrix, plot_confusion_matrix
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import accuracy_score, classification_report 

#directory
train_directory = 'D:/doc/AI/train'
validation_data_dir='D:/doc/AI/validation/test2'

#img prepocessing
image_width = 150
image_height = 150

train = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.2,
    height_shift_range=0.2
    )
train_image_generator = train.flow_from_directory(
   train_directory,
    target_size=(image_width, image_height ),
    batch_size=32,
    class_mode='binary'
    )

test = ImageDataGenerator(
    rescale=1./255)


validation_generator = test.flow_from_directory(
        validation_data_dir,
        target_size=(image_width, image_height),
        batch_size=1,
        shuffle=False,
        class_mode='binary')


model = Sequential()

model.add(Convolution2D(16, 3, 3, input_shape=(image_width, image_height, 3), padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same' ))

model.add(Convolution2D(32, 3, 3,  padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same' ))

model.add(Convolution2D(64, 3, 3,  padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same' ))

model.add(Convolution2D(128, 3, 3,  padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same'))

model.add(Convolution2D(256, 3, 3,  padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2),strides=(2,2), padding='same'))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(1))

model.add(Activation('sigmoid'))

json_file = open('model/pizza_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)

model.compile(
    loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy']
              )

filepath="model/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(
    x=train_image_generator, epochs=5,
     validation_data=validation_generator, 
    validation_steps=800, 
    callbacks=callbacks_list
)


# # save model to JSON
pizza_model_json = model.to_json()
with open("model/pizza_model.json", "w") as json_file:
    json_file.write(pizza_model_json)

# # save weights to HDF5
model.save("model/model_catdogVplat.h5")

loaded_model.load_weights("model/weights.best.hdf5")
probabilities = loaded_model.predict_generator(generator=validation_generator)

y_true = validation_generator.classes
y_pred = probabilities > 0.5
mat = confusion_matrix(y_true, y_pred)
print(mat)

# # needs to be reset each time the generator is called
# validation_generator.reset()
loaded_model.summary()

print(accuracy_score(validation_generator.classes, y_pred))