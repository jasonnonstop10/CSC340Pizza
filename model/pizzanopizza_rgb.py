from keras.preprocessing import image as image_utils
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.callbacks import ModelCheckpoint
import numpy as np
from keras.models import model_from_json
import  glob
from imutils import paths
# import cv2

#print("Hello Milk")

#model
train_directory = 'D:/doc/AI/validation/test1/'
validation_data_dir='D:/doc/AI/validation/test4/'


# image_paths = list(paths.list_images(validation_data_dir))
# labels = []
# #our code
# for img in image_paths:
#   if img.split('/')[-2] == "pizza":
#     labels.append('1')
#   else: labels.append('0')
#   #our code
# labels = np.array(labels)


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
#can use
# four convolutional & pooling layers
model.add(Convolution2D(32, 3, 3, input_shape=(image_width, image_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))


# two fully-connected layers
model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dropout(0.5))
model.add(Dense(1))
# sigmoid activation - good for a binary classification
model.add(Activation('sigmoid'))

#try
# four convolutional & pooling layers
# model.add(Convolution2D(32, 3, 3, input_shape=(image_width, image_height, 3)))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# model.add(Convolution2D(64, 3, 3))
# model.add(Activation('relu'))
# model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

# # two fully-connected layers
# model.add(Flatten())
# model.add(Dense(64))
# model.add(Activation('relu'))

# model.add(Dropout(0.5))
# model.add(Dense(1))
# # sigmoid activation - good for a binary classification
# model.add(Activation('sigmoid'))
#----------------------------------------------
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy']
              )
filepath="model/weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

# pizza_model = model.fit_generator(

#         train_image_generator,
        
#         # number of training samples
#        epochs=100,
    
#         # nb_epoch=100,
#         validation_data=validation_generator,
#         # # number of training samples
#         # nb_val_samples=800,
#         validation_steps=800,
#         # lets me save the best models weights
#         callbacks=callbacks_list
# )


# save model to JSON
pizza_model_json = model.to_json()
with open("model/pizza_model.json", "w") as json_file:
    json_file.write(pizza_model_json)

# save weights to HDF5
model.save_weights("model/pizza_model.h5")

json_file = open('model/pizza_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("model/weights.best.hdf5")
loaded_model.load_weights("model/pizza_model.h5")


# compile loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# loss and accuracy

loaded_model.evaluate(validation_generator, steps=800,max_queue_size=10, workers=1)


# needs to be reset each time the generator is called
validation_generator.reset()
# print("evaluate = " + evaluate)
loaded_model.summary()

# evaluate the model
# scores = model.evaluate(train_image_generator, validation_generator, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# prediction
image = 'D:/doc/AI/validation/test2/pizza/pizza (4217).jpg'
image = image_utils.load_img(image, target_size=(150, 150))
image = image_utils.img_to_array(image)*(1./255.)
image = image.reshape((1,) + image.shape)

# loaded_model.predict_classes(image)
value=loaded_model.predict(image)[0][0]
print(value)
# our code
if value >= 0.5:
    print("pizza")
else : print("not pizza")
# # our code