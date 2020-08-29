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
from keras.preprocessing import image as image_utils

train_directory = 'D:/doc/AI/train/'
validation_data_dir='D:/doc/AI/validation/test3/'

#img prepocessing
image_width = 200
image_height = 200

test = ImageDataGenerator(
    rescale=1./255)


validation_generator = test.flow_from_directory(
        validation_data_dir,
        target_size=(image_width, image_height),
        batch_size=1,
        shuffle=False,
        class_mode='binary')


json_file = open('model/pizza_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model/weights.best.hdf5")


# compile loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# loss and accuracy
print("loss and accuracy")
loaded_model.evaluate_generator(validation_generator, steps=800,max_queue_size=10, workers=1)


# needs to be reset each time the generator is called
validation_generator.reset()

loaded_model.summary()

# evaluate the model
# scores = model.evaluate(train_image_generator, validation_generator, verbose=0)
# print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

# prediction
image = 'D:/doc/AI/validation/test1/nopizza/biriyanitest (5).jpg'
image = image_utils.load_img(image, target_size=(150, 150))
image = image_utils.img_to_array(image)*(1./255.)
image = image.reshape((1,) + image.shape)

# loaded_model.predict_classes(image)
value=loaded_model.predict(image)[0][0]
print(value)
# our code
if value > 0.7:
    print("pizza")
else : print("not pizza")
# # our code