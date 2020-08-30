from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
# Create your views here.

from keras.models import load_model
from keras.preprocessing import image
import tensorflow as tf
import json
from tensorflow import Graph
from tensorflow.python.keras.models import model_from_json

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.compat.v1.Session(config=config)
sess.as_default()

img_height, img_width=150,150
with open('./model/pizza_model.json','r') as f:
    labelInfo=f.read()
    f.close()


model_graph = Graph()
with model_graph.as_default():
    tf_session = tf.compat.v1.Session()
    with tf_session.as_default():
        model=load_model('./model/weights.best.hdf5')

# json_file = open('model/pizza_model.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# loaded_model.load_weights("model/weights.best.hdf5")

def home(request):
    context={'a':1}
    return render(request,'index.html',context)

def predictImage(request):
    print (request)
    print (request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    testimage='.'+filePathName
    
    img = image.load_img(testimage, target_size=(img_height, img_width))
    x = image.img_to_array(img)*(1./255.)
    x=x.reshape(1,img_height, img_width,3)
    with model_graph.as_default():
        with tf_session.as_default():
            predi=model.predict(x)

    
    label = ""
    if predi >= 0.05:
        print("pizza")
        label = "Pizza"
    else : 
        print("not pizza")
        label = "Not pizza"

    context={'predictedlabel': label, 'filePathName':filePathName, 'predi': predi}
    return render(request,'index.html',context)

# def viewDataBase(request):
#     import os
#     listOfImages=os.listdir('./media/')
#     listOfImagesPath=['./media/'+i for i in listOfImages]
#     context={'listOfImagesPath':listOfImagesPath}
#     return render(request,'viewDB.html',context) 

# def predictImage(request):
#     print(request)
#     print(request.POST.dict())
#     fileObj=request.FILES['filePath']
#     fs=FileSystemStorage()
#     filePathName=fs.save(fileObj.name,fileObj)
#     filePathName=fs.url(filePathName)
#     context={'filePathName':filePathName}
#     return render(request,'index.html',context)