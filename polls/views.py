from django.shortcuts import render
from django.http import HttpResponse
from django.core.files.storage import FileSystemStorage
# Create your views here.
def home(request):
    return render(request,'index.html')

def predictImage(request):
    print(request)
    print(request.POST.dict())
    fileObj=request.FILES['filePath']
    fs=FileSystemStorage()
    filePathName=fs.save(fileObj.name,fileObj)
    filePathName=fs.url(filePathName)
    context={'filePathName':filePathName}
    return render(request,'index.html',context)