import os
import numpy as np
from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.conf import settings

from .forms import ClassifierForm
from .models import Classifier
from .predict import inference

# Path to input image
media_path = os.path.join(os.path.dirname(settings.BASE_DIR), 'media_cdn/images')

# Class names
classes = ['Fifty', 'Five Thousand', 'Hundred', 'Ten', 'Thousand', 'Twenty']


# Model paths
model_dir = os.path.join(os.path.dirname(settings.BASE_DIR), 
                         'experiments/regularization/L2/more_features'
                         )

print('Loading model from disk')

def upload_img(request):

    form = ClassifierForm(request.POST or None, request.FILES or None)
    if form.is_valid():
        m = Classifier()
        m.image = form.cleaned_data['image']
        print(type(form.cleaned_data['image']))
        m.save()

        # result = feedfowrd()
        return HttpResponseRedirect('/predict')

    context = {
        "form": form,
    }
    return render(request, 'indian_food.html', context)

def predict(request):

    filename = os.path.join(media_path, os.listdir(media_path)[0])
    label, prob = inference(model_dir, filename)
    prob = list(np.round(prob, 3)[0])
    class_prob = list(zip(classes, prob))

    print(class_prob)
    context = {
        'bill_name': classes[int(label)],
        'classes': class_prob,
    }

    print(context)

    return render(request, 'result.html', context)

def clean_up(request):
    # Delete image instance from model
    Classifier.objects.all().delete()

    # Delete image from media directory
    for img in os.listdir(media_path):
        os.remove(os.path.join(media_path, img))

    return HttpResponseRedirect('/')
