from django.db import models
import os

def path_and_rename(instance, filename):
    upload_to = 'images'
    ext = filename.split('.')[-1]
    filename = '{}.{}'.format('dish', ext)
    return os.path.join(upload_to, filename)

class Classifier(models.Model):
    image = models.ImageField(upload_to=path_and_rename, default='images/None/no-img.jpg')
