from django.contrib import admin

from .models import Classifier

class ClassifierAdmin(admin.ModelAdmin):
    list_display = ['image']

    class Meta:
        model = Classifier

admin.site.register(Classifier, ClassifierAdmin)
