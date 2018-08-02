from django import forms

class ClassifierForm(forms.Form):
    '''Image upload form.'''
    image = forms.ImageField()
