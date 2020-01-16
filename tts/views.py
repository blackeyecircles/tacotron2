from django.shortcuts import render
import base64
import json
import pickle
# Create your views here.
from django.http import HttpResponse
from .services import synthetic_audio


def synthetic(request):
    text = request.GET['text']
    print(text)
    res = synthetic_audio(text)
    data = {"data": str(base64.b64encode(res.tobytes()), "UTF8"), "sr": 22050, "channel": 1, "width": 2}
    return HttpResponse(json.dumps(data))