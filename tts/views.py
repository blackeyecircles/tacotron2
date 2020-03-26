from django.shortcuts import render
import base64
import json
import logging
import pickle
from queue import Queue
from threading import Thread
import time
# Create your views here.
from django.http import HttpResponse, JsonResponse
from dwebsocket.decorators import accept_websocket
from .services import synthetic_audio
from common.status_message import status_dict

global_text_queue = Queue()
HEARTBEAT = 'heartbeat_info'
EXPIRATION_TIME = 60 * 0.5
SAMPLE_RATE = 22050
CHANNEL = 1
QUANTIZATION_WIDTH = 2

logger = logging.getLogger()


@accept_websocket
def synthetic(request):
    last_heartbeat_time = time.time()
    duration = 0
    if request.is_websocket():
        while not request.websocket.closed and duration <= EXPIRATION_TIME:
            message = request.websocket.read()
            if message:
                try:
                    text = json.loads(message)['text']
                except TypeError as e:
                    logger.error(e)
                    request.websocket.send(json.dumps({'code': 1003, 'message': status_dict[1003]}))
                    continue
                logger.info(text)
                sentinel = object()
                if text == HEARTBEAT:
                    last_heartbeat_time = time.time()
                else:
                    result_queue = Queue()
                    Thread(target=synthetic_audio, args=(text, result_queue, sentinel)).start()
                    while True:
                        try:
                            res = result_queue.get(timeout=20)
                        except Exception as e:
                            print(e)
                            break
                        if res == sentinel:
                            break
                        else:
                            response = {'code': 0,
                                        'message': status_dict[0],
                                        'audio': {"sr": SAMPLE_RATE, "channel": CHANNEL, "width": QUANTIZATION_WIDTH,
                                                  'status': res[0], "data": str(base64.b64encode(res[1].tobytes()), "UTF8")}}
                            request.websocket.send(json.dumps(response))
                    last_heartbeat_time = time.time()
            duration = time.time() - last_heartbeat_time
    # request.websocket.close()
    logger.debug("===================end==============================")
