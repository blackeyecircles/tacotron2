import websocket
from urllib import parse
import json
import base64
import pyaudio


def on_message(ws, message):
    global stream
    print('on_message')
    print(ws)
    # res = message.read().decode(encoding='utf-8')
    tmp = json.loads(message)
    print(tmp)
    output = base64.b64decode(tmp['audio']['data'])
    stream.write(output)


def on_error(ws, error):
    print('on_error')
    print(ws)
    print(error)


def on_open(ws):
    import time
    print('on_open')
    text1 = '中瑞福宁，人工智能系统。'
    text2 = '特征为阵发性痉挛性咳嗽，咳嗽末伴有特殊的鸡鸣样吸气吼声。'
    text2 = "感冒熟悉一下：感冒总体上分为普通感冒和流行性感冒，" + "在这里先讨论普通感冒。普通感冒，祖国医学称\"伤风\"，" + \
            "是由多种病毒引起的一种呼吸道常见病，其中30%-50%是由某种血清型的鼻病毒引起，" + \
            "普通感冒虽多发于初冬，但任何季节，如春天，夏天也可发生，不同季节的感冒的致病病毒并非完全一样。" + \
            "流行性感冒，是由流感病毒引起的急性呼吸道传染病。" + \
            "病毒存在于病人的呼吸道中，在病人咳嗽，打喷嚏时经飞沫传染给别人。" + \
            "流感的传染性很强，由于这种病毒容易变异，即使是患过流感的人，当下次再遇上流感流行，他仍然会感染，所以流感容易引起暴发性流行。" + \
            "一般在冬春季流行的机会较多，每次可能有20～40%的人会传染上流感。"
    d = json.dumps({'text': text1})
    # ws.send(d)

    d = json.dumps({'text': text2})
    ws.send(d)

    time.sleep(2)
    ws.send(json.dumps({'text': 'heartbeat_info'}))
    print('on_open_end')
    # time.sleep(10)
    # ws.close()


def on_close(ws):
    print("### closed ###")
    # print(ws)

p = pyaudio.PyAudio()
stream = p.open(format=p.get_format_from_width(2), channels=1, rate=22050, output=True)

text = '百日咳是由百日咳杆菌所致的急性呼吸道传染病。其特征为阵发性痉挛性咳嗽，咳嗽末伴有特殊的鸡鸣样吸气吼声。'
text = '特征为阵发性痉挛性咳嗽，咳嗽末伴有特殊的鸡鸣样吸气吼声。'
# text = "感冒熟悉一下：感冒总体上分为普通感冒和流行性感冒，" + "在这里先讨论普通感冒。普通感冒，祖国医学称\"伤风\"，" + \
#         "是由多种病毒引起的一种呼吸道常见病，其中30%-50%是由某种血清型的鼻病毒引起，" + \
#         "普通感冒虽多发于初冬，但任何季节，如春天，夏天也可发生，不同季节的感冒的致病病毒并非完全一样。" + \
#         "流行性感冒，是由流感病毒引起的急性呼吸道传染病。" + \
#         "病毒存在于病人的呼吸道中，在病人咳嗽，打喷嚏时经飞沫传染给别人。" + \
#         "流感的传染性很强，由于这种病毒容易变异，即使是患过流感的人，当下次再遇上流感流行，他仍然会感染，所以流感容易引起暴发性流行。" + \
#         "一般在冬春季流行的机会较多，每次可能有20～40%的人会传染上流感。"
textmod = {'text': text}
websocket.enableTrace(True)
ws = websocket.WebSocketApp("ws://127.0.0.1:8000/tts/",
                            on_open=on_open,
                            on_message=on_message,
                            on_error=on_error,
                            on_close=on_close)

ws.run_forever()
stream.stop_stream()
stream.close()
p.terminate()
