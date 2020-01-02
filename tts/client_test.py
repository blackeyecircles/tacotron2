import base64
import json
from urllib import parse, request
from scipy.io.wavfile import write
import numpy as np
from threading import Thread
import pyaudio

def run(i, text, callback=None):
    textmod={'text': text}
    textmod = parse.urlencode(textmod)
    print(textmod)
    #输出内容:user=admin&password=admin
    header_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko'}
    url='http://127.0.0.1:8000/tts/'
    req = request.Request(url='%s%s%s' % (url, '?', textmod), headers=header_dict)
    res = request.urlopen(req)

    res = res.read().decode(encoding='utf-8')
    if callback:
        callback()
    tmp = json.loads(res)
    output = base64.b64decode(tmp['data'])
    p = pyaudio.PyAudio()
    stream=p.open(format=p.get_format_from_width(2), channels=1, rate=22050, output=True)
    stream.write(output)
    stream.stop_stream()
    stream.close()
    p.terminate()



for i in range(1):
    text = '我的工作，我的生活，我的人生，尽在于此。'
    text = '中瑞福宁，人工智能系统。'
    # text = '长城是古代中国在不同时期，为了抵御塞北的游牧民族，而修筑的规模浩大的军事工程。'
    # text = '天花是由天花病毒所致的一种烈性传染病，传染性强，病死率高。临床表现为广泛的皮疹成批出现，依序发展成斑疹，丘疹，疱疹，脓疱疹，伴以严重的病毒血症；脓疱疹结痂、脱痂后，终身留下凹陷性瘢痕。天花病毒属于正痘病毒。正痘病毒属内的病毒在形态，大小，结构，对外界抵抗力，免疫学特性等方面均十分相似，天花与类天花病毒所产生的痘疱较小，边缘完整而突起，痘苗病毒所产生者则较大，边缘不整齐，天花病毒致病力较强，但所致细胞病变较痘苗病毒稍慢，天花，类天花，痘苗病毒能在多种细胞组织培养中增生，天花病毒引起典型天花。'

    th = Thread(target=run, args=(i, text))
    th.start()
#
# print("end")

def run_qa(text):
    textmod={'question': text}
    textmod = parse.urlencode(textmod)
    # print(textmod)
    body = {
        'item': "tiancai"
    }
    #输出内容:user=admin&password=admin
    header_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko'}
    url='http://192.168.2.191:8000/bpress/'
    # url='http://192.168.13.169:8080/ask/'
    tmp = json.dumps(body)
    req = request.Request(url='%s%s%s' % (url, '?', textmod), headers=header_dict, method='POST', data=tmp.encode('utf8'))
    res = request.urlopen(req)

    res = res.read().decode(encoding='utf-8')
    print(res)


# for q in ["天花", "咳嗽", "霍乱"]:
#     print(q)
#     th = Thread(target=run_qa, args=(q,))
#     th.start()
