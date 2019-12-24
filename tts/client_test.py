import base64
import json
from urllib import parse, request
from scipy.io.wavfile import write
import numpy as np
from threading import Thread


def run(i, text):
    textmod={'text': text}
    textmod = parse.urlencode(textmod)
    print(textmod)
    #输出内容:user=admin&password=admin
    header_dict = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko'}
    url='http://127.0.0.1:8000/tts/'
    req = request.Request(url='%s%s%s' % (url, '?', textmod), headers=header_dict)
    res = request.urlopen(req)

    res = res.read().decode(encoding='utf-8')
    tmp = json.loads(res)
    output = base64.b64decode(tmp['data']).decode(encoding='utf-8')
    output1 = list(map(int, output[output.find('[') + 1:output.rfind(']')].split(', ')))
    write(f'./outputs/client_{i}.wav', 22050, np.array(output1, dtype=np.int16))


for i in range(1):
    text = '我的工作，我的生活，我的人生，尽在于此' * (i + 1)
    th = Thread(target=run, args=(i, text))
    th.start()

print("end")