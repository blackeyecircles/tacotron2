import base64
import json
from urllib import parse, request
from scipy.io.wavfile import write
import numpy as np
from threading import Thread
import pyaudio

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
    output = base64.b64decode(tmp['data'])
    p = pyaudio.PyAudio()
    stream=p.open(format=p.get_format_from_width(2), channels=1, rate=22050, output=True)
    stream.write(output)
    stream.stop_stream()
    stream.close()
    p.terminate()

    # output = base64.b64decode(tmp['data']).decode(encoding='utf-8')
    # output1 = np.array(list(map(int, output[output.find('[') + 1:output.rfind(']')].split(', '))), dtype=np.int16)
    # write(f'./outputs/client_{i}.wav', 22050, output1)
    # chunk = 2048
    # p=pyaudio.PyAudio()
    # stream=p.open(format=p.get_format_from_width(2), channels=1, rate=22050, output=True)
    # for i in range(output1.size // chunk):
    #     stream.write(output1[i * chunk: (i + 1) * chunk].tobytes())
    # stream.write(output1[(i+1)*chunk:].tobytes())
    # stream.stop_stream()   # 停止数据流
    # stream.close()
    # p.terminate()  # 关闭 PyAudio



for i in range(1):
    text = '我的工作，我的生活，我的人生，尽在于此。' * (i + 1)
    th = Thread(target=run, args=(i, text))
    th.start()

print("end")