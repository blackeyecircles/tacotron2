import re

from pypinyin import Style, load_phrases_dict
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin

from pycnnum import num2cn
import jieba
import numpy as np

split_pat = re.compile(r'[：；、，。？！\s,]\s*')
num_pat = re.compile(r"\d+\.?\d*")

# 将单位符号转为汉字, 需要不断扩充维护
unit_dict = {'km': '千米',
             'cm': '厘米',
             'mm': '毫米',
             'μm': '微米',
             'nm': '纳米',
             'm': '米',
             'km2': '平方千米',
             'c㎡': '平方厘米',
             '㎡': '平方米',
             'km³': '立方千米',
             'cm³': '立方厘米',
             'm³': '立方米',
             't': '吨',
             'kg': '千克',
             'mg': '毫克',
             'g': '克',
             '℃': "摄氏度",
             }
# 假定单位与数字一起使用
units = r'\d{1}' + r'|\d{1}'.join(unit_dict.keys())
unit_pat = re.compile(fr'{units}')

# 纠正一些错误读音，包括多音字。需不断扩充维护
load_phrases_dict({'百日咳': [['bai3'], ['ri4'], ['ke2']],
                   '一骑绝尘': [['yi1'], ['ji4'], ['jue2'], ['chen2']],
                   '我们': [['wo3'], ['men5']],
                   '你们': [['ni3'], ['men5']],
                   '他们': [['ta1'], ['men5']],
                   '尽': [['jin4']],
                   })

# 分词后，不希望被单独放在下一句句首的字
single_word_list = ['的']


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


my_pinyin = Pinyin(MyConverter())
lazy_pinyin = my_pinyin.lazy_pinyin


def num2word(text):
    nums = re.findall(num_pat, text)

    for num in nums:
        try:
            if '.' in num:
                if int(float(num)) != 0:
                    int_part = num2cn(int(float(num)), numbering_type='high', alt_two=False, big=False, traditional=False)
                    float_part = num2cn(float('0' + num[num.find('.'):]), numbering_type='high', alt_two=False, big=False, traditional=False)
                    word = int_part + float_part[1:]
                else:
                    word = num2cn(float(num), numbering_type='high', alt_two=False, big=False, traditional=False)
            else:
                word = num2cn(int(num), numbering_type='high', alt_two=False, big=False, traditional=False)
            text = text.replace(num, word)
        except:
            print("Fail to process " + num)

    return text


def unit2word(result):
    # 获取匹配到的索引
    index = result.regs[0]
    return result.string[index[0]] + unit_dict[result.string[index[0] + 1: index[1]]]


def word_segment(texts):
    # 分词后，n个词连一起，为了优化语音效果
    n = 3
    text_list = []
    for text in texts:
        words = jieba.lcut(text)
        # words1 = words.reshape((n, -1))
        length = len(words)
        i = 0
        connect = ''
        first = False
        new_para = True
        for word in words:
            # 避免类似于‘的’的文字被分在句首
            if first and word in single_word_list:
                text_list[-1] = text_list[-1] + word
                first = False
                continue
            i += 1
            connect += word
            if i % n == 0:
                text_list.append(connect)
                connect = ''
                first = True
                new_para = False
            elif i == length:
                if new_para:
                    text_list.append(connect)
                else:
                    if text_list:
                        text_list[-1] = text_list[-1] + connect
                    else:
                        text_list.append(connect)
            else:
                first = False

    return text_list


def word2pinyin(text):
    text = re.sub(unit_pat, unit2word, text)
    text = num2word(text)
    # print(text)
    # 在‘的’后面断句，缩短单个句子
    # text = text.replace('的', "的，")
    texts = re.split(split_pat, text)
    # texts = word_segment(texts)
    results = [" ".join(lazy_pinyin(text, style=Style.TONE3, errors='ignore')) for text in texts if text]
    # for r in results:
    #     print(r)
    return filter(lambda s: s and s.strip(), results)


if __name__ == '__main__':
    text = "中瑞福宁人工智能1mm系统。0.55kg百日咳2tkm(pertussiswhoopingcough)是由百日咳杆菌所致的急性呼吸道传染病。其特征为阵发性痉挛性咳嗽，咳嗽末伴有特殊的鸡鸣样吸气吼声。病程较长，可达数周甚至3个月左右，故有百日咳之称。"
    # text = '中瑞福宁人工智能系统。'
    print('\n'.join(word2pinyin(text)))
    # print(re.sub(unit_pat, unit2word, text))
    # num = 0
    # c = num2cn(num, numbering_type='high', alt_two=False, big=False, traditional=False)
    # print(c)
    # print(num - int(num))
    # import jieba
    # tmp = jieba.cut('工信处女干事每月经过下属科室都要亲口交代24口交换机等技术性器件的安装工作')
    # for t in tmp:
    #     print(t)
