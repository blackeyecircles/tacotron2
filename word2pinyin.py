import re

from pypinyin import Style, load_phrases_dict
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin

from pycnnum import num2cn

split_pat = re.compile(r'[；、，。？！\s,]\s*')

load_phrases_dict({'百日咳': [['bai3'], ['ri4'], ['ke2']],
                   '一骑绝尘': [['yi1'], ['ji4'], ['jue2'], ['chen2']],
                   '我们': [['wo3'], ['men5']],
                   '你们': [['ni3'], ['men5']],
                   '他们': [['ta1'], ['men5']],
                   '尽': [['jin4']],
                   })


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass


my_pinyin = Pinyin(MyConverter())
lazy_pinyin = my_pinyin.lazy_pinyin


def word2pinyin(text):
    # text = text.replace('的', "的，")
    texts = re.split(split_pat, text)
    results = [" ".join(lazy_pinyin(text, style=Style.TONE3, errors='ignore')) for text in texts if text]
    return results


if __name__ == '__main__':
    # text = "百日咳(pertussis，whoopingcough)是由百日咳杆菌所致的急性呼吸道传染病。其特征为阵发性痉挛性咳嗽，咳嗽末伴有特殊的鸡鸣样吸气吼声。病程较长，可达数周甚至3个月左右，故有百日咳之称。"
    # print(word2pinyin(text))
    num = 32355555
    c = num2cn(num, numbering_type='high', alt_two=False, big=False, traditional=False)
    print(c)
    print(num - int(num))
