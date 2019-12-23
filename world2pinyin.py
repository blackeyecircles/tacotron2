import re

import jieba
from pypinyin import Style, load_phrases_dict
from pypinyin.contrib.neutral_tone import NeutralToneWith5Mixin
from pypinyin.converter import DefaultConverter
from pypinyin.core import Pinyin

split_pat = re.compile(r'[；、，。？！\s]\s*')

load_phrases_dict({'百日咳': [['bai3'], ['ri4'], ['ke2']],
                   '一骑绝尘': [['yi1'], ['ji4'], ['jue2'], ['chen2']],
                   '他们': [['ta1'], ['men5']]
                   })


class MyConverter(NeutralToneWith5Mixin, DefaultConverter):
    pass

my_pinyin = Pinyin(MyConverter())
# pinyin = my_pinyin.pinyin
lazy_pinyin = my_pinyin.lazy_pinyin

text = "百日咳(pertussis，whoopingcough)是由百日咳杆菌所致的急性呼吸道传染病。其特征为阵发性痉挛性咳嗽，咳嗽末伴有特殊的鸡鸣样吸气吼声。病程较长，可达数周甚至3个月左右，故有百日咳之称。"
text = text.replace('的', "的，")
# for word in jieba.cut(text):
#     print(" ".join(lazy_pinyin(word, style=Style.TONE3, errors='ignore')))
texts = re.split(split_pat, text)
results = [" ".join(lazy_pinyin(text, style=Style.TONE3, errors='ignore')) for text in texts if text]
for res in results:
    print(res)

#  新的结果中使用 ``5`` 标识轻声
# print(lazy_pinyin('好了', style=Style.TONE2))
# print(" ".join(lazy_pinyin('长城是古代中国在不同时期，为抵御塞北游牧部落侵袭而修筑的规模浩大的军事工程。', style=Style.TONE3)))
# print(" ".join(lazy_pinyin('他们一骑绝尘地离开，我们就好这门，为人处世，骑行，', style=Style.TONE3)))
# print(" ".join(lazy_pinyin('白皮书说党的十八大以来中国的核安全事业进入安全高效发展的新时期', style=Style.TONE3)))
# print(" ".join(lazy_pinyin('在核安全观引领下中国逐步构建起法律规范行政监管行业自律技术保障人才支撑文化引领社会参与国际合作等为主体的核安全治理体系核安全防线更加牢固', style=Style.TONE3)))
