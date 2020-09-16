# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo06_lbe.py  标签编码器
"""
import numpy as np
import sklearn.preprocessing as sp

raw_samples = np.array(
    ['audi', 'ford', 'audi', 'toyota', 'ford',
     'bmw', 'ford', 'redflag', 'audi'])
print(raw_samples)
# 训练之前，需要标签编码
lbe = sp.LabelEncoder()
result = lbe.fit_transform(raw_samples)
print(result)

# 假设训练之后得到一组测试样本的结果:
test = [0, 0, 1, 1, 4]
print(lbe.inverse_transform(test))
