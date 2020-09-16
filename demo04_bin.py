# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo04_bin.py 二值化
"""
import scipy.misc as sm
import numpy as np
import matplotlib.pyplot as mp
import sklearn.preprocessing as sp

ary = np.array([[10, 21, 5],
                [2, 4, 1],
                [11, 18, 18]])

# 二值化
bin = sp.Binarizer(threshold=10)
result = bin.transform(ary)
print(result)

# 二值化图片
lily = sm.imread('../da_data/lily.jpg', True)
bin = sp.Binarizer(threshold=167)
result = bin.transform(lily)
mp.imshow(result, cmap='gray')
mp.show()
