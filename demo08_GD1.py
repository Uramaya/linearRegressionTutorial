# -*- coding: utf-8 -*-
from __future__ import unicode_literals
"""
demo08_GD.py   梯度下降实现线性回归
"""
import numpy as np
import matplotlib.pyplot as mp

train_x = np.array([0.5, 0.6, 0.8, 1.1, 1.4])
train_y = np.array([5.0, 5.5, 6.0, 6.8, 7.0])

B, A = 1, 1
times = 1000
lrate = 0.01
for i in range(1, times + 1):
    # 求损失函数关于B与A的偏导数，从而更新模型参数
    Loss_B = (B + A * train_x - train_y).sum()
    Loss_A = (train_x * (B + A * train_x - train_y)).sum()
    # 根据梯度下降公式，更新B与A
    B = B - lrate * Loss_B
    A = A - lrate * Loss_A
print('B:', B)
print('A:', A)
# 通过B与A模型参数，绘制回归线
linex = np.linspace(
    train_x.min(), train_x.max(), 100)
liney = A * linex + B
# 画图
mp.figure('Linear Regression', facecolor='lightgray')
mp.title('Linear Regression', fontsize=18)
mp.grid(linestyle=':')
mp.scatter(train_x, train_y, s=80, marker='o',
           color='dodgerblue', label='Samples')
mp.plot(linex, liney, color='orangered',
        linewidth=2, label='Regression Line')
mp.legend()
mp.show()
