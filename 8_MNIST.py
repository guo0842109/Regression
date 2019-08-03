#!/usr/bin/python

# -*- coding: utf-8 -*-

"""

author ： duanxxnj@163.com

time : 2016-07-03-18-04

基于L1惩罚和L2惩罚的LR回归模型

C是用于调节目标函数和惩罚项之间关系的

C越小，惩罚力度越大，所得到的w的最优解越趋近于0，或者说参数向量越稀疏

C越大，惩罚力度越小，越能体现模型本身的特征。

本例子是利用逻辑回归做数字分类

每个样本都是一个8x8的图片，将这些样本分为两类：0-4为一类，5-9为一类

"""

print(__doc__)

import numpy as np

import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

from sklearn import datasets

from sklearn.preprocessing import StandardScaler

from sklearn import datasets, linear_model, discriminant_analysis, cross_validation


# 读取图片数据
# 这个数据集中有共有10种数字图片，数字分别是0-9
# 每种类别的数字大概有180张图片，总共的图片数目为1797张
# 每幅图片的尺寸为8x8，即64个像素点
# 像素的变化范围是0-16
# 也就是说，每个样本点有64个特征
# 即本例中LR模型的特征维度为64

digits = datasets.load_digits()

# 将前12张数字图片显示出来

plt.figure(1)

for i in range(12):
    imageplot = plt.subplot(3, 4, i+1)
    plt.imshow(digits.images[i], interpolation='nearest',cmap='binary', vmax=16, vmin=0)
    imageplot.set_title("digit %d" % i)

# 读取数据X和目标y，并将数据X归一化
# 归一化的结果是均值为0，方差为1，也就是常用的z变换
# 对于机器学习中的很多方法，比如SVM，L1、L2正则化的线性模型等等
# 对于数据，都有一个基本假设，就是数据每个特征都是以0为中心
# 并且所有特征的数据变化都在同一个数量级
# 如果有一个特征的方差特别大，那么这个特征就有可能对机器学习的模型起决定性的作用
# 为了防止上面的现象，所以这里将数据做了归一化，或者叫做正则化

X, y = digits.data, digits.target



#X = StandardScaler().fit_transform(X)

# 将数据集分成两类 0-4之间为一类，5-9之间为另一类

y = y

X_train, X_test, y_train, y_test  = \
    cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)

scaler = StandardScaler()
scaler.fit(X_train)
X_train = scaler.transform(X_train)

plt.figure(2)

clf_softmax_LR = LogisticRegression(C=1, penalty='l2',tol=0.01)

clf_softmax_LR.fit(X_train, y_train)

X_test = scaler.transform(X_test)

score = clf_softmax_LR.score(X_test, y_test)
print ('Accuracy:' + str(score))