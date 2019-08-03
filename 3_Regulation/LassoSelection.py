#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
author ： khuang0430@126.com
time : 2016-06-06_16-39

基于lasso的特征选择
这个功能一般和其他的分类器一起使用
或直接内置于其他分类器算中

"""
import numpy as np
import time

from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score

np.random.seed(int(time.time()))

# 导入iris数据集
# 这个数据集一共有150个样本，特征维数为4维
iris = load_iris()
X, y = iris.data, iris.target
print ('生成矩阵的尺寸：150, 4')
print (X.shape)

# 对原始样本重排列
inds = np.arange(X.shape[0])
np.random.shuffle(inds)

# 提取训练数据集和测试数据集
X_train = X[inds[:100]]
y_train = y[inds[:100]]
X_test = X[inds[100:]]
y_test = y[inds[100:]]

print ('原始特征的维度：', X_train.shape[1])

# 线性核的支持向量机分类器（Linear kernel Support Vector Machine classifier）
# 支持向量机的参数C为0.01,使用l1正则化项
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train, y_train)

print ('原始特征，在测试集上的准确率：', lsvc.score(X_test, y_test))
print ('原始特征，在测试集上的R2可决系数：', r2_score(lsvc.predict(X_test), y_test))

# 基于l1正则化的特征选择
model = SelectFromModel(lsvc, prefit=True)

# 将原始特征，转换为新的特征
X_train_new = model.transform(X_train)
X_test_new = model.transform(X_test)


print ('新特征的维度：', X_train_new.shape[1])
# 用新的特征重新训练模型
lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X_train_new, y_train)

print ('新特征，在测试集上的准确率：', lsvc.score(X_test_new, y_test))
print ('新始特征，在测试集上的R2可决系数：', r2_score(lsvc.predict(X_test_new), y_test))