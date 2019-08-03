# coding:UTF-8

from __future__ import division
from math import exp
import numpy as np
from numpy import *
from random import normalvariate  # 正太分布
from datetime import datetime

trainData = 'diabetes_train.txt'
testData = 'diabetes_test.txt'
featureNum = 8
max_list = []
min_list = []


def normalize(x_list, max_list, min_list):
    norm_list = []
    for index, x in enumerate(x_list):
        x_max = max_list[index]
        x_min = min_list[index]
        if x_max == x_min:
            x = 1.0
        else:
            x = round((x - x_min) / (x_max - x_min), 4)
        norm_list.append(x)
    return norm_list


def load_dataset(data):
    global max_list
    global min_list
    dataMat = []
    labelMat = []

    fr = open(data)  # 打开文件

    for line in fr.readlines():
        currLine = line.strip().split(',')
        # lineArr = [1.0]
        lineArr = []

        for i in range(featureNum):
            lineArr.append(float(currLine[i]))

        dataMat.append(lineArr)

        labelMat.append(float(currLine[-1]) * 2 - 1)

    data_array = np.array(dataMat)
    max_list = np.max(data_array, axis=0)
    min_list = np.min(data_array, axis=0)

    norm_dataMat = []
    for row in dataMat:
        norm_row = normalize(row, max_list, min_list)
        norm_dataMat.append(norm_row)
    return norm_dataMat, labelMat


def sigmoid(inx):
    # return 1. / (1. + exp(-max(min(inx, 15.), -15.)))
    return 1.0 / (1 + exp(-inx))


def stocGradAscent(data_matrix, classLabels, k, iter):
    # dataMatrix用的是mat, classLabels是列表
    m, n = shape(data_matrix)
    alpha = 0.01  # 初始学习率
    # 初始化参数
    # w = random.randn(n, 1)#其中n是特征的个数
    w = zeros((n, 1))
    w_0 = 0.
    v = normalvariate(0, 0.2) * ones((n, k))  # normalvariate(u, s) 生成均值为u方差为s的呈正太分布的值，ones((n,m))生成n行m列的元素均为1的矩阵；

    for it in range(iter):
        print (it)
        for x in range(m):  # 随机优化，对每一个样本而言的
            inter_1 = data_matrix[x] * v
            inter_2 = multiply(data_matrix[x], data_matrix[x]) * multiply(v, v)  # multiply对应元素相乘
            # 完成交叉项
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.

            p = w_0 + data_matrix[x] * w + interaction  # 计算预测的输出
            # print "y: ",p
            loss = sigmoid(classLabels[x] * p[0, 0]) - 1
            # print "loss: ",loss

            w_0 = w_0 - alpha * loss * classLabels[x]

            for i in range(n):  #维度遍历
                if data_matrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * data_matrix[x, i]
                    for j in range(k):
                        v[i, j] = v[i, j] - alpha * loss * classLabels[x] * (
                                    data_matrix[x, i] * inter_1[0, j] - v[i, j] * data_matrix[x, i] * data_matrix[x, i])

    return w_0, w, v


# 预测
def predict(x, w_0, w, v):
    inter_1 = x * v
    inter_2 = multiply(x, x) * multiply(v, v)  # multiply对应元素相乘
    # 完成交叉项
    interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
    p = w_0 + x * w + interaction  # 计算预测的输出

    if p < 0.5:
        result = 1.
    else:
        result = -1.

    return result


# 获取正确率
def get_accuracy(data_matrix, classLabels, w_0, w, v):
    m, n = shape(data_matrix)
    allItem = 0
    error = 0
    result = []
    for x in range(m):
        allItem += 1
        inter_1 = data_matrix[x] * v
        inter_2 = multiply(data_matrix[x], data_matrix[x]) * multiply(v, v)  # multiply对应元素相乘
        # 完成交叉项
        interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + data_matrix[x] * w + interaction  # 计算预测的输出

        pre = sigmoid(p[0, 0])

        result.append(pre)

        if pre < 0.5 and classLabels[x] == 1.0:
            error += 1
        elif pre >= 0.5 and classLabels[x] == -1.0:
            error += 1
        else:
            continue

    print (result)

    return float(error) / allItem


if __name__ == '__main__':
    dataTrain, labelTrain = load_dataset(trainData)
    dataTest, labelTest = load_dataset(testData)
    date_startTrain = datetime.now()

    # 训练
    print ("开始训练")
    w_0, w, v = stocGradAscent(mat(dataTrain), labelTrain, k=20, iter=200)

    # 预测
    for line in mat(dataTrain):
        print ("预测结果为: %f" % (predict(line, w_0, w, v)))
    print ("训练准确性为：%f" % (1 - get_accuracy(mat(dataTrain), labelTrain, w_0, w, v)))
    date_endTrain = datetime.now()
    print ("训练时间为：%s" % (date_endTrain - date_startTrain))

    # 测试
    print ("开始测试")
    print ("测试准确性为：%f" % (1 - get_accuracy(mat(dataTest), labelTest, w_0, w, v)))