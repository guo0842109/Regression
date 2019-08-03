#!/usr/bin/python
# -*- coding: utf-8 -*-


"""
author ： khuang0430@126.com
time : 2016-06-07_13-47

基于CV/AIC/BIC的 Lasso模型选择

CV（cross-validation）交叉验证
AIC（Akaike information criterion）赤池信息准则
BIC（Bayes Information criterion）贝叶斯信息准则

这里AIC和BIC信息准则使用的是LassoLarsIC实现的，使用的是LARS算法

基于信息准则的模型选择的速度是非常的快的
但是，其依赖于模型是正确的基本假设，且对模型自由度需要有恰当的估计，
这样才能在大量的样本上得到一个渐进的结果。
当特征的数量远大于样本的数量的时候，信息准则的模型选择效果并不理想

对于交叉验证而言，基于坐标轴下降算法的交叉验证可以使用LassoCV
基于LARS算法的交叉验证可以使用LassoLarsCV
在实际使用中，这两种算法仅仅是在速度上存在一定的差异，其结果几乎差不多

由于参数的选择对未知的数据可能不是最优的
所以在评价一个使用交叉验证得到的参数的方法的时候
嵌套交叉验证是有必要的

"""
print(__doc__)

import time
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoCV, LassoLarsCV, LassoLarsIC
from sklearn import datasets

# 加载数据集
# 该数据集有442个样本，特征的维度为10
diabetes = datasets.load_diabetes()
X = diabetes.data
y = diabetes.target
print (X.shape)

# 选择随机种子
rng = np.random.RandomState(int(time.time())%100)
# 添加14个噪声的特征
X = np.c_[X, rng.randn(X.shape[0], 14)]

# 对每个特征做数据归一化，这个过程是LARS算法需要的
X /= np.sqrt(np.sum(X ** 2, axis=0))


# LassoLarsIC: 使用LARS算法做BIC/AIC信息准则
model_bic = LassoLarsIC(criterion='bic')
t1 = time.time()
model_bic.fit(X, y)
t_bic = time.time() - t1
alpha_bic_ = model_bic.alpha_

model_aic = LassoLarsIC(criterion='aic')
model_aic.fit(X, y)
alpha_aic_ = model_aic.alpha_

# 这里alpha_是最终选择的参数
# alphas_是所有的alpha选择
# alpha_就是alphas_中，对于的信息准则最小的那个值
# criterion_是和alphas_对应的信息准则的结果
def plot_ic_criterion(model, name, color):
    alpha_ = model.alpha_
    alphas_ = model.alphas_
    criterion_ = model.criterion_
    plt.plot(-np.log10(alphas_), criterion_, '--', color=color,
             linewidth=3, label='%s criterion' % name)
    plt.axvline(-np.log10(alpha_), color=color, linewidth=3,
                label='alpha: %s estimate' % name)
    plt.xlabel('-log(alpha)')
    plt.ylabel('criterion')

plt.figure()
plot_ic_criterion(model_aic, 'AIC', 'b')
plot_ic_criterion(model_bic, 'BIC', 'r')
plt.legend()
plt.title('Information-criterion for model selection (training time %.3fs)'
          % t_bic)


# LassoCV: 基于坐标轴下降法的Lasso交叉验证
print("使用坐标轴下降法计算参数正则化路径:")
t1 = time.time()
# 这里是用20折的交叉验证
model = LassoCV(cv=20).fit(X, y)
t_lasso_cv = time.time() - t1

# 最终alpha的结果，因为有的alpha实在是太小了
# 所以使用负对数形式表示
m_log_alphas = -np.log10(model.alphas_)


# 由于这里使用的是20折交叉验证
# 所以model.mse_path_有20列
# model.mse_path_中每一列，是对应交叉验证，在alpha选择不同值的时候
# 其对应的均方误差（mean square error）
# 模型最终选择的alpha是所有交叉验证结果的平均值中
# 最小的那个平均的均方误差对应的alpha
plt.figure()
ymin, ymax = 2300, 3800
plt.plot(m_log_alphas, model.mse_path_, ':')
plt.plot(m_log_alphas, model.mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha: CV estimate')

plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: coordinate descent '
          '(train time: %.2fs)' % t_lasso_cv)
plt.axis('tight')
plt.ylim(ymin, ymax)

# LassoLarsCV: 基于LARS算法的交叉验证

print("使用LARS算法计算参数正则化路径:")
t1 = time.time()
model = LassoLarsCV(cv=20).fit(X, y)
t_lasso_lars_cv = time.time() - t1

# 最终alpha的结果，因为有的alpha实在是太小了
# 所以使用负对数形式表示
m_log_alphas = -np.log10(model.cv_alphas_)

# 参数说明和上面是一样的
plt.figure()
plt.plot(m_log_alphas, model.cv_mse_path_, ':')
plt.plot(m_log_alphas, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()

plt.xlabel('-log(alpha)')
plt.ylabel('Mean square error')
plt.title('Mean square error on each fold: Lars (train time: %.2fs)'
          % t_lasso_lars_cv)
plt.axis('tight')
plt.ylim(ymin, ymax)

plt.show()