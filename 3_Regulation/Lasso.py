import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model, discriminant_analysis, cross_validation

def load_data():
    diabetes = datasets.load_diabetes()
    return cross_validation.train_test_split(diabetes.data, diabetes.target, test_size=0.25, random_state=0)

def t_lasso(*data):
    X_train, X_test, y_train, y_test = data
    lassoRegression = linear_model.Lasso()
    lassoRegression.fit(X_train, y_train)
    print("权重向量:%s, b的值为:%.2f" % (lassoRegression.coef_, lassoRegression.intercept_))
    print("损失函数的值:%.2f" % np.mean((lassoRegression.predict(X_test) - y_test) ** 2))
    print("预测性能得分: %.2f" % lassoRegression.score(X_test, y_test))

#测试不同的α值对预测性能的影响
def t_lasso_alpha(*data):
    X_train, X_test, y_train, y_test = data
    alphas = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
    scores = []
    for i, alpha in enumerate(alphas):
        lassoRegression = linear_model.Lasso(alpha=alpha)
        lassoRegression.fit(X_train, y_train)
        scores.append(lassoRegression.score(X_test, y_test))
    return alphas, scores

def show_plot(alphas, scores):
    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)
    ax.plot(alphas, scores)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"score")
    ax.set_xscale("log")
    ax.set_title("Lasso")
    plt.show()

if __name__=='__main__':
    X_train, X_test, y_train, y_test = load_data()
    # 使用默认的alpha
    #t_lasso(X_train, X_test, y_train, y_test)
    # 使用自己设置的alpha
    alphas, scores = t_lasso_alpha(X_train, X_test, y_train, y_test)
    show_plot(alphas, scores)

'''
alpha : float, 可选，默认 1.0。当 alpha 为 0 时算法等同于普通最小二乘法，可通过 Linear Regression 实现，因此不建议将 alpha 设为 0.

fit_intercept : boolean 
是否进行拦截计算（intercept）。若 false，则不计算（比如数据已经经过集中了）。此处不太明白，仿佛与偏度有关。

normalize : boolean, 可选, 默认 False 
若 True，则先 normalize 再 regression。若 fit_intercept 为 false 则忽略此参数。当 regressors 被 normalize 的时候，需要注意超参（hyperparameters）的学习会更稳定，几乎独立于 sample。对于标准化的数据，就不会有此种情况。如果需要标准化数据，请对数据预处理。然后在学习时设置 normalize=False。

copy_X : boolean, 可选, 默认 True 
若 True，则会复制 X；否则可能会被覆盖。

precompute : True | False | array-like, 默认=False 
是否使用预计算的 Gram 矩阵来加速计算。如果设置为 ‘auto’ 则机器决定。Gram 矩阵也可以 pass。对于 sparse input 这个选项永远为 True。

max_iter : int, 可选 
最大循环次数。

tol : float, 可选 
优化容忍度 The tolerance for the optimization: 若更新后小于 tol，优化代码检查优化的 dual gap 并继续直到小于 tol 为止。

warm_start : bool, 可选 
为 True 时, 重复使用上一次学习作为初始化，否则直接清除上次方案。

positive : bool, 可选 
设为 True 时，强制使系数为正。

selection : str, 默认 ‘cyclic’ 
若设为 ‘random’, 每次循环会随机更新参数，而按照默认设置则会依次更新。设为随机通常会极大地加速交点（convergence）的产生，尤其是 tol 比 1e-4 大的情况下。

random_state : int, RandomState instance, 或者 None (默认值) 
pseudo random number generator 用来产生随机 feature 进行更新时需要用的

seed。仅当 selection 为 random 时才可用。

'''