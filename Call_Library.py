import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np


ra = 1
rb = 100


# Rosenbrock函数
def rosenbrock(x):
    return (ra - x[0]) ** 2 + rb * (x[1] - x[0] ** 2) ** 2


# 优化过程
def Myoptim(method, x, imax):
    x_list = []
    delta_list = []
    count = 1  # 迭代次数
    x_list.append(x.data.clone())

    for i in range(imax):
        method.zero_grad()  # 因为 backward 自动求导，导数会增加，所以每次迭代前需要导数清零
        y = rosenbrock(x)
        y.backward()  # 求导
        method.step()  # 算法更新 x，对于SGD它等同于 x = x - lr*x.grad
        x_list.append(x.data.clone())  # 拷贝时不共享内存
        delta = y.data - 0
        delta_list.append(delta)

        if count % 5 == 0:
            print("count = %d, delta = %e" % (count, delta))

        if delta <= 1e-5:
            break

        count = count + 1

    return x.data[0], x.data[1], count, x_list, delta_list


# 优化过程
def Myoptim2(method, x, imax):
    x_list = []
    delta_list = []
    count = 1  # 迭代次数
    x_list.append(x.data.clone())

    for i in range(imax):
        def closure():
            method.zero_grad()  # 因为 backward 自动求导，导数会增加，所以每次迭代前需要导数清零
            y = rosenbrock(x)
            y.backward()  # 求导
            return y

        method.step(closure)  # 算法更新 x
        x_list.append(x.data.clone())  # 拷贝时不共享内存
        delta = closure().data - 0
        delta_list.append(delta)

        print("count = %d, delta = %e" % (count, delta))

        if delta <= 1e-5:
            break

        count = count + 1

    return x.data[0], x.data[1], count, x_list, delta_list


# 结果图
def Draw_Figure(W, count_list, delta_list):
    plt.figure(num="结果图", figsize=(10, 5), dpi=200)

    plt.subplot(121)
    X1 = np.arange(-4, 6 + 0.05, 0.05)
    X2 = np.arange(-11, 26 + 0.05, 0.05)
    [x1, x2] = np.meshgrid(X1, X2)
    f = (ra - x1) ** 2 + rb * (x2 - x1 ** 2) ** 2  # 给定的函数
    plt.contour(x1, x2, f, 30)  # 画出函数的30条轮廓线
    plt.plot(W[0, :], W[1, :], 'g*-')  # 画出迭代点收敛的轨迹
    plt.plot(W[0, -1], W[1, -1], 'g*-', c='r')

    plt.subplot(122)
    plt.title("Delta", fontsize=20)
    plt.xlabel("iteration number", fontsize=12)
    plt.ylabel("delta", fontsize=12)
    # plt.xlim((0, len(count_list) + 1))
    plt.xticks([])
    plt.plot(count_list, delta_list, marker='o')

    plt.show()

    return 0


if __name__ == '__main__':
    # x0 = torch.rand(2)
    npx0 = np.array([-3.0, -10.0])
    x0 = torch.Tensor(npx0)
    imax = 2000000
    x = Variable(x0.clone(), requires_grad=True)
    sgd = torch.optim.SGD(params=[x], lr=0.0001)  # lr是学习率（步长）
    adagrad = torch.optim.Adagrad(params=[x], lr=0.9, lr_decay=0, weight_decay=0)
    rmsprop = torch.optim.RMSprop(params=[x], lr=0.0001, alpha=0.9999, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    adam = torch.optim.Adam(params=[x], lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
    lbfgs = torch.optim.LBFGS(params=[x], lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05, tolerance_change=1e-09,
                      history_size=100, line_search_fn=None)

    # myx1, myx2, count, x_list, delta_list = Myoptim(sgd, x, imax)
    # myx1, myx2, count, x_list, delta_list = Myoptim(adagrad, x, imax)
    # myx1, myx2, count, x_list, delta_list = Myoptim(rmsprop, x, imax)
    # myx1, myx2, count, x_list, delta_list = Myoptim(adam, x, imax)
    myx1, myx2, count, x_list, delta_list = Myoptim2(lbfgs, x, imax)

    # 画图
    W = np.zeros((2, len(x_list)))  # 2行len(x_list)列，用于存储每个迭代点
    i = 0
    for myx in x_list:
        W[:, i] = myx.numpy()  # W的第i列为当前迭代点x
        i = i + 1

    count_list = []
    for i in range(1, count + 1):
        count_list.append(i)

    Draw_Figure(W, count_list, delta_list)

    print('Rosenbrock函数中的a = %d, b = %d:' % (ra, rb))
    print("初始点为: [%f, %f]" % (x0.data[0], x0.data[1]))
    print("迭代次数为:", count)
    print("最终误差为:", delta_list[-1].clone().numpy())
    print("近似最优解为: [%f, %f]" % (myx1, myx2))
