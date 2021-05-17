import random
import numpy as np
import matplotlib.pyplot as plt
import math


ra = 1
rb = 100

# ra = 3
# rb = 10

# ra = 5
# rb = 10


# Rosenbrock函数
def rosenbrock(x):
    return (ra - x[0]) ** 2 + rb * (x[1] - x[0] ** 2) ** 2


# Rosenbrock函数的梯度向量（对每个变量求偏导）
def jacobian(x):
    return np.array([-4 * rb * x[0] * (x[1] - x[0] ** 2) - 2 * (ra - x[0]), 2 * rb * (x[1] - x[0] ** 2)])


# 海森矩阵
def hess(x):
    return np.array([[12 * rb * x[0] ** 2 - 4 * rb * x[1] + 2, -4 * rb * x[0]], [-4 * rb * x[0], 2 * rb]])


# 梯度向量的模长
def grad_len(grad):
    vec_len = math.sqrt(pow(grad[0], 2) + pow(grad[1], 2))
    return vec_len


# 牛顿法
def newton(x0):
    imax = 10000000
    eps = 10 ** (-5)
    x = x0
    grad = jacobian(x)  # 计算当前梯度
    delta = grad_len(grad)  # 计算当前梯度对应的模长

    W = np.zeros((2, imax))  # 2行imax列，用于存储每个迭代点
    epo = np.zeros((2, imax))  # 用于存储迭代次数与其对应的delta
    W[:, 0] = x0  # 冒号表示选中所有行，0表示选中第0列
    i = 1

    count_list = []
    delta_list = []

    while i < imax and delta > eps:
        H = hess(x)
        p = -1.0 * np.linalg.solve(H, grad)  # 求解矩阵方程
        x = x + p

        W[:, i] = x  # W的第i列为当前迭代点x
        epo[:, i] = np.array((i, delta))
        count_list.append(i)
        delta_list.append(delta)
        print("i = %d, delta = %e" % (i, delta))

        grad = jacobian(x)
        delta = grad_len(grad)
        i = i + 1

    count_list.append(i)
    delta_list.append(delta)
    print("i = %d, delta = %e" % (i, delta))
    print('Rosenbrock函数中的a = %d, b = %d:' % (ra, rb))
    print('初始点为:', x0)
    print("迭代次数为:", i)
    print("最终误差为:", delta)
    print("近似最优解为:", x)
    W = W[:, 0:i]  # 记录迭代点

    return [W, epo, count_list, delta_list]


# 结果图
def Draw_Figure(W, count_list, delta_list):
    plt.figure(num="结果图", figsize=(10, 5), dpi=100)

    plt.subplot(121)
    X1 = np.arange(-4, 6 + 0.05, 0.05)
    X2 = np.arange(-50, 50 + 0.05, 0.05)
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


if __name__ == "__main__":
    x0 = np.array([-3.0, -10.0])
    list_out = newton(x0)  # Newton法
    W = list_out[0]
    epo = list_out[1]
    count_list = list_out[2]
    delta_list = list_out[3]

    Draw_Figure(W, count_list, delta_list)
