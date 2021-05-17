import numpy as np
import matplotlib.pyplot as plt


ra = 1
rb = 100


# Rosenbrock函数
def rosenbrock(x):
    return (ra - x[0]) ** 2 + rb * (x[1] - x[0] ** 2) ** 2


# 结果图
def Draw_Figure(x1_list, x2_list, count_list, delta_list):
    plt.figure(num="结果图", figsize=(10, 5), dpi=100)

    plt.subplot(121)
    X1 = np.arange(-4, 6 + 0.05, 0.05)
    X2 = np.arange(-11, 26 + 0.05, 0.05)
    [x1, x2] = np.meshgrid(X1, X2)
    f = (ra - x1) ** 2 + rb * (x2 - x1 ** 2) ** 2  # 给定的函数
    plt.contour(x1, x2, f, 40)  # 画出函数的30条轮廓线
    plt.plot(x1_list, x2_list, 'g*-')  # 画出迭代点收敛的轨迹
    plt.plot(x1_list[-1], x2_list[-1], 'g*-', c='r')    # 最终点的位置

    plt.subplot(122)
    plt.title("Delta", fontsize=20)
    plt.xlabel("iteration number", fontsize=12)
    plt.ylabel("delta", fontsize=12)
    # plt.xlim((0, len(count_list) + 1))
    # plt.xticks([])
    plt.plot(count_list, delta_list, marker='o')

    plt.show()

    return 0


def HookeandJeeves(xk):
    x0 = xk

    # 输入初始探测搜索步长delta, 加速因子alpha(alpha>=1), 缩减率beta(0<beta<1), 允许误差epsilon(epsilon>0), 初始点xk
    delta, alpha, beta, epsilon = 0.5, 1, 0.5, 10 ** (-5)
    yk = xk.copy()

    # 求问题维数
    dim = len(xk)

    # 初始化迭代次数
    k = 1

    # 存储点
    x1_list = []
    x2_list = []
    x1_list.append(xk[0])
    x2_list.append(xk[1])

    # 存储count和delta
    count_list = []
    delta_list = []

    while delta > epsilon:
        # 输出本次搜索的基本信息
        print('进入第', k, '轮迭代')
        print('基点:', xk)
        print('基点处函数值:', rosenbrock(xk))
        print('探测出发点为:', yk)
        print('探测出发点处的函数值', rosenbrock(yk))
        print('探测搜索步长delta:', delta)

        # 进入探测移动
        for i in range(dim):
            # 生成本次探测的坐标方向
            e = np.zeros([1, dim])[0]
            e[i] = 1

            # 计算探测得到的点
            t1, t2 = rosenbrock(yk + delta * e), rosenbrock(yk)
            if t1 < t2:
                yk = yk + delta * e
            else:
                t1, t2 = rosenbrock(yk - delta * e), rosenbrock(yk)
                if t1 < t2:
                    yk = yk - delta * e
            print('第', i + 1, '次探测得到的点为', yk)
            print('函数值', rosenbrock(yk))

        # 确定新的基点和计算新的探测初始点
        t1, t2 = rosenbrock(yk), rosenbrock(xk)
        if t1 < t2:
            xk, yk = yk, yk + alpha * (yk - xk)
        else:
            delta, yk = delta * beta, xk
        x1_list.append(xk[0])
        x2_list.append(xk[1])
        count_list.append(k)
        delta_list.append(delta)

        k += 1

        print("\n")

    print('Rosenbrock函数中的a = %d, b = %d:' % (ra, rb))
    print('初始点为:', x0)
    print("迭代次数为:", k - 1)
    print("最终误差为:", delta)
    print("近似最优解为:", xk)

    return x1_list, x2_list, count_list, delta_list


if __name__ == '__main__':

    x0 = np.array([-3.0, -10.0])
    x1_list, x2_list, count_list, delta_list = HookeandJeeves(x0)
    Draw_Figure(x1_list, x2_list, count_list, delta_list)
