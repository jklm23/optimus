import random
import math
import numpy as np
import matplotlib.pyplot as plt


ra = 1  # Rosenbrock的参数a
rb = 100  # Rosenbrock的参数b

# ra = 1
# rb = 98

# ra = 1
# rb = 10

# ra = 3
# rb = 10

# ra = 5
# rb = 10


# 适用最速下降和共轭梯度
def goldsteinsearch(f, df, d, x, limit, rho, alpha, beta):
    '''
    线性搜索子函数
    数f，导数df，当前迭代点x和当前搜索方向d
    一维线性搜索中的goldstein近似搜索，不是黄金分割
    lamada为步长，limit为步长上限，lamada在0到limit之间，rho为基础参数，
    alpha为步长增大系数，beta为步长缩短系数
    '''

    flag = 0
    phi0 = f(x)
    dphi0 = np.dot(df(x), d)
    # lamada = limit * random.uniform(0, 1)
    lamada = limit * 0.4

    i = 0
    imax = 10000  # 以防找不到最大步长，设置迭代次数
    while flag == 0 and i < imax:
        phi = f(x + lamada * d)

        if (phi - phi0) <= (rho * lamada * dphi0):
            if (phi - phi0) >= ((1 - rho) * lamada * dphi0):
                flag = 1
            else:
                lamada = alpha * lamada
        else:
            lamada = beta * lamada

        i = i + 1

    return lamada


# 适用动量和Nesterov动量
def goldsteinsearch2(f, df, d, x, v, epss, limit, rho, alpha, beta):
    '''
    线性搜索子函数
    数f，导数df，当前迭代点x和当前搜索方向d
    一维线性搜索中的goldstein近似搜索，不是黄金分割
    lamada为步长，limit为步长上限，lamada在0到limit之间，rho为基础参数，
    alpha为步长增大系数，beta为步长缩短系数
    '''

    flag = 0
    phi0 = f(x)
    dphi0 = np.dot(df(x), d)
    # lamada = limit * random.uniform(0, 1)
    lamada = limit * 0.4

    i = 0
    imax = 10000  # 以防找不到最大步长，设置迭代次数
    while flag == 0 and i < imax:
        v = epss * v + lamada * d  # 确定v，这里注意这是方向而非梯度，所以要变号
        phi = f(x + v)

        if (phi - phi0) <= (rho * lamada * dphi0):
            if (phi - phi0) >= ((1 - rho) * lamada * dphi0):
                flag = 1
            else:
                lamada = alpha * lamada
        else:
            lamada = beta * lamada

        i = i + 1

    return lamada


# Rosenbrock函数
def rosenbrock(x):
    return (ra - x[0]) ** 2 + rb * (x[1] - x[0] ** 2) ** 2


# Rosenbrock函数的梯度向量（对每个变量求偏导）
def jacobian(x):
    return np.array([-4 * rb * x[0] * (x[1] - x[0] ** 2) - 2 * (ra - x[0]), 2 * rb * (x[1] - x[0] ** 2)])


# 梯度向量的模长
def grad_len(grad):
    vec_len = math.sqrt(pow(grad[0], 2) + pow(grad[1], 2))
    return vec_len


# 最速下降法
def steepest(x0):
    imax = 10000000  # 设定最大迭代次数
    eps = 10 ** (-5)  # 设定阈值
    x = x0
    grad = jacobian(x)  # 计算当前梯度
    delta = grad_len(grad)  # 计算当前梯度对应的模长
    alpha = 0.0001

    W = np.zeros((2, imax))  # 2行imax列，用于存储每个迭代点
    epo = np.zeros((2, imax))  # 用于存储迭代次数与其对应的delta
    W[:, 0] = x0  # 冒号表示选中所有行，0表示选中第0列
    i = 1

    count_list = []
    delta_list = []

    while i < imax and delta > eps:
        p = -jacobian(x)  # 确定方向
        # alpha = goldsteinsearch(rosenbrock, jacobian, p, x, 1, 0.1, 1.5, 0.5)  # 确定步长（一维线搜索）
        x = x + alpha * p  # 迭代求点

        W[:, i] = x  # W的第i列为当前迭代点x
        count_list.append(i)
        delta_list.append(delta)
        if i % 5 == 0:
            epo[:, i] = np.array((i, delta))
            print("i = %d, delta = %e" % (i, delta))

        grad = jacobian(x)
        delta = grad_len(grad)
        i = i + 1

    count_list.append(i)
    delta_list.append(delta)
    print("i = %d, delta = %e" % (i, delta))
    print('Rosenbrock函数中的a = %d, b = %d:' %(ra, rb))
    print('初始点为:', x0)
    print("迭代次数为:", i)
    print("最终误差为:", delta)
    print("近似最优解为:", x)
    W = W[:, 0:i]  # 记录迭代点

    return [W, epo, count_list, delta_list]


# 共轭梯度法
def frcg(x0):
    imax = 10000000
    eps = 10 ** (-5)
    x = x0
    grad = jacobian(x)  # 计算当前梯度
    delta = grad_len(grad)  # 计算当前梯度对应的模长
    exgrad = np.zeros(2)
    exp = np.zeros(2)
    alpha = 0.0001

    W = np.zeros((2, imax))  # 2行imax列，用于存储每个迭代点
    epo = np.zeros((2, imax))  # 用于存储迭代次数与其对应的delta
    W[:, 0] = x0  # 冒号表示选中所有行，0表示选中第0列
    i = 1

    count_list = []
    delta_list = []

    while i < imax and delta > eps:
        # 确定方向
        if i == 1:
            p = -grad
        else:
            beta = 1.0 * np.dot(grad, grad) / np.dot(exgrad, exgrad)
            p = -grad + beta * exp
            gp = np.dot(grad, p)  # 向量点积（其实就是矩阵相乘法则）
            if gp >= 0.0:  # p与grad同向情况需重新定义p
                p = -grad
        # alpha = goldsteinsearch(rosenbrock, jacobian, p, x, 1, 0.1, 1.5, 0.5)  # 确定步长（一维线搜索）
        x = x + alpha * p  # 迭代求点

        W[:, i] = x  # W的第i列为当前迭代点x
        count_list.append(i)
        delta_list.append(delta)
        if i % 5 == 0:
            epo[:, i] = np.array((i, delta))
            print("i = %d, delta = %e" % (i, delta))

        exgrad = grad
        exp = p
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


# 动量法
def momentum(x0):
    imax = 10000000  # 设定最大迭代次数
    eps = 10 ** (-5)  # 设定阈值
    x = x0
    grad = jacobian(x)  # 计算当前梯度
    delta = grad_len(grad)  # 计算当前梯度对应的模长

    W = np.zeros((2, imax))  # 2行imax列，用于存储每个迭代点
    epo = np.zeros((2, imax))  # 用于存储迭代次数与其对应的delta
    W[:, 0] = x0  # 冒号表示选中所有行，0表示选中第0列
    i = 1

    epss = 0.5  # 动量参数，可取0.5,0.9，或者0.99，分别表示最大速度2倍，10倍，100倍于SGD的算法
    v = np.zeros(2)  # 初始化速度
    # alpha = 0.0001  # 确定步长为常数

    count_list = []
    delta_list = []

    while i < imax and delta > eps:
        alpha = goldsteinsearch2(rosenbrock, jacobian, -grad, x, v, epss, 1, 0.1, 1.5, 0.5)  # 确定步长（一维线搜索）
        v = epss * v - alpha * grad  # 确定v
        x = x + v  # 迭代求点

        W[:, i] = x  # W的第i列为当前迭代点x
        count_list.append(i)
        delta_list.append(delta)
        if i % 5 == 0:
            epo[:, i] = np.array((i, delta))
            print("i = %d, delta = %e" % (i, delta))

        grad = jacobian(x)
        delta = grad_len(grad)
        i = i + 1

    count_list.append(i)
    delta_list.append(delta)
    print("i = %d, delta = %e" % (i, delta))
    print('Rosenbrock函数中的a = %d, b = %d:' %(ra, rb))
    print('初始点为:', x0)
    print("迭代次数为:", i)
    print("最终误差为:", delta)
    print("近似最优解为:", x)
    W = W[:, 0:i]  # 记录迭代点

    return [W, epo, count_list, delta_list]


# Nesterov动量法
def nesterov_momentum(x0):
    imax = 10000000  # 设定最大迭代次数
    eps = 10 ** (-5)  # 设定阈值
    x = x0
    grad = jacobian(x)  # 计算当前梯度
    delta = grad_len(grad)  # 计算当前梯度对应的模长

    W = np.zeros((2, imax))  # 2行imax列，用于存储每个迭代点
    epo = np.zeros((2, imax))  # 用于存储迭代次数与其对应的delta
    W[:, 0] = x0  # 冒号表示选中所有行，0表示选中第0列
    i = 1

    epss = 0.5  # 动量参数，可取0.5,0.9，或者0.99，分别表示最大速度2倍，10倍，100倍于SGD的算法
    v = np.zeros(2)  # 初始化速度
    # alpha = 0.0001  # 确定步长为常数

    count_list = []
    delta_list = []

    while i < imax and delta > eps:
        g_future = jacobian(x + epss * v)  # 将来位置梯度
        alpha = goldsteinsearch2(rosenbrock, jacobian, -g_future, x, v, epss, 1, 0.1, 1.5, 0.5)  # 确定步长（一维线搜索）
        v = epss * v - alpha * g_future  # 确定v
        x = x + v  # 迭代求点

        W[:, i] = x  # W的第i列为当前迭代点x
        count_list.append(i)
        delta_list.append(delta)
        if i % 5 == 0:
            epo[:, i] = np.array((i, delta))
            print("i = %d, delta = %e" % (i, delta))

        grad = jacobian(x)
        delta = grad_len(grad)
        i = i + 1

    count_list.append(i)
    delta_list.append(delta)
    print("i = %d, delta = %e" % (i, delta))
    print('Rosenbrock函数中的a = %d, b = %d:' %(ra, rb))
    print('初始点为:', x0)
    print("迭代次数为:", i)
    print("最终误差为:", delta)
    print("近似最优解为:", x)
    W = W[:, 0:i]  # 记录迭代点

    return [W, epo, count_list, delta_list]


# Adagrad法
def adagrad(x0):
    imax = 10000000  # 设定最大迭代次数
    eps = 10 ** (-5)  # 设定阈值
    x = x0
    grad = jacobian(x)  # 计算当前梯度
    delta = grad_len(grad)  # 计算当前梯度对应的模长

    W = np.zeros((2, imax))  # 2行imax列，用于存储每个迭代点
    epo = np.zeros((2, imax))  # 用于存储迭代次数与其对应的delta
    W[:, 0] = x0  # 冒号表示选中所有行，0表示选中第0列
    i = 1

    mysum = np.zeros(2)
    alpha = 0.99  # 确定步长为常数
    mye = 0.00000001

    count_list = []
    delta_list = []

    while i < imax and delta > eps:
        mysum = mysum + np.square(grad)
        x = x - alpha * grad / (mye + np.sqrt(mysum))  # 迭代求点

        W[:, i] = x  # W的第i列为当前迭代点x
        count_list.append(i)
        delta_list.append(delta)
        if i % 5 == 0:
            epo[:, i] = np.array((i, delta))
            print("i = %d, delta = %e" % (i, delta))

        grad = jacobian(x)
        delta = grad_len(grad)
        i = i + 1

    count_list.append(i)
    delta_list.append(delta)
    print("i = %d, delta = %e" % (i, delta))
    print('Rosenbrock函数中的a = %d, b = %d:' %(ra, rb))
    print('初始点为:', x0)
    print("迭代次数为:", i)
    print("最终误差为:", delta)
    print("近似最优解为:", x)
    W = W[:, 0:i]  # 记录迭代点

    return [W, epo, count_list, delta_list]


# RMSProp法
def rmsprop(x0):
    imax = 10000000  # 设定最大迭代次数
    eps = 10 ** (-5)  # 设定阈值
    x = x0
    grad = jacobian(x)  # 计算当前梯度
    delta = grad_len(grad)  # 计算当前梯度对应的模长

    W = np.zeros((2, imax))  # 2行imax列，用于存储每个迭代点
    epo = np.zeros((2, imax))  # 用于存储迭代次数与其对应的delta
    W[:, 0] = x0  # 冒号表示选中所有行，0表示选中第0列
    i = 1

    mysum = np.zeros(2)
    alpha = 0.0001  # 确定步长为常数
    mye = 0.00000001
    gama = 0.9999  # 确定衰减率为常数（ra = 1, rb = 100）
    # gama = 0.99999  # # 确定衰减率为常数（ra = 5, rb = 10）

    count_list = []
    delta_list = []

    while i < imax and delta > eps:
        if i == 1:
            mysum = np.square(grad)
        else:
            mysum = gama * mysum + (1 - gama) * np.square(grad)
        x = x - alpha * grad / (mye + np.sqrt(mysum))  # 迭代求点

        W[:, i] = x  # W的第i列为当前迭代点x
        count_list.append(i)
        delta_list.append(delta)
        if i % 5 == 0:
            epo[:, i] = np.array((i, delta))
            print("i = %d, delta = %e" % (i, delta))

        grad = jacobian(x)
        delta = grad_len(grad)
        i = i + 1

    count_list.append(i)
    delta_list.append(delta)
    print("i = %d, delta = %e" % (i, delta))
    print('Rosenbrock函数中的a = %d, b = %d:' %(ra, rb))
    print('初始点为:', x0)
    print("迭代次数为:", i)
    print("最终误差为:", delta)
    print("近似最优解为:", x)
    W = W[:, 0:i]  # 记录迭代点

    return [W, epo, count_list, delta_list]


# Adadelta法
def adadelta(x0):
    imax = 10000000  # 设定最大迭代次数
    eps = 10 ** (-5)  # 设定阈值
    x = x0
    grad = jacobian(x)  # 计算当前梯度
    delta = grad_len(grad)  # 计算当前梯度对应的模长

    W = np.zeros((2, imax))  # 2行imax列，用于存储每个迭代点
    epo = np.zeros((2, imax))  # 用于存储迭代次数与其对应的delta
    W[:, 0] = x0  # 冒号表示选中所有行，0表示选中第0列
    i = 1

    mysum = np.zeros(2)
    delta_x = np.zeros(2)
    mye = 0.00000001
    gama = 0.9999  # 确定衰减率为常数（ra = 1, rb = 100）
    # gama = 0.99999  # # 确定衰减率为常数（ra = 5, rb = 10）

    count_list = []
    delta_list = []

    while i < imax and delta > eps:
        if i == 1:
            mysum = np.square(grad)
        else:
            mysum = gama * mysum + (1 - gama) * np.square(grad)

        gg = np.multiply(np.sqrt((mye + delta_x) / (mye + mysum)), grad)
        x = x - gg  # 迭代求点
        delta_x = gama * delta_x + (1 - gama) * np.square(gg)

        W[:, i] = x  # W的第i列为当前迭代点x
        count_list.append(i)
        delta_list.append(delta)
        if i % 5 == 0:
            epo[:, i] = np.array((i, delta))
            print("i = %d, delta = %e" % (i, delta))

        grad = jacobian(x)
        delta = grad_len(grad)
        i = i + 1

    count_list.append(i)
    delta_list.append(delta)
    print("i = %d, delta = %e" % (i, delta))
    print('Rosenbrock函数中的a = %d, b = %d:' %(ra, rb))
    print('初始点为:', x0)
    print("迭代次数为:", i)
    print("最终误差为:", delta)
    print("近似最优解为:", x)
    W = W[:, 0:i]  # 记录迭代点

    return [W, epo, count_list, delta_list]


# Adam法
def adam(x0):
    imax = 10000000  # 设定最大迭代次数
    eps = 10 ** (-5)  # 设定阈值
    x = x0
    grad = jacobian(x)  # 计算当前梯度
    delta = grad_len(grad)  # 计算当前梯度对应的模长

    W = np.zeros((2, imax))  # 2行imax列，用于存储每个迭代点
    epo = np.zeros((2, imax))  # 用于存储迭代次数与其对应的delta
    W[:, 0] = x0  # 冒号表示选中所有行，0表示选中第0列
    i = 1

    mysum = np.zeros(2)
    myv = np.zeros(2)
    mysum_correct = np.zeros(2)
    myv_correct = np.zeros(2)
    mye = 0.00000001
    gamav = 0.9
    gamas = 0.999
    alpha = 0.0001

    count_list = []
    delta_list = []

    while i < imax and delta > eps:
        if i == 1:
            myv = grad
            mysum = np.square(grad)
        else:
            myv = gamav * myv + (1 - gamav) * grad
            mysum = gamas * mysum + (1 - gamas) * np.square(grad)
        myv_correct = myv / (1 - math.pow(gamav, i))
        mysum_correct = mysum / (1 - math.pow(gamas, i))
        x = x - alpha * myv_correct / (mye + np.sqrt(mysum_correct))  # 迭代求点

        W[:, i] = x  # W的第i列为当前迭代点x
        count_list.append(i)
        delta_list.append(delta)
        if i % 5 == 0:
            epo[:, i] = np.array((i, delta))
            print("i = %d, delta = %e" % (i, delta))

        grad = jacobian(x)
        delta = grad_len(grad)
        i = i + 1

    count_list.append(i)
    delta_list.append(delta)
    print("i = %d, delta = %e" % (i, delta))
    print('Rosenbrock函数中的a = %d, b = %d:' %(ra, rb))
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
    X2 = np.arange(-11, 26 + 0.05, 0.05)
    [x1, x2] = np.meshgrid(X1, X2)
    f = (ra - x1) ** 2 + rb * (x2 - x1 ** 2) ** 2  # 给定的函数
    plt.contour(x1, x2, f, 30)  # 画出函数的30条轮廓线
    plt.plot(W[0, :], W[1, :], 'go-')  # 画出迭代点收敛的轨迹
    plt.plot(W[0, -1], W[1, -1], 'go-', c='r')

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
    # list_out = steepest(x0)  # 最速下降法
    list_out = frcg(x0)  # 共轭梯度法
    # list_out = momentum(x0)  # 动量法
    # list_out = nesterov_momentum(x0)  # Nesterov动量法
    # list_out = adagrad(x0)  # Adagrad法
    # list_out = rmsprop(x0)  # RMSProp法
    # list_out = adadelta(x0)  # Adadelta法
    # list_out = adam(x0)  # Adam法
    W = list_out[0]
    epo = list_out[1]
    count_list = list_out[2]
    delta_list = list_out[3]

    Draw_Figure(W, count_list, delta_list)
