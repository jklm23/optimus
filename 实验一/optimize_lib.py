
import math
import numpy as np
def golden_section(f, section, delta,flag=1):
    '''
    黄金分割法
    :param f: 单谷函数
    :param section: 初始区间
    :param delta: 区间精度
    :return: 目标函数的最小值
    '''
    t = 0.618  # 黄金分割比
    # 当前区间
    alpha = section[0]
    beta = section[1]

    lam = alpha + (1 - t) * (beta - alpha)
    mu = alpha + t * (beta - alpha)

    alpha_list = [alpha]
    beta_list = [beta]
    lam_list = [lam]
    mu_list = [mu]

    while (beta - alpha >= delta):
        if f(lam) - f(mu) > 0:
            alpha = lam
            lam = mu
            mu = alpha + t * (beta - alpha)
        else:
            beta = mu
            mu = lam
            lam = alpha + (1 - t) * (beta - alpha)
        alpha_list.append(alpha)
        beta_list.append(beta)
        lam_list.append(lam)
        mu_list.append(mu)
    if flag==1:
        print('alpha_list:' + str(alpha_list) + '\n')
        print('beta_list:' + str(beta_list) + '\n')
        print('lambda_list:' + str(lam_list) + '\n')
        print('mu_list:' + str(mu_list) + '\n')

    return (alpha + beta) / 2


def fibonacci_search(f, section, delta):
    '''
    斐波那契法
    '''
    arr = [1, 1]  # Fibonacci数组
    stop_condition = (section[1] - section[0]) / delta  # 求n
    i = 2
    alpha = section[0]
    beta = section[1]
    alpha_list = [alpha]
    beta_list = [beta]
    lam_list = []
    mu_list = []

    while True:
        tmp = arr[i - 1] + arr[i - 2]
        arr.append(tmp)
        if tmp >= stop_condition:
            break
        i += 1

    k = len(arr) - 1
    lam = alpha + (arr[k - 2] / arr[k]) * (beta - alpha)
    mu = alpha + (arr[k - 1] / arr[k]) * (beta - alpha)
    lam_list.append(lam)
    mu_list.append(mu)

    while k >= 2:

        if f(lam) - f(mu) > 0:
            alpha = lam
            lam = mu
            mu = alpha + (arr[k - 1] / arr[k]) * (beta - alpha)
        else:
            beta = mu
            mu = lam
            lam = alpha + (arr[k - 2] / arr[k]) * (beta - alpha)

        alpha_list.append(alpha)
        beta_list.append(beta)
        lam_list.append(lam)
        mu_list.append(mu)
        k -= 1
    print('Fibonacci:' + str(arr) + '\n')
    print('alpha_list:' + str(alpha_list) + '\n')
    print('beta_list:' + str(beta_list) + '\n')
    print('lambda_list:' + str(lam_list) + '\n')
    print('mu_list:' + str(mu_list) + '\n')

    return (alpha + beta) / 2


def bin_split(f_dao, section, delta):
    '''
    二分法
    :param f_dao: 函数导数
    :param section:
    :param delta:
    :return:
    '''
    alpha = section[0]
    beta = section[1]
    alpha_list = [alpha]
    beta_list = [beta]
    lam_list = []
    dao = []
    while (beta - alpha >= delta):
        lam = (alpha + beta) / 2
        lam_list.append(lam)
        dao.append(f_dao(lam))
        if f_dao(lam) == 0:
            break
        elif f_dao(lam) > 0:
            beta = lam
        else:
            alpha = lam
        alpha_list.append(alpha)
        beta_list.append(beta)

    print('alpha_list:' + str(alpha_list) + '\n')
    print('beta_list:' + str(beta_list) + '\n')
    print('lambda_list:' + str(lam_list) + '\n')
    print('dao_list:' + str(dao) + '\n')
    return lam_list[-1]


def dichotomous(f, section, delta, eps):
    '''
    dichotomous法
    :param section:
    :param delta:
    :param eps:
    :return:
    '''
    alpha = section[0]
    beta = section[1]
    alpha_list = [alpha]
    beta_list = [beta]
    lam = (alpha + beta) / 2 - eps
    mu = (alpha + beta) / 2 + eps
    lam_list = [lam]
    mu_list = [mu]

    while (beta - alpha >= delta):

        if f(lam) > f(mu):
            alpha = lam
        else:
            beta = mu
        lam = (alpha + beta) / 2 - eps
        mu = (alpha + beta) / 2 + eps

        lam_list.append(lam)
        mu_list.append(mu)
        alpha_list.append(alpha)
        beta_list.append(beta)

    print('alpha_list:' + str(alpha_list) + '\n')
    print('beta_list:' + str(beta_list) + '\n')
    print('lambda_list:' + str(lam_list) + '\n')
    print('mu_list:' + str(mu_list) + '\n')

    return (alpha + beta) / 2



def goldstein(f,gradf,x,d,lamb,alpha,beta,p):
    '''
    :param f: 目标函数
    :param gradf: 函数梯度
    :param x: 初始点
    :param d: 初始方向
    :param lamb: 初始步长
    :param alpha: 步长增大系数
    :param beta: 步长缩短系数
    :return: 最佳lambda
    '''

    lamb_list=[lamb]
    while True:
        x_next=x+lamb*d
        # print(p)
        # print(gradf(x))
        # print(lamb)
        # print(d)
        # print(gradf(x).T.dot(lamb))
        if f(x_next)-f(x)>p*lamb*float(gradf(x).T.dot(d)):
            lamb*=beta
            lamb_list.append(lamb)
            continue
        if f(x_next)-f(x)<(1-p)*lamb*float(gradf(x).T.dot(d)):
            lamb*=alpha
            lamb_list.append(lamb)
            continue
        print('lambda的变化情况：'+str(lamb_list))
        return lamb
def goldstein_price(f,gradf,x,d,lamb,alpha,beta,p,sigma):
    '''
    :param f: 目标函数
    :param gradf: 函数梯度
    :param x: 初始点
    :param d: 初始方向
    :param lamb: 初始步长
    :param alpha: 步长增大系数
    :param beta: 步长缩短系数
    :param sigma: 条件2的参数
    :return: 最佳lambda
    '''

    lamb_list=[lamb]
    while True:
        x_next=x+lamb*d
        # print(p)
        # print(gradf(x))
        # print(lamb)
        # print(d)
        # print(gradf(x).T.dot(lamb))
        if f(x_next)-f(x)>p*lamb*float(gradf(x).T.dot(d)):
            lamb*=beta
            lamb_list.append(lamb)
            continue
        if f(x_next)-f(x)<sigma*lamb*float(gradf(x).T.dot(d)):
            lamb*=alpha
            lamb_list.append(lamb)
            continue
        print('lambda的变化情况：'+str(lamb_list))
        return lamb



def wolf_powell(f,gradf,x,d,lamb,alpha,beta,p):
    '''
    :param f: 目标函数
    :param gradf: 函数梯度
    :param x: 初始点
    :param d: 初始方向
    :param lamb: 初始步长
    :param alpha: 步长增大系数
    :param beta: 步长缩短系数
    :return: 最佳lambda
    '''

    lamb_list=[lamb]
    while True:
        x_next=x+lamb*d
        # print(p)
        # print(gradf(x))
        # print(lamb)
        # print(d)
        # print(gradf(x).T.dot(lamb))
        if f(x_next)-f(x)>p*lamb*float(gradf(x).T.dot(d)):
            lamb*=beta
            lamb_list.append(lamb)
            continue
        if gradf(x_next).T.dot(lamb*d)<(1-p)*lamb*float(gradf(x).T.dot(d)):
            lamb*=alpha
            lamb_list.append(lamb)
            continue
        print('lambda的变化情况：'+str(lamb_list))
        return lamb


def Hfunc(H, p, q):
    '''
    DFP算法的修正公式
    :param H:
    :param p:
    :param q:
    :return:
    '''
    return H+(1/(p.T.dot(q)))*p.dot(p.T)-(1/(q.T.dot(H).dot(q)))*(H.dot(q).dot(q.T).dot(H))

def Hfunc_bfgs(H,p,q):
    '''
    BFGS的修正公式
    :param H:
    :param p:
    :param q:
    :return:
    '''
    return H+(1+1/p.T.dot(q)*q.T.dot(H).dot(q))*(1/p.T.dot(q))*p.dot(p.T)-(1/p.T.dot(q))*(p.dot(q.T).dot(H)+H.dot(q).dot(p.T))


def fAlpha(x, a, judge,function):
    '''
    用进退法找到搜索区间中用到的函数，用于计算φ(α)
    '''
    if(judge == 0):
        return (function(x + a*np.array([[1, 0]])))
    if(judge == 1):
        return (function(x + a*np.array([[0, 1]])))
    pass


def SearchRegion(x, judge,function):  # x为变量的矩阵，judge为判断迭代为变量x1还是变量x2
    '''
    进退法求搜索区间
    '''
    x=x.reshape(1,2)
    a_0 = 0
    h = 0.1
    a_1 = a_0
    a_2 = a_0 + h
    
    while(1):
        f1 = fAlpha(x, a_1, judge,function)
        f2 = fAlpha(x, a_2, judge,function)

        # 判断前进还是后退
        if(f2 < f1):
            a_3 = a_2 + h
            f3 = fAlpha(x, a_3, judge,function)
            
            #判断搜索区间
            if(f2 <= f3): # 满足高低高条件，直接输出搜索区间
                return np.array([a_1, a_3])
            if(f2 > f3): # 不满足高低高条件，继续搜索
                h = 2*h
                a_1 = a_2
                a_2 = a_3
        
        # 判断前进还是后退
        if(f2 >= f1):
            h = -h
            # 对调a_1和a_2
            t = a_1
            a_1 = a_2
            a_2 = t
            # 对调f1和f2
            t = f1
            f1 = f2
            f2 = t
            
            a_3 = a_2 + h
            f3 = fAlpha(x, a_3, judge,function)
            
            # 判断搜索区间
            if(f3 >= f2): # 满足高低高条件，直接输出搜索区间
                return np.array([a_3, a_1])
            if(f3 < f2): # 不满足高低高条件，继续搜索
                h = -2*h
                a_1 = a_2
                a_2 = a_3
        pass
    pass


def quasi_Newton(x,eps,gradf,f,hfunc,**lists):
    '''
    拟牛顿法
    :param x: 初始点
    :param eps: 允许误差
    :param gradf: 梯度函数
    :param hfunc: 修正公式
    :return:
    '''
    # 设定初始值
    n=x.shape[0]
    g=gradf(x)
    H=np.identity(n)
    k=1
    d = -g.dot(H)

    def f_lambda(lamb_x):
        return f(x + lamb_x * d)
    lam = golden_section(f_lambda, [-100, 100], 0.03, flag=0)
    lists['x_list'].append(x)

    while True:
        d = -g.dot(H)

        def f_lambda(lamb_x):
            return f(x + lamb_x * d)
        # 使用进退法确定初始区间
        # 使用黄金分割法确定步长
        qujian=SearchRegion(x,0,f)
        print(qujian)
        lam = golden_section(f_lambda, list(qujian), 0.03,flag=0)
        x_next=x+lam*d
        if abs(np.linalg.norm(gradf(x_next),ord=1))<=eps:
            x=x_next
            break
        if k==n:
            lists['x_list'].append(x_next)
            lists['grad_list'].append(gradf(x_next))
            lists['H_list'].append(H)
            lists['d_list'].append(d)
            lists['lambda_list'].append(lam)
            lists['k_list'].append(k)
            return quasi_Newton(x_next,eps,gradf,f,hfunc,
                       x_list=lists['x_list'],
                       grad_list=lists['grad_list'],
                       H_list=lists['H_list'],
                       d_list=lists['d_list'],
                       lambda_list=lists['lambda_list'],
                       k_list=lists['k_list'])
        else:
            g_next=gradf(x_next)
            p=x_next-x
            q=g_next-g
            H=hfunc(H,p.reshape((2,1)),q.reshape((2,1)))
            x=x_next
            g=g_next
            k+=1

            lists['x_list'].append(x_next)
            lists['grad_list'].append(gradf(x_next))
    lists['x_list'].append(x_next)

def con_gra(x,grad,f,Q,eps):
    '''
    共轭梯度法
    :param x:
    :param grad:
    :param f:
    :param Q: 正定矩阵
    :return:
    '''
    def lam(p, Q, x):
        '''
        步长因子
        :param p:
        :param Q:
        :return:
        '''
        return -float(np.dot(p.T, grad(x))) / float(np.dot(np.dot(p.T, Q), p))

    def direction(p, Q, x):
        '''
        共轭方向的计算
        :param p:
        :param x:
        :return:
        '''
        return -grad(x) + float((p.T.dot(Q).dot(grad(x))) / (p.T.dot(Q).dot(p))) * p

    gradi = grad(x)  # 初始梯度
    p_i = gradi  # 初始方向
    print('初始方向：'+str(p_i))
    lambda_i = float(lam(p_i, Q, x))
    p_i_res = [p_i]
    lambda_i_res = [lambda_i]
    f_res = [f(x)]
    x1 = [x[0][0]]
    x2 = [x[1][0]]

    while abs(np.linalg.norm(gradi, ord=1)) > eps:  # ord=1代表1范数
        # print('lambda_i:'+str(lambda_i))
        # print('p_i:'+str(p_i))
        x += lambda_i * p_i
        # print('x:'+str(x))
        # print('新梯度：'+str(grad(x)))
        gradi = grad(x)
        f_res.append(f(x))
        x1.append(x[0][0])
        x2.append(x[1][0])
        if abs(np.linalg.norm(gradi, ord=1)) <= eps:
            # 新梯度超过eps，终止
            break

        p_i = direction(p_i, Q, x)
        lambda_i = float(lam(p_i, Q, x))
        lambda_i_res.append(lambda_i)
        p_i_res.append(p_i)
        # print('-----')
    print('-------')
    print('步长因子变化：' + str(lambda_i_res))
    print('方向变化:' + str(p_i_res))
    print('迭代点：')
    for i in range(len(x1)):
        print('(%f,%f)'%(x1[i],x2[i]))
    print('目标函数值变化:' + str(f_res))

    return f_res[-1]

def momentum_gd(x,gradf,lr,momentum,eps,f):
    '''
    带动量的梯度下降
    :param x:
    :param gradf:
    :param lr: 学习率
    :param momentum: 动量
    :return:
    '''

    x_list=[str(x)]
    f_list=[str(f(x))]
    v=np.zeros(x.shape) # 初始变化量为0
    while np.linalg.norm(gradf(x),ord=1)>eps:
        v=momentum*v+lr*gradf(x)
        x-=v
        # print(x)
        # print(np.linalg.norm(gradf(x),ord=1))
        x_list.append(str(x))
        f_list.append(str(f(x)))
    return x_list,f_list

def sqrt_arr(x):
    '''
    array的每个元素进行运算
    :param x:
    :return:
    '''
    return math.sqrt(x)

def one_(x):
    return 1/x


def adagrad(x,gradf,lr,mineps,eps,f):
    '''

    :param x:
    :param gradf:
    :param lr:
    :param mineps: 防止分母为0
    :param f:
    :return:
    '''
    x_list = [str(x)]
    f_list = [str(f(x))]
    n=np.zeros(x.shape)
    while np.linalg.norm(gradf(x), ord=1) > eps:
        n+=gradf(x)*gradf(x)

        tmp=n+mineps
        tmp=np.array(list(map(sqrt_arr,tmp)))
        tmp = np.array(list(map(one_, tmp)))
        tmp*=gradf(x)

        delta=-lr*tmp
        x+=delta
        # print(x)
        x_list.append(str(x))
        f_list.append(str(f(x)))
    return x_list,f_list
