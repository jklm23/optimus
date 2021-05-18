import numpy as np
import matplotlib.pyplot as plt

ra = 1
rb = 100

#函数表达式
fun = lambda x:(ra - x[0]) ** 2 + rb * (x[1] - x[0] ** 2) ** 2

#梯度向量
gfun = lambda x:np.array([-4 * rb * x[0] * (x[1] - x[0] ** 2) - 2 * (ra - x[0]), 2 * rb * (x[1] - x[0] ** 2)])
#Hessian矩阵
hess = lambda x:np.array([[12 * rb * x[0] ** 2 - 4 * rb * x[1] + 2, -4 * rb * x[0]], [-4 * rb * x[0], 2 * rb]])

def Draw_Figure(x1_list, x2_list):
    plt.figure(num="结果图", figsize=(10, 5), dpi=100)

    plt.subplot(121)
    X1 = np.arange(-4, 6 + 0.05, 0.05)
    X2 = np.arange(-11, 26 + 0.05, 0.05)
    [x1, x2] = np.meshgrid(X1, X2)
    f = (ra - x1) ** 2 + rb * (x2 - x1 ** 2) ** 2  # 给定的函数
    plt.contour(x1, x2, f, 40)  # 画出函数的30条轮廓线
    plt.plot(x1_list, x2_list, 'go-')  # 画出迭代点收敛的轨迹
    plt.plot(x1_list[-1], x2_list[-1], 'go-', c='r')    # 最终点的位置
    plt.savefig('./results/quasi_newton.png')


def dfp(fun,gfun,hess,x0):
    #用DFP算法求解无约束问题：min fun(x)
    #输入：x0式初始点，fun,gfun，hess分别是目标函数和梯度,Hessian矩阵格式
    #输出：x,val分别是近似最优点，最优解，k是迭代次数
    maxk = 1e5
    rho = 0.05
    sigma = 0.4
    epsilon = 1e-5 #迭代停止条件
    k = 0
    n = np.shape(x0)[0]
    #将Hessian矩阵初始化为单位矩阵
    Hk = np.linalg.inv(hess(x0))
    x1_list=[x0[0]]
    x2_list=[x0[1]]
    while k < maxk:
        gk = gfun(x0)
        if np.linalg.norm(gk) < epsilon:
            break
        dk = -1.0*np.dot(Hk,gk)
        # print(dk)

        m = 0
        mk = 0
        while m < 20:#用Armijo搜索步长
            if fun(x0 + rho**m*dk) < fun(x0) + sigma*rho**m*np.dot(gk,dk):
                mk = m
                break
            m += 1
        # print(mk)
        #DFP校正
        x = x0 + rho**mk*dk
        print ("第"+str(k)+"次的迭代结果为："+str(x))
        sk = x - x0
        yk = gfun(x) - gk

        if np.dot(sk,yk) > 0:
            Hy = np.dot(Hk,yk)
            sy = np.dot(sk,yk) #向量的点积
            yHy = np.dot(np.dot(yk,Hk),yk) #yHy是标量
            Hk = Hk - 1.0*Hy.reshape((n,1))*Hy/yHy + 1.0*sk.reshape((n,1))*sk/sy

        k += 1
        x0 = x
        x1_list.append(x[0])
        x2_list.append(x[1])
    return x0,fun(x0),k,x1_list,x2_list

def bfgs(fun,gfun,hess,x0):
    #功能：用BFGS族算法求解无约束问题：min fun(x) 优化的问题请参考文章开头给出的链接
    #输入：x0是初始点，fun,gfun分别是目标函数和梯度，hess为Hessian矩阵
    #输出：x,val分别是近似最优点和最优解,k是迭代次数  
    maxk = 1e5
    rho = 0.55
    sigma = 0.4
    gama = 0.7
    epsilon = 1e-5
    k = 0
    n = np.shape(x0)[0]
    #海森矩阵可以初始化为单位矩阵
    Bk = np.eye(n) #np.linalg.inv(hess(x0)) #或者单位矩阵np.eye(n)
    x1_list=[]
    x2_list=[]
    while k < maxk:
        gk = gfun(x0)
        if np.linalg.norm(gk) < epsilon:
            break
        dk = -1.0*np.linalg.solve(Bk,gk)
        m = 0
        mk = 0
        while m < 20: # 用Wolfe条件搜索求步长
            gk1 = gfun(x0 + rho**m*dk)
            if fun(x0+rho**m*dk) < fun(x0)+sigma*rho**m*np.dot(gk,dk) and np.dot(gk1.T, dk) >=  gama*np.dot(gk.T,dk):
                mk = m
                break
            m += 1

        #BFGS校正
        x = x0 + rho**mk*dk
        print ("第"+str(k)+"次的迭代结果为："+str(x))
        sk = x - x0
        yk = gfun(x) - gk   

        if np.dot(sk,yk) > 0:    
            Bs = np.dot(Bk,sk)
            ys = np.dot(yk,sk)
            sBs = np.dot(np.dot(sk,Bk),sk) 

            Bk = Bk - 1.0*Bs.reshape((n,1))*Bs/sBs + 1.0*yk.reshape((n,1))*yk/ys

        k += 1
        x0 = x
        x1_list.append(x[0])
        x2_list.append(x[1])
    return x0,fun(x0),k,x1_list,x2_list#分别是最优点坐标，最优值，迭代次数 



x0 ,fun0 ,k,x1_list,x2_list = bfgs(fun,gfun,hess,np.array([-3.0,-10.0]))

Draw_Figure(x1_list,x2_list)
print('终点：'+str(x0))