# 坐标轮换法（第一题）  编译环境：JupyterNotebook
import numpy as np
import matplotlib.pyplot as plt

# 初始值的设置
x_0 = np.array([[-3.0,-10.0]])  # 初始矩阵x_0
epoch = 200 # 算法的迭代次数

ra = 1
rb = 100

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
    plt.show()


# 目标函数
def function(x):
    return (ra - x[0][0]) ** 2 + rb * (x[0][1] - x[0][0] ** 2) ** 2

# 用进退法找到搜索区间中用到的函数，用于计算φ(α)
def fAlpha(x, a, judge):
    if(judge == 0):
        return (function(x + a*np.array([[1, 0]])))
    if(judge == 1):
        return (function(x + a*np.array([[0, 1]])))
    pass

#进退法求搜索区间
def SearchRegion(x, judge):  # x为变量的矩阵，judge为判断迭代为变量x1还是变量x2
    a_0 = 0
    h = 0.1
    a_1 = a_0
    a_2 = a_0 + h
    
    while(1):
        f1 = fAlpha(x, a_1, judge)
        f2 = fAlpha(x, a_2, judge)

        # 判断前进还是后退
        if(f2 < f1):
            a_3 = a_2 + h
            f3 = fAlpha(x, a_3, judge)
            
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
            f3 = fAlpha(x, a_3, judge)
            
            # 判断搜索区间
            if(f3 >= f2): # 满足高低高条件，直接输出搜索区间
                return np.array([a_3, a_1])
            if(f3 < f2): # 不满足高低高条件，继续搜索
                h = -2*h
                a_1 = a_2
                a_2 = a_3
        pass
    pass

# 黄金分割法求最优步长
def GoldenSection(x, search_region, judge):
    a = search_region[0]
    b = search_region[1]
    
    a_1 = b - 0.618*(b - a)
    a_2 = a + 0.618*(b - a)
    
    E_golden_section = 0.01 # 黄金分割法收敛精度预设为0.01
    
    f1 = fAlpha(x, a_1, judge)
    f2 = fAlpha(x, a_2, judge)
    
    # 循环搜索最优步长
    while((abs((b-a)/b) > E_golden_section) & (abs((f2-f1)/f2) > E_golden_section)):
        if(f1 <= f2):
            b = a_2
            a_2 = a_1
            a_1 = b - 0.618*(b - a)
            
        if(f1 > f2):
            a = a_1
            a_1 = a_2
            a_2 = a + 0.618*(b - a)
        pass
    
    f1 = fAlpha(x, a_1, judge)
    f2 = fAlpha(x, a_2, judge)
    return ((a + b)/2)

# 坐标轮换法求最终结果
E_univariate_search = 0.001  # 坐标轮换法的精度预设为0.01
x = x_0
x1_list=[]
x2_list=[]
count=0
for i in range(epoch):
    count+=2
    x0 = x
    search_region = SearchRegion(x, 0)
    a = GoldenSection(x, search_region, 0)
    x = x + a*np.array([[1, 0]])
    x1_list.append(x[0,0])
    x2_list.append(x[0,1])
    search_region = SearchRegion(x, 1)
    a = GoldenSection(x, search_region, 1)
    x = x + a*np.array([[0, 1]])
    x1_list.append(x[0,0])
    x2_list.append(x[0,1])

    fanshu = x - x0
    if ((((fanshu[0][0])**2 + (fanshu[0][1])**2))**0.5 < E_univariate_search):
        break
    pass


print('坐标轮换点的迭代次数：',count)
print("近似最优解为：", x[0][0], "," ,x[0][1])
print("极值点为：" , function(x))
Draw_Figure(x1_list,x2_list)