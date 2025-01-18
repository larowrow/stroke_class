import numpy as np
from scipy.signal import hilbert
# from gtda.time_series import PearsonDissimilarity

# phase locking value
"""计算plv矩阵"""
def plv_matrix(X):
    ch = X.shape[1] # 通道数59 # (3000, 59)
    PLV = np.ones((ch, ch))  # 创建一个全1的矩阵 （channel，channe）
    for ch1 in range(ch - 1): # 从0到57，不包括58
        for ch2 in range(ch1 + 1, ch): # 不包括59
            # 获取两个通道的数据
            # 计算plv
            PLV[ch1,ch2] = phase_locking_value(X[:, ch1], X[:, ch2])
            PLV[ch2,ch1] = PLV[ch1,ch2]
    return PLV
# 计算plv
def phase_locking_value(theta1, theta2):
    # continuous hilbert phase function
    # 希尔伯特变换后，得到信号的瞬时相位值
    theta1 = hilbert(theta1)
    theta2 = hilbert(theta2)
    #  unwrap函数会将绝对值超过pi的角度增加或减少(2*k*pi)角度后换算回(-pi,pi)区间中
    theta1 = np.unwrap(np.angle(theta1))
    theta2 = np.unwrap(np.angle(theta2))
    complex_phase_diff = np.exp(complex(0, 1) * (theta1 - theta2))
    plv = np.abs(np.sum(complex_phase_diff)) / len(theta1)
    return plv

"""计算PLI矩阵"""
def PLI_matrix(X):
    ch = X.shape[1]  # 通道数59 # (3000, 59)
    PLI = np.ones((ch, ch))  # 创建一个全1的矩阵 （channel，channel）
    for ch1 in range(ch - 1):
        for ch2 in range(ch1 + 1, ch):
            # 获取两个通道的数据
            # 计算pli
            PLI[ch1, ch2] = Phase_Lag_Index(X[:, ch1], X[:, ch2])
            PLI[ch2, ch1] = PLI[ch1, ch2]
    for i in range(ch):
        PLI[i, i] = 0
    return PLI

# 计算PLI
def Phase_Lag_Index(x, y):
    # 希尔伯特变换
    x = np.angle(hilbert(x))
    y = np.angle(hilbert(y))
    # phase difference 相位差
    PDiff = x - y
    # print(PDiff)
    # 只计算不对称性
    # np.sin()计算所有x(作为数组元素)的三角正弦值。
    # sign()是Python的Numpy中的取数字符号（数字前的正负号）的函数。x>0为1，x=0为0，x<0为-1
    pli = np.abs(np.sum(np.sign(np.sin(PDiff))) / len(PDiff))  # only count the asymmetry
    return pli

"""计算PCC矩阵"""
# 构建脑网络（PCC）
def pcc_matrix(X):
    # 也即建立了59个“脑区”之间的关联
    PCC = np.corrcoef(X, rowvar=False)
    PCC = np.abs(PCC)
    # PCC对角线数值为1，且取它的绝对值
    return PCC

def weight2binary(matrix, Threshold=0.3):
    # 将矩阵中大于等于阈值的元素置为1，否则置为0
    binary_matrix = np.where(matrix >= Threshold, 1, 0)
    return binary_matrix


