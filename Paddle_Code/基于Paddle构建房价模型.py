"""
    1.加载库文件
    2.数据预处理
        2.1.加载数据集
        2.2.数据形状变换
        2.3.数据集划分
        2.4.数据归一化
        2.5.封装函数
    3.搭建神经网络
    4.配置训练模型
    5.模型训练
    6.测试训练模型
"""
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Linear
import numpy as np
import os,random

def load_data():
    #2.1 加载数据集
    datafile = "datasets/boston.data"
    data = np.fromfile(datafile,sep=" ")
    feature_names = ["CRIM","ZN","INDUS","CHAS","NOX","RM","RM","AGE","DIS","RAD","TAX","B","LSTAT","MEDV"]
    feature_num = len(feature_names)
    #2.2数据形状转换
    # 将原始数据Reshape,变成[N,14]这样的形状
    data = data.reshape([data.shape[0]//feature_num,feature_num])
    #2.3数据集划分
    radio = 0.8
    offset = int(data.shape[0]*radio)

    # 计算train数据集最大值，最小值，平均值
    