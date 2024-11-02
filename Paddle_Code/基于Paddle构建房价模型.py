"""
    1.加载库文件
    2.数据预处理
        2.1.加载数据集
        2.2.数据形状变换
        2.3.数据集划分
        2.4.数据归一化
        2.5.封装函数
    3.搭建神经网络
        3.1.定义--init--两个数
        3.2.定义forward函数:实现前向计算并返回预测结果
    4.配置训练模型
        4.1.指定运行训练的机器资源
        4.2.声明模型实例
        4.3.加载训练和测试数据
        4.4.设置优化算法和学习率
    5.模型训练
        5.1.数据准备
        5.2.前向计算
        5.3.计算损失函数
        5.4.反向传播
    6.测试训练模型并保存

"""
" 版本:,Python==3.9,Paddle==2.4 "
import paddle
import paddle.fluid as fluid
import paddle.fluid.dygraph as dygraph
from paddle.fluid.dygraph import Linear
import numpy as np
import os,random  

max_values =None
min_values = None
avgs_values = None

def load_data():#数据预处理
    #2.1 加载数据集
    datafile = "datasets/boston.data"
    data = np.fromfile(datafile,sep=" ")
    feature_names = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","MEDV"]
    feature_num = len(feature_names)
    #2.2数据形状转换
    # 将原始数据Reshape,变成[N,14]这样的形状
    data = data.reshape([data.shape[0]//feature_num,feature_num])
    #2.3数据集划分
    radio = 0.8# 数据集划分参数
    offset = int(data.shape[0]*radio)# 偏移量
    training_data = data[:offset]
    # 计算train数据集最大值，最小值，平均值
    maximuns,minimuns,avgs = training_data.max(axis=0),training_data.min(axis=0),training_data.sum(axis=0)/training_data.shape[0]
    #记录数据归一化参数，在预测对数据归一化
    global max_values
    global min_values
    global avgs_values
    max_values = maximuns
    min_values = minimuns
    avgs_values = avgs
    #对数据进行归一化处理
    for i in range(feature_num):
        data[:,i] = (data[:,i] - avgs[i])/(maximuns[i] - minimuns[i])

    #训练集和预测集划分比例
    training_data = data[:offset]
    test_data = data[offset:]
    return training_data,test_data


class Regressor(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(Regressor,self).__init__(name_scope)
        # name_scope = self.full_name()
        #定义一层全连接层，输出维度为1，激活函数为None(即不使用激活函数)
        self.fc = Linear(input_dim= 13,output_dim= 1,act=None)
        #网络的前向计算函数
    def forward(self,inputs):
        x = self.fc(inputs)
        return x

with fluid.dygraph.guard():# 定义飞桨动态图工作环境
    model = Regressor("Regressor")# 声明定义好，为线性回归模型
    model.train()# 开启训练模型模式
    training_data,test_data = load_data()# 加载数据
    #定义优化算法，此处运用SGD，学习率默认为0.01
    opt = fluid.optimizer.SGD(learning_rate=0.01,parameter_list=model.parameters())

with dygraph.guard():
    EPOCH_NUM = 10 # 设置外层循环次数
    BATCH_SIZE = 10 # 设置batch大小
    for epoch_id in range(EPOCH_NUM):# 定义外层循环
        np.random.shuffle(training_data)# 将训练数据顺序打乱
        mini_batches = [training_data[k:k+BATCH_SIZE] for k in range(0,len(training_data),BATCH_SIZE)]# 将训练数据拆分,每个batch包含十条数据
        for iter_id,mini_batch in enumerate(mini_batches): # 定义内层循环
            x = np.array(mini_batch[:,:-1]).astype("float32")#获取当前批次的训练数据
            y = np.array(mini_batch[:,-1:]).astype("float32")#获取当前批次训练的标签
            house_feature = dygraph.to_variable(x)# 将numpy数据转换成飞桨动态图
            prices = dygraph.to_variable(y)
            predicts = model(house_feature)#前向计算
            loss = fluid.layers.square_error_cost(predicts,label=prices)# 计算损失
            avg_loss = fluid.layers.mean(loss)# 损失平均值
            if iter_id%20 == 0:
                print("epoch:{},iter:{},loss is :{}".format(epoch_id,iter_id,avg_loss.numpy()))  
            avg_loss.backward()#反向传播
            opt.minimize(avg_loss)# 最小化loss,更新参数
            model.clear_gradients()#消除梯度
    fluid.save_dygraph(model.state_dict(),"LR_model") # 保存模型到文件夹

def load_one_example(data_dir):
    global max_values, min_values, avgs_values
    f = open(data_dir,'r')
    datas = f.readlines()
    tmp = datas[-10]#选择倒数第10条数据用于测试
    tmp = tmp.strip().split()
    one_data = [float(v) for v in tmp]
    #对数据进行归一化处理
    for i in range(len(one_data)-1):
        one_data[i] = (one_data[i]-avgs_values[i])/(max_values[i] - min_values[i])
    data = np.reshape(np.array(one_data[:-1]),[1,-1]).astype(np.float32)
    label = one_data[-1]
    return data,label

with dygraph.guard():
    # 参数为保存模型参数的文件地址
    model_dict,_ = fluid.load_dygraph("LR_model")
    model.load_dict(model_dict)
    model.eval()
    # 参数为数据集文件地址
    test_data,label = load_one_example("D:\\Python\\datasets\\boston.data")
    test_data = dygraph.to_variable(test_data)
    results = model(test_data)
    # 对结果做反归一化处理
    results = results*(max_values[-1]-min_values[-1])+avgs_values[-1]
    print("Inference result is {},the corrsponding label is {}".format(results.numpy(),label))