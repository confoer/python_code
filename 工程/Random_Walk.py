from random import choice
import matplotlib.pyplot as plt

class RandomWalk:
    def __init__(self,num_points=5000):
        self.num_points = num_points

        #所有漫步都从(0,0)开始
        self.x_values = [0]
        self.y_values = [0]
    
    def fill_Walk(self):
        """计算随机漫步包含的所有点"""
        # 不断漫步，直到到达指定地点
        while len(self.x_values)<self.num_points:
            #决定前进方向及前进距离
            x_direction = choice([1,-1])
            x_distance = choice([0,1,2,3,4])
            x_step = x_direction * x_distance
            y_direction = choice([1,-1])
            y_distance = choice([0,1,2,3,4])
            y_step = y_direction * y_distance

            #拒绝原地踏步
            if x_step == 0 and y_step ==0 :
                continue

            #计算下一个点的x值与y值
            x = self.x_values[-1] + x_step
            y = self.y_values[-1] + y_step

            self.x_values.append(x)
            self.y_values.append(y)

# 模拟多次随机漫步
while True:
    rw  = RandomWalk(50_000)
    rw.fill_Walk()
    plt.style.use('classic')
    fig,ax = plt.subplots(figsize=(15,9))#figsize:调整屏幕适合大小
    point_numbers = range(rw.num_points)
    ax.scatter(rw.x_values,rw.y_values,c =point_numbers,cmap = plt.cm.Blues,edgecolors = 'none',s = 5)
    # 突出起点终点
    ax.scatter(0,0,c='green',edgecolors = 'none',s=15)
    ax.scatter(rw.x_values[-1],rw.y_values[-1],c='red',edgecolors = 'none',s=15)
    # 隐藏坐标轴
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    plt.show()

    keep_running = input("是否继续(Y/N):")
    if keep_running == 'n':
        break