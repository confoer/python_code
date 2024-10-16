import matplotlib.pyplot as plt

# plt.rc("font", family='Microsoft YaHei')  # 设置全局字体为微软雅黑

# input_values = [1,2,3,4,5]
# squares = [1,4,9,16,25]

# fig,ax = plt.subplots()#fig:表示整张图片 ax:表示图片中的图表
# ax.plot(input_values,squares,linewidth = 3)
# ax.plot(squares,linewidth=3)

# ax.set_title("平方数",fontsize = 24)# 设置标题
# ax.set_xlabel("值",fontsize = 24)# 横坐标标题
# ax.set_ylabel("值的平方",fontsize = 24)# 纵坐标标题
# ax.tick_params(axis = 'both',labelsize = 14)# 设置刻度大小

# print(plt.style.available)# 展示plt内置样式
# plt.show()

plt.style.use("seaborn-v0_8-dark")
fig,ax = plt.subplots()#可在一张图片中绘制多个图标
# 指定坐标
# x_values = [1,2,3,4,5]
# y_values = [1,4,9,16,25]
x_values = range(1,1001)
y_values = [x**2 for x in x_values]
# ax.axis([0,1100,0,11000000])#指定每个坐标轴的取值范围
# ax.scatter(x_values,y_values,color='red',s=100)# 绘制坐标点 color:指定颜色
# ax.scatter(x_values,y_values,color =[0.1,0.7,0.5],s = 10)# 值越接近0 颜色越深 越接近1 颜色越浅

# 颜色渐变
ax.scatter(x_values,y_values,c= y_values,cmap=plt.cm.Blues,s = 10)
plt.savefig('squares_plot.png',bbox_inches = 'tight')#bbox_inches:将多余部分去除
# plt.show()