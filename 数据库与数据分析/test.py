import matplotlib.pyplot as plt
import numpy as np
# 画一条（0，0）到（6，250）的线段
xpoints = np.array([0,6])
ypoints = np.array([0,250])
xpie = np.array([15,35,20,30])
explode = [0.2,0,0,0]

plt.legend()# 生成图例
plt.pie(xpoints)# 创建饼图
plt.pie(xpie,explode=explode)# 将某块区域突出
plt.bar(xpoints,ypoints)# 创建柱状图
plt.barh(xpoints,ypoints)# 创建水平柱状图
plt.hist(xpoints,ypoints)# 创建直方图
plt.scatter(xpoints,ypoints)# 创建散点图
plt.plot(xpoints,ypoints)# plot:用于绘制图形上的点
# 'o':仅绘制起始点与终点
plt.plot(xpoints,ypoints,'o')

plt.plot(xpoints,ypoints,marker = 'o')# marker:强调点
# 'o':加粗 '*':用星星标记

plt.plot(xpoints,ypoints,marker = 'o',ms = 20)# ms:设置点的大小
plt.plot(xpoints,ypoints,marker = 'o',ms = 20,mec = 'r')# mec:设置点的边缘颜色
plt.plot(xpoints,ypoints,marker = 'o',ms = 20,mfc = 'r')# mfc:设置点的内部颜色

plt.plot(xpoints,ypoints,'o:r')# 点的样式|线的样式|颜色

plt.plot(xpoints,ypoints,linestyle = 'dotted')# linestyle：设置线条样式