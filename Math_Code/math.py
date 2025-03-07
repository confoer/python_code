import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# 示例：加载时间序列数据（假设数据存储在一个 CSV 文件中）
# 这里我们使用一个简单的示例数据集，你可以替换为自己的数据
data = sm.datasets.co2.load_pandas()
y = data.data
y = y['co2']  # 获取时间序列数据

# 数据预处理：处理缺失值（如果存在）
y = y.fillna(y.mean())

# 绘制原始时间序列
plt.figure(figsize=(10, 4))
plt.plot(y)
plt.title('Original Time Series')
plt.show()

# 定义 ARIMA 模型的参数 (p, d, q)
# p: 自回归阶数
# d: 差分阶数（用于使序列平稳）
# q: 移动平均阶数
p, d, q = 1, 1, 1

# 创建并拟合 ARIMA 模型
model = ARIMA(y, order=(p, d, q))
model_fit = model.fit()

# 输出模型摘要
print(model_fit.summary())

# 进行预测
# 这里我们预测未来 12 个时间点的值
forecast = model_fit.get_forecast(steps=12)
forecast_mean = forecast.predicted_mean
forecast_conf_int = forecast.conf_int()

# 打印预测结果
print("\n预测结果（均值）：")
print(forecast_mean)

print("\n预测结果（置信区间）：")
print(forecast_conf_int)

# 可视化预测结果
plt.figure(figsize=(10, 4))
plt.plot(y, label='观测值')
plt.plot(forecast_mean, label='预测值', color='red')
plt.fill_between(forecast_conf_int.index,
                 forecast_conf_int.iloc[:, 0],
                 forecast_conf_int.iloc[:, 1], color='pink', alpha=0.5)
plt.title('ARIMA 模型预测')
plt.legend()
plt.show()