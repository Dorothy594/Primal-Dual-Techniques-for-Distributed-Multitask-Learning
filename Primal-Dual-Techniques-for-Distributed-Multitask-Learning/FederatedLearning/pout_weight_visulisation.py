import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


# 定义改进的反比例函数
def improved_inverse_func(x, a, b, c):
    return a / (x + b) + c


# 定义指数衰减函数
def exp_decay(x, a, b):
    return a * np.exp(-b * x)


# 文件路径
file_path = r'C:\Users\DELL\Desktop\20 Jun\pout-weight_error.xlsx'

# 读取Excel文件的Sheet2工作表
data = pd.read_excel(file_path, sheet_name='Sheet2')

# 打印前几行数据以确认读取成功
print(data.head())

weights = data['weight']
pouts = data['pout']

# 使用 curve_fit 拟合改进的反比例函数到数据，并增加参数约束
popt, pcov = curve_fit(improved_inverse_func, weights, pouts, bounds=([0, 0, 0], [np.inf, np.inf, np.inf]))

# 拟合的参数
a, b, c = popt
print(f"Fitted parameters: a = {a}, b = {b}, c = {c}")

# 生成拟合曲线
# x_fit = np.linspace(0.1, 200, 400)
pouts_fit = improved_inverse_func(weights, *popt)

# # 使用 curve_fit 拟合指数衰减函数到数据
# popt, pcov = curve_fit(exp_decay, weights, pouts)
#
# # 拟合的参数
# a, b = popt
# print(f"Fitted parameters: a = {a}, b = {b}")
#
# # 计算拟合值
# pouts_fit = exp_decay(weights, *popt)

# 计算误差（MSE 和 RMSE）
mse = np.mean((pouts - pouts_fit) ** 2)
rmse = np.sqrt(mse)
print(f"MSE: {mse}")
print(f"RMSE: {rmse}")

# 生成拟合曲线
x_fit = np.linspace(0.1, 200, 400)
y_fit = improved_inverse_func(x_fit, *popt)

# 绘制原始数据点和拟合曲线
plt.figure()
plt.scatter(weights[:-1], pouts[:-1], color='blue', label='Data')
plt.plot(x_fit, y_fit, color='red', label='Fitted curve')
plt.xlabel('Weight')
plt.ylabel('pout')
plt.title('pout vs Weight')
plt.legend()
plt.grid(True)
plt.show()