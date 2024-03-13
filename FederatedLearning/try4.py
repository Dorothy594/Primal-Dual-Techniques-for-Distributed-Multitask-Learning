import matplotlib.pyplot as plt
import numpy as np

mse = {'total': 0.1, 'train': 0.2, 'test': 0.3}

MSE = []
for i in range(3):
    MSE.append(mse)

# 提取出每个字典中的值
total_values = [item['total'] for item in MSE]
train_values = [item['train'] for item in MSE]
test_values = [item['test'] for item in MSE]

import pdb; pdb.set_trace()

# 生成横坐标
x = np.arange(len(MSE))

# 绘制折线图
plt.plot(x, total_values, label='total')
plt.plot(x, train_values, label='train')
plt.plot(x, test_values, label='test')

# 添加标签和标题
plt.xlabel('Index')
plt.ylabel('MSE')
plt.title('MSE Values')
plt.legend()

# 显示图形
plt.show()
