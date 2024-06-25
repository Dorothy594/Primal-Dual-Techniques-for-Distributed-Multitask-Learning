import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 读取 Excel 文件中的数据
excel_file_path = 'datapoints_10_2.xlsx'  # 修改为你的文件路径
df = pd.read_excel(excel_file_path)

# 假设 Excel 文件中有列 'feature1', 'feature2', 'label'
x = df['feature1']
y = df['feature2']
z = df['label']

# 将数据分成两个组
group1 = df.iloc[:10]
group2 = df.iloc[10:]

# 创建 3D 图
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# 绘制第一个组的 3D 散点图
ax.scatter(group1['feature1'], group1['feature2'], group1['label'], color='blue', label='Group 1')

# 绘制第二个组的 3D 散点图
ax.scatter(group2['feature1'], group2['feature2'], group2['label'], color='red', label='Group 2')

# 设置标签
ax.set_xlabel('Feature1')
ax.set_ylabel('Feature2')
ax.set_zlabel('Label')

# 添加图例
ax.legend()

# 显示图形
plt.show()
