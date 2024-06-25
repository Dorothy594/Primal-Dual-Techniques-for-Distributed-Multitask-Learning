import matplotlib.pyplot as plt
import pandas as pd

# 数据
data = {
    'x': [10, 11, 12, 13, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'mean': [0.018832761, 0.019779795, 0.104600407, 0.456927341, 0.231203195, 0.301369163,
             0.280494742, 0.178241955, 0.204705311, 0.270132776, 0.2560688, 0.229692976,
             0.265820557, 0.273314245]
}

df = pd.DataFrame(data)

# 画图
plt.figure(figsize=(10, 6))
plt.plot(df['x'], df['mean'], marker='o', linestyle='-', color='b')
plt.xlabel('nodes per block')
plt.ylabel('Mean')
plt.title('Mean values for different numbers of nodes per block')
plt.grid(True)
plt.xticks(df['x'])
plt.show()
