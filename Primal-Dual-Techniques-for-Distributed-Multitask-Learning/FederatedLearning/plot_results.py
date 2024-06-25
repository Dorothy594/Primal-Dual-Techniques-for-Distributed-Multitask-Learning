import matplotlib.pyplot as plt
import numpy as np

# 数据
labels = ['algorithm 1 (norm1)', 'algorithm 1 (norm2)', 'algorithm 1 (mocha)', 'federated learning',
          'linear regression', 'decision tree', 'consensus + innovation']
mean_train_MSEs = [7.203032806309109e-06, 6.585860329978702e-06, 0.0009437205525703871,
                   4.019349722589996, 3.9161373492829874, 4.082649251140066, 0.0271164743464928]
mean_test_MSEs = [8.541033838411509e-06, 6.795361832838249e-06, 0.0663734850441899,
                  4.5036236421885665, 4.280801344339161, 4.740012104012171, 0.06306635593487728]

# 计算标准差作为误差条
std_train_MSEs = np.zeros(len(labels))
std_test_MSEs = np.zeros(len(labels))

# 定义x位置
x_pos = np.arange(len(labels))

# 设置柱子宽度
bar_width = 0.35

# 画图
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# 第一张图：训练集MSE
axs[0].bar(x_pos, mean_train_MSEs, yerr=std_train_MSEs, align='center', alpha=0.5, ecolor='black', capsize=10, color='blue', width=bar_width)
axs[0].set_ylabel('MSE (Train)')
axs[0].set_xticks(x_pos)
axs[0].set_xticklabels(labels, rotation=45, ha='right')
axs[0].set_yscale('log')
axs[0].set_title('Train MSE')

# 第二张图：测试集MSE
axs[1].bar(x_pos, mean_test_MSEs, yerr=std_test_MSEs, align='center', alpha=0.5, ecolor='red', capsize=10, color='green', width=bar_width)
axs[1].set_ylabel('MSE (Test)')
axs[1].set_xticks(x_pos)
axs[1].set_xticklabels(labels, rotation=45, ha='right')
axs[1].set_yscale('log')
axs[1].set_title('Test MSE')

plt.tight_layout()
plt.savefig('result_train_test.png')
plt.show()
plt.close()
