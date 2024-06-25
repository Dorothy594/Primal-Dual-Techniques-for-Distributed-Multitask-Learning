import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def generate_labels(W):
    features_list = []
    labels_list = []
    for i in range(2):  # 对于每个簇
        for j in range(nodes_num):  # 对于每个节点
            features = np.random.normal(loc=0.0, scale=1.0, size=(m, n))
            label = np.dot(features, W[i]) + np.random.normal(0, noise_sd)
            features_list.append(features)
            labels_list.append(label)
    features_array = np.array(features_list).reshape(-1, n)
    labels_array = np.array(labels_list).reshape(-1)
    return features_array, labels_array


# 定义簇的大小和权重向量
nodes_num = 100
# weight = [0.2, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0, 20.0]
weight = np.array(range(0, 200, 1))
m, n = 1, 2  # 特征向量的形状
noise_sd = 0  # 噪声标准差
correlation = []
noise = []


for i in weight:
    W1 = np.array([i, i])
    W2 = np.array([-i, i])
    W = [W1, W2]
    print(f"weight: {i}")
    features_small, labels_small = generate_labels(W)

    # 1. 标签值的范围增大
    print("Label Range:", np.min(labels_small), np.max(labels_small))

    # 2. 特征与标签的线性关系增强
    correlation_small = np.corrcoef(features_small.T, labels_small)[0, 1]
    print("Feature-Label Correlation:", correlation_small)
    correlation.append(np.abs(correlation_small))

    # 3. 噪声的相对影响减小
    noise_small = labels_small - np.dot(features_small, W[0])
    noise_ratio_small = np.var(noise_small) / np.var(labels_small)
    print("Noise Ratio:", noise_ratio_small)
    noise.append(noise_ratio_small)
    print("-----------------------------------------------------------------")

    # # 可视化标签分布
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # sns.histplot(labels_small, kde=True)
    # plt.title("Labels Distribution with Small Weights")
    # plt.show()
    # plt.close()
plt.plot(weight, correlation)
plt.xlabel("Weight")
plt.ylabel("Correlation Coefficient")
plt.show()
plt.close()

plt.plot(weight, noise)
plt.xlabel("Weight")
plt.ylabel("Noise Ratio")
plt.show()
plt.close()