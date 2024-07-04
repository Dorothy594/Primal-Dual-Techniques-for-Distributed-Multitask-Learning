import numpy as np
import torch
from torch.autograd import Variable
import networkx as nx
import algorithm.optimizer as opt
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt


def get_sbm_data(cluster_sizes, G, W, m=1, n=2, noise_sd=0, is_torch_model=True):
    N = len(G.nodes)
    E = len(G.edges)

    B = np.zeros((E, N))
    weight_vec = np.zeros(E)
    cnt = 0
    for i, j in G.edges:
        if i > j:
            continue
        B[cnt, i] = 1
        B[cnt, j] = -1
        weight_vec[cnt] = 1
        cnt += 1

    weight_vec = weight_vec[:cnt]
    B = B[:cnt, :]

    # node_degrees = np.array((1.0 / (np.sum(abs(B), 0)))).ravel()
    node_degrees = np.array(np.sum(abs(B), 0)).ravel()
    datapoints = {}
    true_labels = []

    cnt = 0
    for i, cluster_size in enumerate(cluster_sizes):
        for j in range(cluster_size):
            features = np.random.normal(loc=0.0, scale=1.0, size=(m, n))
            label = np.dot(features, W[i]) + np.random.normal(0, noise_sd)
            true_labels.append(label)

            if is_torch_model:
                model = opt.TorchLinearModel(n)
                optimizer = opt.TorchLinearOptimizer(model)
                features = Variable(torch.from_numpy(features)).to(torch.float32)
                label = Variable(torch.from_numpy(label)).to(torch.float32)
            else:
                model = opt.LinearModel(node_degrees[i], features, label)
                optimizer = opt.LinearOptimizer(model)

            datapoints[cnt] = {
                'features': features,
                'degree': node_degrees[i],
                'label': label,
                'optimizer': optimizer
            }
            cnt += 1

    return B, weight_vec, np.array(true_labels), datapoints


def get_sbm_2blocks_data(m=1, n=2, pin=0.5, pout=0.01, noise_sd=0, is_torch_model=True):
    cluster_sizes = [100, 100]
    probs = np.array([[pin, pout], [pout, pin]])
    G = nx.stochastic_block_model(cluster_sizes, probs, seed=0)

    W1 = np.array([2, 2])
    W2 = np.array([-2, 2])
    W = [W1, W2]

    # N = len(G.nodes)
    # E = len(G.edges)

    return get_sbm_data(cluster_sizes, G, W, m, n, noise_sd, is_torch_model)


_, _, _, datapoints = get_sbm_2blocks_data()
feature = []
label = []
for idx, data in datapoints.items():
    feature.append(data['features'].flatten().tolist())
    label.append(data['label'].item())

print("Features:", feature)
print("Labels:", label)

plt.scatter(range(200), label)
plt.xlabel('Data Points')
plt.ylabel('Labels')
plt.title('Original Datapoints')
plt.show()
plt.close()

# 绘制标签值在 x 轴上的线
plt.figure(figsize=(10, 6))

# 绘制标签值在 x 轴上的线，y轴坐标使用np.zeros_like(labels)表示，也就是全部是0
# plt.scatter(label, np.zeros_like(label), marker='o', color='b', label='Labels')
# 将数据分为前100个数和后面的数
labels_first_100 = label[:100]
labels_rest = label[100:]

# 根据是否是前100个数来分类，绘制不同颜色的数据点
plt.scatter(labels_first_100, np.zeros_like(labels_first_100), marker='o', color='b', label='First 100 Labels')
plt.scatter(labels_rest, np.zeros_like(labels_rest), marker='o', color='r', label='Rest Labels')


plt.title('Labels on X-axis')
plt.xlabel('Label Values')
plt.gca().axes.get_yaxis().set_visible(False)
plt.show()
plt.close()

kmeans = KMeans(n_clusters=2, random_state=0).fit(np.array(label).reshape(-1, 1))

# Print cluster centers and labels
print("Cluster centers:", kmeans.cluster_centers_[:, 0])
print("Labels after K-means clustering:", kmeans.labels_)

# Plotting the results
plt.figure(figsize=(8, 6))

# Plot cluster centers
plt.scatter(kmeans.cluster_centers_[:, 0], np.zeros_like(kmeans.cluster_centers_), marker='x', color='red', s=100, label='Cluster Centers')

# Plot K-means labels
plt.scatter(label, np.zeros_like(label), c=kmeans.labels_, cmap='viridis', marker='o', label='K-means Labels')

plt.title('K-means Clustering on Labels')
plt.xlabel('Data Points')
plt.ylabel('Labels')
plt.legend()
plt.tight_layout()
plt.show()
plt.close()

num1 = 0
num1_100 = 0
for index, i in enumerate(kmeans.labels_):
    if i == 1:
        num1 += 1
        if index < 100:
            num1_100 += 1
num0 = len(kmeans.labels_) - num1
print(num0, num1, num1_100)
