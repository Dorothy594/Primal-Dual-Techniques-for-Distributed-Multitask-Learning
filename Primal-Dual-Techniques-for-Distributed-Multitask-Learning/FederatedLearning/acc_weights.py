import numpy as np
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


def generate_sbm(nodes_num, pin, pout, weight, noise_sd):
    cluster_sizes = [nodes_num, nodes_num]
    probs = np.array([[pin, pout], [pout, pin]])
    G = nx.stochastic_block_model(cluster_sizes, probs, seed=0)

    W1 = np.array([weight, weight])
    W2 = np.array([-weight, weight])

    W = [W1, W2]

    true_labels = []
    features = []

    for i in range(2):
        for j in range(nodes_num):
            true_labels.append(i)
            noise = np.random.normal(0, noise_sd, size=W[i].shape)
            features.append(W[i] + noise)
    return G, np.array(features), np.array(true_labels)


def graph(G):
    adj_matrix = nx.adjacency_matrix(G).todense()
    return adj_matrix


def calculate_a_kl(k, l, matrix, degrees):
    if k != l:
        if matrix[k][l] == 1:
            return 1 / max(degrees[k], degrees[l])
        else:
            return 0
    else:
        return 1 - sum(
            calculate_a_kl(k, i, matrix, degrees) for i in range(len(matrix)) if i != k and matrix[k][i] == 1)


def create_a_matrix(matrix, degrees):
    size = len(matrix)
    a_matrix = np.zeros((size, size))
    for k in range(size):
        for l in range(size):
            a_matrix[k][l] = calculate_a_kl(k, l, matrix, degrees)
    return a_matrix


def visualisation(matrix, nodes, w):
    count = 0
    for i in range(nodes):
        for j in range(nodes + 1, 2 * nodes):
            if matrix[i, j] != 0:
                count += 1
    ratio = 2 * count / (matrix.shape[0] * matrix.shape[1])

    sum_prob_no_diagonal = 0
    for i in range(2 * nodes):
        for j in range(2 * nodes):
            if i != j:
                sum_prob_no_diagonal += matrix[i, j]
    avg_prob_no_diagonal = sum_prob_no_diagonal / (2 * nodes * 2 * nodes - 2 * nodes)

    sum_prob = 0
    for m in range(2 * nodes):
        for n in range(2 * nodes):
            sum_prob += matrix[m, n]
    avg_prob = sum_prob / (2 * nodes * 2 * nodes)

    # 绘制热力图
    plt.imshow(matrix, cmap='Blues', interpolation='nearest')
    plt.colorbar()  # 添加颜色条
    plt.title(f"{nodes} nodes, weight = {w}")
    plt.show()

    return count, ratio, avg_prob, avg_prob_no_diagonal


def run_experiment(nodes_num, pin, pout, weights, noise_sd):
    accuracies = []
    max_weight_element = []
    for weight in weights:
        G, features, labels = generate_sbm(nodes_num, pin, pout, weight, noise_sd)
        adj = graph(G)
        degrees = [sum(row) for row in adj]
        normalized_matrix = create_a_matrix(adj, degrees)
        _, _, _, _, max_value = visualisation(normalized_matrix, nodes_num, weight)

        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42)

        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        max_weight_element.append(max_value)

    return accuracies, max_weight_element


# 实验参数
nodes_num = 15
pin = 0.5
pout = 0.01
weights = [0.5, 1.0, 2.0, 5.0]
noise_sd = 1.0

# 运行实验
accuracies, max_weight_element = run_experiment(nodes_num, pin, pout, weights, noise_sd)
print("Accuracies with different weights:", accuracies)
print("Max weight element with different weights:", max_weight_element)