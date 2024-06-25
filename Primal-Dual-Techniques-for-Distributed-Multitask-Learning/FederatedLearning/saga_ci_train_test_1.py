import algorithm.optimizer as opt
import algorithm.penalty as penalty
import torch
import numpy as np
from torch.autograd import Variable
import networkx as nx
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing
import pandas as pd
from torch.autograd import Variable


np.random.seed(0)


def clip_gradient(grad, max_norm):
    norm = np.linalg.norm(grad)
    if norm > max_norm:
        grad = grad * (max_norm / norm)
    return grad


def f_grad(w, idx):
    node_data = datapoints[idx]
    features = node_data['features']
    label = node_data['label']
    # 确保 w 是一个 Tensor
    w_tensor = torch.from_numpy(w).float()  # 转换 numpy 数组到 torch Tensor，并确保数据类型为 float
    features_tensor = torch.from_numpy(features).float()
    label_tensor = torch.from_numpy(label).float()
    prediction = torch.matmul(features_tensor, w_tensor)
    error = prediction.float() - label_tensor
    gradient = torch.matmul(features_tensor.T, error)
    grad = gradient.float() / len(features)  # 平均梯度
    return grad.data.numpy()  # 将梯度转换回 numpy 数组以适应后续计算


def saga(f_grad, w0, n_iter, step_size, n, data):
    w = w0.copy()
    stored_grads = np.array([f_grad(w, i) for i in range(n)])
    avg_grad = np.mean(stored_grads, axis=0)
    for i in range(n_iter):
        idx = np.random.randint(n)
        grad_current = f_grad(w, idx)

        # 应用梯度剪裁
        grad_current = clip_gradient(grad_current, max_norm=1.0)  # 假设最大范数为1.0

        # 计算用于更新的梯度，包括剪裁
        update_gradient = grad_current - stored_grads[idx] + avg_grad
        update_gradient = clip_gradient(update_gradient, max_norm=1.0)  # 再次剪裁更新梯度

        # 更新权重
        w -= step_size * update_gradient

        # 更新平均梯度和存储的梯度
        avg_grad += (grad_current - stored_grads[idx]) / n
        stored_grads[idx] = grad_current

    return w


def consensus_innovation(K, samplingset, new_matrix, network_size=200, M=2):
    # 初始化 w_ci，假设每个节点的特征/权重维度为 M
    w_ci = np.random.randn(M, network_size, K)  # 或者其他具体的初始化方式
    error_ci = np.zeros((network_size, K))
    # shape = network_size/2
    A = np.ones((100, 100))
    prob = np.block([[0.5 * A, 0.01 * A], [0.01 * A, 0.5 * A]])

    # 设置初始权重
    for k in range(len(samplingset)):
        w_ci[:, k, 0] = datapoints[samplingset[k]]['w']  # 假设 datapoints 已经包含了初始权重

    for i in range(1, K):
        # 使用 SAGA 更新每个节点
        for k in range(len(samplingset)):
            n_iter = 10  # 定义内部 SAGA 迭代次数
            step_size = 0.01
            n = len(samplingset)  # 假设每个节点考虑所有样本（可以调整）
            w_ci[:, k, i] = saga(datapoints[samplingset[k]]['f_grad'], w_ci[:, k, i-1], n_iter, step_size, n, samplingset)
            w_ci[:, k, i] *= prob[k, k]
            error_ci[k, 0] = np.mean(np.square(np.linalg.norm(w_ci[:, k, 0] - w_star)))
            error_ci[k, 1] = np.mean(np.square(np.linalg.norm(w_ci[:, k, 1] - w_star)))

        # 应用 consensus step
        for k in range(network_size):
            for l in range(network_size):
                w_ci[:, k, i] += new_matrix[l, k] * w_ci[:, l, i-1]  # 注意这里使用 i-1 来计算
                w_ci[:, k, i] *= prob[l, k]
                error_ci[k, i] = np.mean(np.square(np.linalg.norm(w_ci[:, k, i] - w_star)))

    return error_ci, w_ci


def get_sbm_data(cluster_sizes, G, W, m=1, n=2, noise_sd=0, is_torch_model=True):
    '''
    :param G: generated SBM graph with defined clusters using sparsebm.generate_SBM_dataset
    :param W: a list containing the weight vectors for each cluster
    :param m, n: shape of features vector for each node
    :param pin: the probability of edges inside each cluster
    :param pout: the probability of edges between the clusters
    :param noise_sd: the standard deviation of the noise for calculating the labels

    :return B: adjacency matrix of the graph
    :return weight_vec: a list containing the edges's weights of the graph
    :return true_labels: a list containing the true labels of the nodes
    :return datapoints: a dictionary containing the data of each node in the graph needed for the algorithm 1
    '''

    N = len(G.nodes)
    E = len(G.edges)
    '''
    N: total number of nodes
    E: total number of edges
    '''

    # create B(adjacency matrix) and edges's weights vector(weight_vec) based on the graph G
    B = np.zeros((E, N))
    '''
    B: adjacency matrix of the graph with the shape of E*N
    '''
    weight_vec = np.zeros(E)
    '''
    weight_vec: a list containing the edges's weights of the graph with the shape of E
    '''

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

    # create the data of each node needed for the algorithm 1

    node_degrees = np.array((1.0 / (np.sum(abs(B), 0)))).ravel()
    '''
    node_degrees: a list containing the nodes degree for the alg1 (1/N_i)
    '''

    datapoints = {}
    '''
    datapoints: a dictionary containing the data of each node in the graph needed for the algorithm 1,
    which are features, label, degree, and also the optimizer model for each node
    '''
    true_labels = []
    '''
    true_labels: the true labels for the nodes of the graph
    '''

    cnt = 0
    for i, cluster_size in enumerate(cluster_sizes):
        for j in range(cluster_size):
            features = np.random.normal(loc=0.0, scale=1.0, size=(m, n))
            '''
            features: the feature vector of node i which are i.i.d. realizations of a standard Gaussian random vector x~N(0,I)
            '''
            label = np.dot(features, W[i]) + np.random.normal(0, noise_sd)
            '''
            label: the label of the node i that is generated according to the linear model y = x^T w + e
            '''

            true_labels.append(label)

            if is_torch_model:
                model = opt.TorchLinearModel(n)
                optimizer = opt.TorchLinearOptimizer(model)
                features = Variable(torch.from_numpy(features)).to(torch.float32)
                label = Variable(torch.from_numpy(label)).to(torch.float32)

            else:
                model = opt.LinearModel(node_degrees[i], features, label)
                optimizer = opt.LinearOptimizer(model)
            '''
            model : the linear model for the node i 
            optimizer : the optimizer model for the node i 
            '''

            datapoints[cnt] = {
                'features': features,
                'degree': node_degrees[i],
                'label': label,
                'optimizer': optimizer
            }
            cnt += 1

    return B, weight_vec, np.array(true_labels), datapoints


# Compare Results
# %load results/compare_results.py


def get_consensus_innovation_MSE(K, datapoints, samplingset, matrix):
    not_samplingset = [i for i in range(len(datapoints)) if i not in samplingset]
    # total_error, _ = consensus_innovation(K, datapoints, matrix)
    train_error, _ = consensus_innovation(K, samplingset, matrix)
    test_error, _ = consensus_innovation(K, not_samplingset, matrix)
    consensus_innovation_MSE = {'train': train_error,
                                'test': test_error}
    return consensus_innovation_MSE


# Two Clusters
# from sparsebm import generate_SBM_dataset

def get_sbm_2blocks_data(m=1, n=2, pin=0.5, pout=0.01, noise_sd=0, is_torch_model=True):
    '''
    :param m, n: shape of features vector for each node
    :param pin: the probability of edges inside each cluster
    :param pout: the probability of edges between the clusters
    :param noise_sd: the standard deviation of the noise for calculating the labels

    :return B: adjacency matrix of the graph
    :return weight_vec: a list containing the edges's weights of the graph
    :return true_labels: a list containing the true labels of the nodes
    :return datapoints: a dictionary containing the data of each node in the graph needed for the algorithm 1
    '''
    cluster_sizes = [100, 100]
    probs = np.array([[pin, pout], [pout, pin]])

    G = nx.stochastic_block_model(cluster_sizes, probs, seed=0)
    '''
    G: generated SBM graph with 2 clusters
    '''

    # define weight vectors for each cluster of the graph

    W1 = np.array([2, 2])
    '''
    W1: the weigh vector for the first cluster
    '''
    W2 = np.array([-2, 2])
    '''
    W2: the weigh vector for the second cluster
    '''

    W = [W1, W2]

    return get_sbm_data(cluster_sizes, G, W, m, n, noise_sd, is_torch_model)


PENALTY_FUNCS = ['norm1', 'norm2', 'mocha']

sigma_h_squared = 1
sigma_v_squared = 1
sigma_w_squared = 1
sigma_w_variation_squared = 1
K = 1000  # number of iteration
M = 2
N = 200  # 160 for training and 40 for testing (line 313)
network_size = 200  # number of nodes

mu = 0.1
rho = 0.1

lim = 1000

w = np.random.multivariate_normal(np.zeros(M), np.square(sigma_w_squared)*np.eye(M), network_size).T

h = np.zeros((M, N, network_size))
v = np.zeros((N, network_size))
gamma = np.zeros((N, network_size))
for k in range(network_size):
    h[:, :, k] = np.random.multivariate_normal(np.zeros(M), sigma_h_squared*np.eye(M), N).T
    v[:, k] = np.random.normal(0, sigma_v_squared, N).T
    gamma[:, k] = np.matmul(h[:, :, k].T, w[:, k]) + v[:, k]


H_k = np.zeros((M, M, network_size))
d_k = np.zeros((M, network_size))

H = np.zeros((M, M))
d = np.zeros(M)

w_k_star = np.zeros((M, network_size))

for k in range(network_size):
    for n in range(N):
        H_k[:, :, k] += np.outer(h[:, n, k], h[:, n, k])
        d_k[:, k] += gamma[n, k]*h[:, n, k]
    w_k_star[:, k] = np.linalg.solve(H_k[:, :, k] + rho*np.eye(M), d_k[:, k])

    H += H_k[:, :, k]
    d += d_k[:, k]

w_star = np.linalg.solve(H + rho*np.eye(M), d)

B, weight_vec, true_labels, datapoints = get_sbm_2blocks_data(pin=0.5, pout=0.01, is_torch_model=False)


# 初始化权重和数据
for i in range(network_size):
    datapoints[i]['w'] = np.random.randn(M)
    datapoints[i]['f_grad'] = lambda w, idx=i: f_grad(w, idx)  # 为每个节点定义梯度函数


def incidence_to_adjacency(incidence_matrix):
    num_edges, num_nodes = incidence_matrix.shape
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for edge_idx in range(num_edges):
        # 找到边的两个端点
        node_indices = np.where(incidence_matrix[edge_idx, :] != 0)[0]
        if len(node_indices) == 2:
            # 对应的邻接矩阵元素设为 1
            adjacency_matrix[node_indices[0], node_indices[1]] = 1
            adjacency_matrix[node_indices[1], node_indices[0]] = 1  # 对称地设置，如果是无向图

    return adjacency_matrix


adjacency_matrix = incidence_to_adjacency(B)

# Calculate the degrees of nodes
degrees = [sum(row) for row in adjacency_matrix]


def modified_laplacian_rule(adjacency_matrix):
    n = adjacency_matrix.shape[0]
    a = np.zeros((n, n), dtype=float)
    degrees = adjacency_matrix.sum(axis=1)

    for k in range(n):
        for l in range(n):
            if adjacency_matrix[k, l] == 1 and k != l:
                a[k, l] = 1.0 / degrees[k]

        a[k, k] = 1 - a[k].sum()  # 调整对角线元素保证行和为1

    return a


new_matrix = modified_laplacian_rule(adjacency_matrix)

consensus_innovation_scores = defaultdict(list)
num_tries = 1
num_cores = multiprocessing.cpu_count()


def fun(penalty_func, matrix):
    samplingset = random.sample([j for j in range(200)], k=int(0.8 * 200))
    # return samplingset, penalty_func, get_consensus_innovation_MSE(K, datapoints, samplingset, matrix)
    return penalty_func, get_consensus_innovation_MSE(K, datapoints, samplingset, matrix)


# samplingset, _ = (fun)('norm1', new_matrix)
# not_samplingset = [i for i in range(len(datapoints)) if i not in samplingset]
# print(not_samplingset)
# print(len(not_samplingset))
#
# test_error, _ = consensus_innovation(K, not_samplingset, new_matrix)
#
# plt.semilogy(np.array(len(test_error)), np.mean(test_error, axis=(0, 1)), label='test')
# plt.title('Test')
# plt.show()
# # plt.savefig("sage_ci")


results = Parallel(n_jobs=num_cores)(delayed(fun)('norm1', new_matrix) for i in range(num_tries))

for _, scores in results:
    consensus_innovation_score = scores
    consensus_innovation_scores['norm1'].append(consensus_innovation_score)

# import pdb; pdb.set_trace()

# # Extract values for norm1 penalty
# total_values = [item['total'] for item in consensus_innovation_scores['norm1']]
#
# print('consensus + innovation:',
#       '\n mean total MSE:', np.mean(total_values[0][0]),
#       '\n std_dev total MSE:', np.std(total_values[0][0]))

print('consensus+innovation:',
      '\n mean train MSE:', np.mean([item['train'] for item in consensus_innovation_scores['norm1']]),
      '\n std train MSE:', np.std([item['train'] for item in consensus_innovation_scores['norm1']]),
      '\n mean test MSE:', np.mean([item['test'] for item in consensus_innovation_scores['norm1']]),
      '\n std test MSE:', np.std([item['test'] for item in consensus_innovation_scores['norm1']]))

# x_total = np.arange(len(total_values[0][0]))
#
# plt.semilogy(x_total[0:lim], np.mean(total_values, axis=(0, 1)), label='total')
# plt.title('Train')
# plt.show()
# # plt.savefig("sage_ci")
