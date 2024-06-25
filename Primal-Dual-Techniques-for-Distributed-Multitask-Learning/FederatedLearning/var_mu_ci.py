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


np.random.seed(0)


def consensus_innovation(K, samplingset, new_matrix, initial_mu=0.05, decay_rate=0.5, decay_step=50):
    def update_mu(mu, iteration):
        return mu * (decay_rate ** (iteration // decay_step))

    M = 2  # 根据 H_k 的形状和描述，M 应该是 2
    network_size = len(new_matrix)
    w_ci = np.random.randn(M, network_size, K)  # 确保初始化的形状正确
    error_ci = np.zeros((network_size, K))

    mu = initial_mu  # 初始化 mu

    for k in range(network_size):
        for l in range(network_size):
            w_ci[:, k, 1] += new_matrix[l, k] * w_ci[:, l, 0]

        # 这里确保点乘操作的两个数组形状一致
        Hk_transpose_dot_w = np.dot(H_k[:, :, k].T,
                                    w_ci[:, k, 0])  # H_k[:, :, k].T 的形状是 (2, 2)，w_ci[:, k, 0] 的形状应为 (2,)
        w_ci[:, k, 1] += - np.true_divide(mu, len(samplingset)) * rho * w_ci[:, k, 0] + np.true_divide(mu,
                                                                                                       len(samplingset)) * (
                                     d_k[:, k] - Hk_transpose_dot_w)

        error_ci[k, 0] = np.mean(np.square(np.linalg.norm(w_ci[:, k, 0] - w_star)))
        error_ci[k, 1] = np.mean(np.square(np.linalg.norm(w_ci[:, k, 1] - w_star)))

    for i in range(2, K):
        mu = update_mu(mu, i)

        for k in range(network_size):
            for l in range(network_size):
                w_ci[:, k, i] += new_matrix[l, k] * w_ci[:, l, i - 1]
            Hk_transpose_dot_w = np.dot(H_k[:, :, k].T, w_ci[:, k, i - 1])
            w_ci[:, k, i] += - np.true_divide(mu, len(samplingset)) * rho * w_ci[:, k, i - 1] + np.true_divide(mu,
                                                                                                               len(samplingset)) * (
                                         d_k[:, k] - Hk_transpose_dot_w)
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
    total_error, _ = consensus_innovation(K, datapoints, matrix)
    consensus_innovation_MSE = {'total': total_error}
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

# mu = 0.1
rho = 0.5

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
    return penalty_func, get_consensus_innovation_MSE(K, datapoints, samplingset, matrix)


results = Parallel(n_jobs=num_cores)(delayed(fun)('norm1', new_matrix) for i in range(num_tries))

for _, scores in results:
    consensus_innovation_score = scores
    consensus_innovation_scores['norm1'].append(consensus_innovation_score)

# Extract values for norm1 penalty
total_values = [item['total'] for item in consensus_innovation_scores['norm1']]

print('consensus + innovation:',
      '\n mean total MSE:', np.mean(total_values[0][0]),
      '\n std_dev total MSE:', np.std(total_values[0][0]))

x_total = np.arange(len(total_values[0][0]))

plt.plot(x_total[0:lim], np.mean(total_values, axis=(0, 1)), label='total')
plt.title('Train')
plt.show()