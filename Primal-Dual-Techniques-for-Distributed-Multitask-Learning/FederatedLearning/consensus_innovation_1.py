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


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def consensus_innovation(K, samplingset):
    w_ci = np.random.randn(M, len(samplingset), K)
    error_ci = np.zeros((len(samplingset), K))
    for i in range(1, K):
        # Consensus + Innovations
        for k in range(len(samplingset)):
            for l in range(len(samplingset)):
                w_ci[:, k, i] += B[l, k] * np.array(w_ci[:, l, i - 1])

        for k in range(len(samplingset)):
            w_ci[:, k, i] += -np.true_divide(mu, N) * rho * w_ci[:, k, i - 1] + np.true_divide(mu, N) * (
                        d_k[:, k] - np.dot(H_k[:, :, k].T, w_ci[:, k, i - 1]))
            error_ci[k, i] = np.mean(np.square(np.linalg.norm(w_ci[:, k, i] - w_star)))
    return error_ci, w_ci


# Creat SBM Graph
# from algorithm.optimizer import *


def get_sbm_data(cluster_sizes, G, W, m=5, n=2, noise_sd=0, is_torch_model=True):
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


def get_consensus_innovation_MSE(K, datapoints, samplingset):
    not_samplingset = [i for i in range(len(datapoints)) if i not in samplingset]
    total_error, _ = consensus_innovation(K, datapoints)
    train_error, _ = consensus_innovation(K, samplingset)
    test_error, _ = consensus_innovation(K, not_samplingset)
    consensus_innovation_MSE = {'total': total_error,
                                'train': train_error,
                                'test': test_error}
    return consensus_innovation_MSE


# Two Clusters
# from sparsebm import generate_SBM_dataset

def get_sbm_2blocks_data(m=5, n=2, pin=0.5, pout=0.01, noise_sd=0, is_torch_model=True):
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

K = 2000
M = 2
N = 5

rho = 0.1

w = np.random.multivariate_normal(np.zeros(M), np.square(sigma_w_squared)*np.eye(M), K).T

h = np.zeros((M, N, K))
v = np.zeros((N, K))
gamma = np.zeros((N, K))
for k in range(K):
    h[:, :, k] = np.random.multivariate_normal(np.zeros(M), sigma_h_squared*np.eye(M), N).T
    v[:, k] = np.random.normal(0, sigma_v_squared, N).T
    gamma[:, k] = np.matmul(h[:, :, k].T, w[:, k]) + v[:, k]


H_k = np.zeros((M, M, K))
d_k = np.zeros((M, K))

H = np.zeros((M, M))
d = np.zeros(M)

w_k_star = np.zeros((M, K))

for k in range(K):
    for n in range(N):
        H_k[:, :, k] += np.outer(h[:, n, k], h[:, n, k])
        d_k[:, k] += gamma[n, k]*h[:, n, k]
    w_k_star[:, k] = np.linalg.solve(H_k[:, :, k] + rho*np.eye(M), d_k[:, k])

    H += H_k[:, :, k]
    d += d_k[:, k]

# w_star = np.linalg.solve(H + rho*np.eye(M), d)
array1 = [np.array([2, 2]) for _ in range(100)]
array2 = [np.array([-2, 2]) for _ in range(100)]
w_star = [array1, array2]

# import pdb; pdb.set_trace()

mu = 0.01

B, weight_vec, true_labels, datapoints = get_sbm_2blocks_data(pin=0.5, pout=0.01, is_torch_model=False)
# E, N = B.shape

consensus_innovation_scores = defaultdict(list)
num_tries = 1
num_cores = multiprocessing.cpu_count()


def fun(penalty_func):
    samplingset = random.sample([j for j in range(N)], k=int(0.4* N))
    return penalty_func, get_consensus_innovation_MSE(K, datapoints, samplingset)


results = Parallel(n_jobs=num_cores)(delayed(fun)('norm1') for i in range(num_tries))

for penalty_func, scores in results:
    consensus_innovation_score = scores
    consensus_innovation_scores[penalty_func].append(consensus_innovation_score)

# Extract values for norm1 penalty
total_values = [item['total'] for item in consensus_innovation_scores['norm1']]
train_values = [item['train'] for item in consensus_innovation_scores['norm1']]
test_values = [item['test'] for item in consensus_innovation_scores['norm1']]

print('consensus + innovation:',
      '\n mean train MSE:', np.mean(train_values[0][0]),
      '\n mean test MSE:', np.mean(test_values[0][0]))

# 生成横坐标
x_train = np.arange(len(train_values[0][0]))
x_test = np.arange(len(test_values[0][0]))

plt.plot(x_train[0:2000], train_values[0][0][0:2000], label='train')
plt.show()
# plt.savefig('test.png')
plt.clf()
plt.plot(x_train[0:2000], test_values[0][0][0:2000], label='test')
plt.show()
# plt.savefig('train.png')