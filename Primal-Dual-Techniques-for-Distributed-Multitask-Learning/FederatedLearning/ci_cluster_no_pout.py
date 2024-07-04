# import numpy as np
# import torch
# from torch.autograd import Variable
# import networkx as nx
# import random
# import matplotlib.pyplot as plt
# from collections import defaultdict
# from joblib import Parallel, delayed
# import multiprocessing
# import algorithm.optimizer as opt
# from sklearn.preprocessing import StandardScaler
# import pandas as pd
#
#
# np.random.seed(0)
#
#
# def mean_squared_error(y_true, y_pred):
#     return np.mean((y_true - y_pred) ** 2)
#
#
# def consensus_innovation(iterations, datapoints, adj_matrix, learning_rate=0.06, lambda_reg=0, calculate_score=False):
#     num_nodes = adj_matrix.shape[1]
#     weights = np.zeros((num_nodes, datapoints[0]['features'].shape[1]))
#     iteration_scores = []
#
#     for i in range(num_nodes):
#         weights[i] = np.zeros(datapoints[i]['features'].shape[1])
#
#     for _ in range(iterations):
#         weights_new = np.copy(weights)
#
#         for i in range(num_nodes):
#             consensus_sum = np.zeros(weights[i].shape)
#             for j in range(num_nodes):
#                 if adj_matrix[i, j] != 0:
#                     consensus_sum += adj_matrix[i, j] * weights[j]
#
#             gradient = np.dot(datapoints[i]['features'].T, (np.dot(datapoints[i]['features'], weights[i]) - datapoints[i]['label']))
#             regularization_term = lambda_reg * weights[i]
#             gradient += regularization_term
#
#             weights_new[i] = consensus_sum - learning_rate * gradient
#
#         weights = np.copy(weights_new)
#
#         if calculate_score:
#             Y_pred = np.array([datapoints[i]['features'] @ weights[i] for i in range(num_nodes)])
#             true_labels = np.array([datapoints[i]['label'] for i in range(num_nodes)])
#             mse = mean_squared_error(true_labels, Y_pred)
#             iteration_scores.append(mse)
#
#     return iteration_scores, weights
#
#
# def generate_clustered_data(num_clusters, cluster_size, m=1, n=2, noise_sd=0, is_torch_model=True):
#     total_size = num_clusters * cluster_size
#     datapoints = {}
#     true_labels = []
#     cluster_labels = np.repeat(np.arange(num_clusters), cluster_size)
#
#     W = [np.random.randn(n) for _ in range(num_clusters)]
#
#     for i in range(total_size):
#         cluster = cluster_labels[i]
#         features = np.random.normal(loc=0.0, scale=1.0, size=(m, n))
#         label = np.dot(features, W[cluster]) + np.random.normal(0, noise_sd)
#         true_labels.append(label)
#
#         if is_torch_model:
#             model = opt.TorchLinearModel(n)
#             optimizer = opt.TorchLinearOptimizer(model)
#             features = Variable(torch.from_numpy(features)).to(torch.float32)
#             label = Variable(torch.from_numpy(label)).to(torch.float32)
#         else:
#             model = opt.LinearModel(cluster, features, label)
#             optimizer = opt.LinearOptimizer(model)
#
#         datapoints[i] = {
#             'features': features,
#             'label': label,
#             'optimizer': optimizer
#         }
#
#     return cluster_labels, np.array(true_labels), datapoints
#
#
# def create_adjacency_matrix(cluster_labels):
#     num_nodes = len(cluster_labels)
#     adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
#
#     for i in range(num_nodes):
#         for j in range(i + 1, num_nodes):
#             if cluster_labels[i] == cluster_labels[j]:
#                 adjacency_matrix[i, j] = 1
#                 adjacency_matrix[j, i] = 1
#
#     return adjacency_matrix
#
#
# def calculate_a_kl(k, l, matrix, degrees):
#     if k != l:
#         if matrix[k][l] == 1:
#             return 1 / max(degrees[k], degrees[l])
#         else:
#             return 0
#     else:
#         return 1 - sum(calculate_a_kl(k, i, matrix, degrees) for i in range(len(matrix)) if i != k and matrix[k][i] == 1)
#
#
# def create_a_matrix(matrix, degrees):
#     size = len(matrix)
#     a_matrix = np.zeros((size, size))
#     for k in range(size):
#         for l in range(size):
#             a_matrix[k][l] = calculate_a_kl(k, l, matrix, degrees)
#     return a_matrix
#
# def get_consensus_innovation_MSE(K, datapoints, samplingset, matrix):
#     total_error, _ = consensus_innovation(K, datapoints, matrix, calculate_score=True)
#     consensus_innovation_MSE = {'total': total_error}
#     return consensus_innovation_MSE
#
#
# # Parameters
# num_clusters = 2
# cluster_size = 100
# iteration = 1000
#
# # Generate data and clusters based on label values
# cluster_labels, true_labels, datapoints = generate_clustered_data(num_clusters, cluster_size, is_torch_model=False)
# adjacency_matrix = create_adjacency_matrix(cluster_labels)
# degrees = np.sum(adjacency_matrix, axis=1)
# new_matrix = create_a_matrix(adjacency_matrix, degrees)
#
# num_tries = 1
# num_cores = multiprocessing.cpu_count()
#
#
# def fun(matrix):
#     samplingset = random.sample([j for j in range(len(cluster_labels))], k=int(0.8 * len(cluster_labels)))
#     return get_consensus_innovation_MSE(iteration, datapoints, samplingset, matrix)
#
#
# results = Parallel(n_jobs=num_cores)(delayed(fun)(new_matrix) for i in range(num_tries))
#
# consensus_innovation_scores = defaultdict(list)
# for result in results:
#     consensus_innovation_scores['norm1'].append(result)
#
# total_values = [item['total'] for item in consensus_innovation_scores['norm1']]
#
# last_100_data = np.copy(total_values[0][900:1000])
#
# print(f'consensus + innovation:',
#       '\n mean total MSE:', np.mean(last_100_data),
#       '\n std_dev total MSE:', np.std(last_100_data))
#
# x_total = np.arange(len(total_values[0]))
# plt.semilogy(x_total, np.mean(total_values, axis=0), label='total')
# plt.title('Train, learning_rate=0.06')
# plt.show()
#
# # Plot adjacency matrix heatmap
# plt.figure(figsize=(8, 6))
# plt.imshow(new_matrix, cmap='binary', interpolation='none')
# plt.colorbar()
# plt.title('Adjacency Matrix Heatmap')
# plt.xlabel('Node Index')
# plt.ylabel('Node Index')
# plt.show()


import numpy as np
import torch
from torch.autograd import Variable
import networkx as nx
import random
import matplotlib.pyplot as plt
from collections import defaultdict
from joblib import Parallel, delayed
import multiprocessing
import algorithm.optimizer as opt
from sklearn.preprocessing import StandardScaler
import pandas as pd

np.random.seed(0)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def consensus_innovation(iterations, datapoints, adj_matrix, learning_rate=0.06, lambda_reg=0, calculate_score=False):
    num_nodes = adj_matrix.shape[1]
    num_features = datapoints[0]['features'].shape[1]
    weights = np.zeros((num_nodes, num_features))
    iteration_scores = []

    for _ in range(iterations):
        weights_new = np.copy(weights)

        for i in range(num_nodes):
            consensus_sum = np.zeros(weights[i].shape)
            for j in range(num_nodes):
                if adj_matrix[i, j] != 0:
                    consensus_sum += adj_matrix[i, j] * weights[j]

            gradient = np.dot(datapoints[i]['features'].T,
                              (np.dot(datapoints[i]['features'], weights[i]) - datapoints[i]['label']))
            regularization_term = lambda_reg * weights[i]
            gradient += regularization_term

            weights_new[i] = consensus_sum - learning_rate * gradient

        weights = np.copy(weights_new)

        if calculate_score:
            Y_pred = np.array([datapoints[i]['features'] @ weights[i] for i in range(num_nodes)])
            true_labels = np.array([datapoints[i]['label'] for i in range(num_nodes)])
            mse = mean_squared_error(true_labels, Y_pred)
            iteration_scores.append(mse)

    return iteration_scores, weights


def generate_data(num_clusters, cluster_size, m=1, n=2, noise_sd=0, is_torch_model=True):
    # Generate either SBM graph or clustered data based on is_torch_model flag
    if is_torch_model:
        # Generate SBM graph
        block_sizes = [cluster_size] * num_clusters
        p_in = 0.6  # Intra-community edge probability
        p_out = 0.1  # Inter-community edge probability
        block_prob = np.ones((num_clusters, num_clusters)) * p_out
        np.fill_diagonal(block_prob, p_in)

        sbm_graph = nx.stochastic_block_model(block_sizes, block_prob, seed=0)
        adjacency_matrix = nx.to_numpy_array(sbm_graph)

        # Generate random features and labels for SBM nodes
        datapoints = {}
        for i in range(len(sbm_graph.nodes)):
            features = np.random.normal(loc=0.0, scale=1.0, size=(m, n))
            label = np.dot(features, np.random.randn(n)) + np.random.normal(0, noise_sd)
            datapoints[i] = {
                'features': features,
                'label': label
            }

        return adjacency_matrix, datapoints

    else:
        # Generate clustered data (similar to previous implementation)
        cluster_labels = np.repeat(np.arange(num_clusters), cluster_size)
        datapoints = {}
        true_labels = []
        W = [np.random.randn(n) for _ in range(num_clusters)]

        for i in range(num_clusters * cluster_size):
            cluster = cluster_labels[i]
            features = np.random.normal(loc=0.0, scale=1.0, size=(m, n))
            label = np.dot(features, W[cluster]) + np.random.normal(0, noise_sd)
            true_labels.append(label)

            model = opt.LinearModel(cluster, features, label)
            optimizer = opt.LinearOptimizer(model)

            datapoints[i] = {
                'features': features,
                'label': label,
                'optimizer': optimizer
            }

        adjacency_matrix = create_adjacency_matrix(cluster_labels)
        return adjacency_matrix, datapoints


def create_adjacency_matrix(cluster_labels):
    num_nodes = len(cluster_labels)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if cluster_labels[i] == cluster_labels[j]:
                adjacency_matrix[i, j] = 1
                adjacency_matrix[j, i] = 1

    return adjacency_matrix


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


def get_consensus_innovation_MSE(K, datapoints, samplingset, matrix):
    total_error, _ = consensus_innovation(K, datapoints, matrix, calculate_score=True)
    consensus_innovation_MSE = {'total': total_error}
    return consensus_innovation_MSE


# Parameters
num_clusters = 2
cluster_size = 100
iteration = 1000

# Generate data and clusters based on label values
adjacency_matrix, datapoints = generate_data(num_clusters, cluster_size, is_torch_model=False)
degrees = np.sum(adjacency_matrix, axis=1)
new_matrix = create_a_matrix(adjacency_matrix, degrees)

num_tries = 1
num_cores = multiprocessing.cpu_count()


def fun(matrix):
    samplingset = random.sample([j for j in range(len(datapoints))], k=int(0.8 * len(datapoints)))
    return get_consensus_innovation_MSE(iteration, datapoints, samplingset, matrix)


results = Parallel(n_jobs=num_cores)(delayed(fun)(new_matrix) for i in range(num_tries))

consensus_innovation_scores = defaultdict(list)
for result in results:
    consensus_innovation_scores['norm1'].append(result)

total_values = [item['total'] for item in consensus_innovation_scores['norm1']]
last_100_data = np.copy(total_values[0][900:1000])

print(f'consensus + innovation:',
      '\n mean total MSE:', np.mean(last_100_data),
      '\n std_dev total MSE:', np.std(last_100_data))

x_total = np.arange(len(total_values[0]))
plt.semilogy(x_total, np.mean(total_values, axis=0), label='total')
plt.title('Train, learning_rate=0.06')
plt.show()

# Plot adjacency matrix heatmap
plt.figure(figsize=(8, 6))
plt.imshow(new_matrix, cmap='binary', interpolation='none')
plt.colorbar()
plt.title('Adjacency Matrix Heatmap, learning_rate=0.06')
plt.xlabel('Node Index')
plt.ylabel('Node Index')
plt.show()
