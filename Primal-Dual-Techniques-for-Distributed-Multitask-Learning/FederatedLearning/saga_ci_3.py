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


def saga_consensus_innovation(iterations, datapoints, adj_matrix, learning_rate=0.11, lambda_reg=0, calculate_score=False):
    num_nodes = adj_matrix.shape[1]  # Number of nodes
    weights = np.zeros((num_nodes, datapoints[0]['features'].shape[1]))  # Initialize local variables
    gradients_memory = np.zeros((num_nodes, len(datapoints), datapoints[0]['features'].shape[1]))  # For SAGA
    iteration_scores = []

    # Initialize gradients
    for i in range(num_nodes):
        for n in range(len(datapoints)):
            gradients_memory[i, n] = datapoints[n]['features'].T @ (datapoints[n]['features'] @ weights[i] - datapoints[n]['label'])

    for iteration in range(iterations):
        weights_new = np.copy(weights)

        for i in range(num_nodes):
            # Consensus part
            consensus_sum = np.zeros(weights[i].shape)
            for j in range(num_nodes):
                if adj_matrix[i, j] != 0:
                    consensus_sum += adj_matrix[i, j] * weights[j]

            # SAGA update
            ni = np.random.randint(len(datapoints))
            gradient = datapoints[ni]['features'].T @ (datapoints[ni]['features'] @ weights[i] - datapoints[ni]['label'])
            avg_gradient = np.mean(gradients_memory[i], axis=0)
            saga_update = gradient + avg_gradient - gradients_memory[i, ni]

            # Regularization term
            regularization_term = lambda_reg * weights[i]
            saga_update += regularization_term

            # Update weights
            weights_new[i] = consensus_sum - learning_rate * saga_update
            gradients_memory[i, ni] = gradient

        weights = np.copy(weights_new)

        if calculate_score:
            Y_pred = np.array([datapoints[i]['features'] @ weights[i] for i in range(num_nodes)])
            true_labels = np.array([datapoints[i]['label'] for i in range(num_nodes)])
            mse = mean_squared_error(true_labels, Y_pred)
            iteration_scores.append(mse)

    print(weights)

    return iteration_scores, weights


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

    node_degrees = np.array((1.0 / (np.sum(abs(B), 0)))).ravel()
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


def incidence_to_adjacency(incidence_matrix):
    num_edges, num_nodes = incidence_matrix.shape
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    for edge_idx in range(num_edges):
        node_indices = np.where(incidence_matrix[edge_idx, :] != 0)[0]
        if len(node_indices) == 2:
            adjacency_matrix[node_indices[0], node_indices[1]] = 1
            adjacency_matrix[node_indices[1], node_indices[0]] = 1

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


def get_sbm_2blocks_data(m=1, n=2, pin=0.5, pout=0.01, noise_sd=0, is_torch_model=True):
    cluster_sizes = [100, 100]
    probs = np.array([[pin, pout], [pout, pin]])
    G = nx.stochastic_block_model(cluster_sizes, probs, seed=0)

    W1 = np.array([2, 2])
    W2 = np.array([-2, 2])
    W = [W1, W2]

    return get_sbm_data(cluster_sizes, G, W, m, n, noise_sd, is_torch_model)


def get_consensus_innovation_MSE(K, datapoints, samplingset, matrix):
    total_error, _ = saga_consensus_innovation(K, datapoints, matrix, calculate_score=True)
    consensus_innovation_MSE = {'total': total_error}
    return consensus_innovation_MSE


iteration = 5000
B, weight_vec, true_labels, datapoints = get_sbm_2blocks_data(pin=0.5, pout=0.01, is_torch_model=False)
adjacency_matrix = incidence_to_adjacency(B)

degrees = [sum(row) for row in adjacency_matrix]
new_matrix = create_a_matrix(adjacency_matrix, degrees)

num_tries = 1
num_cores = multiprocessing.cpu_count()


def fun(matrix):
    samplingset = random.sample([j for j in range(200)], k=int(0.8 * 200))
    return get_consensus_innovation_MSE(iteration, datapoints, samplingset, matrix)


results = Parallel(n_jobs=num_cores)(delayed(fun)(new_matrix) for i in range(num_tries))

consensus_innovation_scores = defaultdict(list)
for result in results:
    consensus_innovation_scores['norm1'].append(result)

total_values = [item['total'] for item in consensus_innovation_scores['norm1']]

print('consensus + innovation:',
      '\n mean total MSE:', np.mean(total_values[0]),
      '\n std_dev total MSE:', np.std(total_values[0]))

x_total = np.arange(len(total_values[0]))

plt.semilogy(x_total, np.mean(total_values, axis=0), label='total')
plt.title('Train')
plt.show()
plt.close()