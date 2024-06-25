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

np.random.seed(0)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def grad_Q(w, x, y):
    # Example gradient of squared loss for linear regression
    return x * (np.dot(x, w) - y)


def combined_saga_consensus_innovation(max_iter, datapoints, adj_matrix, learning_rate=0.01, tol=1e-5,
                                       calculate_score=False):
    num_nodes = adj_matrix.shape[0]
    d = datapoints[0]['features'].shape[1]
    weights = np.zeros((num_nodes, d))  # Initialize weights for each node
    g = np.zeros((num_nodes, datapoints[0]['features'].shape[0], d))  # Initialize gradient table for each node
    avg_g = np.zeros((num_nodes, d))  # Initialize average gradient for each node
    iteration_scores = []

    # Compute initial gradients for each node
    for i in range(num_nodes):
        for n in range(datapoints[i]['features'].shape[0]):
            g[i, n] = grad_Q(weights[i], datapoints[i]['features'][n], datapoints[i]['label'][n])
        avg_g[i] = np.mean(g[i], axis=0)

    for iter_num in range(max_iter):
        weights_new = np.copy(weights)

        for i in range(num_nodes):
            # Consensus step
            consensus_sum = np.zeros(weights[i].shape)
            for j in range(num_nodes):
                if adj_matrix[i, j] != 0:
                    consensus_sum += adj_matrix[i, j] * weights[j]

            # SAGA step
            ni = np.random.randint(datapoints[i]['features'].shape[0])
            grad_wni = grad_Q(weights[i], datapoints[i]['features'][ni], datapoints[i]['label'][ni])
            weights_new[i] = consensus_sum - learning_rate * (grad_wni - g[i, ni] + avg_g[i])

            avg_g[i] += (grad_wni - g[i, ni]) / datapoints[i]['features'].shape[0]
            g[i, ni] = grad_wni

        weights = np.copy(weights_new)

        if calculate_score:
            Y_pred = np.array([datapoints[k]['features'] @ weights[k] for k in range(num_nodes)])
            true_labels = np.array([datapoints[k]['label'] for k in range(num_nodes)])
            mse = mean_squared_error(np.hstack(true_labels), np.hstack(Y_pred))
            iteration_scores.append(mse)

            # Debugging: Print MSE at each iteration
            print(f"Iteration {iter_num}, MSE: {mse}")
        # print(iter_num)
        # Convergence check (optional, based on weight updates)
        # if np.all(np.linalg.norm(weights - weights_new, axis=1) < tol):
        #     break

    # print(weights)

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
    total_error, _ = combined_saga_consensus_innovation(K, datapoints, matrix, calculate_score=True)
    consensus_innovation_MSE = {'total': total_error}
    return consensus_innovation_MSE


iteration = 1000
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

# Debugging: Check if total_values contains valid data
print(f"Total values (length {len(total_values)}): {total_values}")

if total_values and len(total_values[0]) > 0:
    print('consensus + innovation:',
          '\n mean total MSE:', np.mean(total_values[0]),
          '\n std_dev total MSE:', np.std(total_values[0]))

    x_total = np.arange(len(total_values[0]))

    plt.semilogy(x_total, np.mean(total_values, axis=0), label='total')
    plt.title('Train')
    plt.show()
    plt.close()
else:
    print("No valid data to plot.")
