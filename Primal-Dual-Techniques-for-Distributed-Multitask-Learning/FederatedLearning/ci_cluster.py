import numpy as np
import random
from sklearn.cluster import KMeans
from joblib import Parallel, delayed
import multiprocessing
import matplotlib.pyplot as plt
from collections import defaultdict

np.random.seed(0)


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def consensus_innovation(iterations, datapoints, adj_matrix, learning_rate=0.1, lambda_reg=0, calculate_score=False):
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


def generate_data(num_clusters, cluster_size, m=1, n=2, noise_sd=0):
    cluster_labels = np.repeat(np.arange(num_clusters), cluster_size)
    datapoints = {}
    true_labels = []

    # Manually set weights for clusters
    W = [
        np.array([2, 2]),  # Weights for cluster 1
        np.array([-2, 2])  # Weights for cluster 2
    ]

    for i in range(num_clusters * cluster_size):
        cluster = cluster_labels[i]
        features = np.random.normal(loc=0.0, scale=1.0, size=(m, n))
        label = np.dot(features, W[cluster]) + np.random.normal(0, noise_sd)
        true_labels.append(label)

        datapoints[i] = {
            'features': features,
            'label': label,
        }

    adjacency_matrix = create_full_adjacency_matrix(cluster_labels, p_in=0.5, p_out=0.01)
    return adjacency_matrix, datapoints, cluster_labels


def create_full_adjacency_matrix(cluster_labels, p_in=0.5, p_out=0.01):
    num_nodes = len(cluster_labels)
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    num_clusters = len(np.unique(cluster_labels))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if cluster_labels[i] == cluster_labels[j]:
                if np.random.rand() < p_in:
                    adjacency_matrix[i, j] = 1
                    adjacency_matrix[j, i] = 1
            else:
                if np.random.rand() < p_out:
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


def re_cluster(weights, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(weights)
    return kmeans.labels_


def optimize_and_recluster(num_clusters, cluster_size, iteration, num_reclusters, learning_rate=0.1, lambda_reg=0):
    adjacency_matrix, datapoints, cluster_labels = generate_data(num_clusters, cluster_size)
    degrees = np.sum(adjacency_matrix, axis=1)
    new_matrix = create_a_matrix(adjacency_matrix, degrees)

    num_cores = multiprocessing.cpu_count()

    mean_mse_list = []
    std_mse_list = []

    for i in range(num_reclusters):
        def fun(matrix):
            samplingset = random.sample([j for j in range(len(datapoints))], k=int(0.8 * len(datapoints)))
            return get_consensus_innovation_MSE(iteration, datapoints, samplingset, matrix)

        results = Parallel(n_jobs=num_cores)(delayed(fun)(new_matrix) for i in range(1))

        consensus_innovation_scores = defaultdict(list)
        for result in results:
            consensus_innovation_scores['norm1'].append(result)

        total_values = [item['total'] for item in consensus_innovation_scores['norm1']]
        last_100_data = np.copy(total_values[0][900:1000])

        mean_mse = np.mean(last_100_data)
        std_mse = np.std(last_100_data)

        mean_mse_list.append(mean_mse)
        std_mse_list.append(std_mse)

        print(f'consensus + innovation iteration {i}:',
              '\n mean total MSE:', mean_mse,
              '\n std_dev total MSE:', std_mse)

        _, weights = consensus_innovation(iteration, datapoints, new_matrix, learning_rate, lambda_reg)

        cluster_labels = re_cluster(weights, num_clusters)
        new_matrix = create_full_adjacency_matrix(cluster_labels, p_in=0.5, p_out=0.01)
        degrees = np.sum(new_matrix, axis=1)
        new_matrix = create_a_matrix(new_matrix, degrees)

    return new_matrix, weights, mean_mse_list, std_mse_list


# Parameters
num_clusters = 2
cluster_size = 100
iteration = 1000
num_reclusters = 100

new_matrix, final_weights, mean_mse_list, std_mse_list = optimize_and_recluster(
    num_clusters, cluster_size, iteration, num_reclusters)

# Plot adjacency matrix heatmap
plt.figure(figsize=(8, 6))
plt.imshow(new_matrix, cmap='binary', interpolation='none')
plt.colorbar()
plt.title('Adjacency Matrix Heatmap after re-clustering')
plt.xlabel('Node Index')
plt.ylabel('Node Index')
plt.show()

# Plot mean total MSE and std_dev total MSE
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(range(num_reclusters), mean_mse_list, marker='o', label='Mean Total MSE')
plt.xlabel('Reclustering Iterations')
plt.ylabel('Mean Total MSE')
plt.title('Mean Total MSE over Reclustering Iterations')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(range(num_reclusters), std_mse_list, marker='o', label='Std Dev Total MSE')
plt.xlabel('Reclustering Iterations')
plt.ylabel('Std Dev Total MSE')
plt.title('Std Dev Total MSE over Reclustering Iterations')
plt.legend()

plt.tight_layout()
plt.show()
