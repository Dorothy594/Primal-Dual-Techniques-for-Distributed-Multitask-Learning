import algorithm.optimizer as opt
import torch
import numpy as np
from torch.autograd import Variable
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score


def get_sbm_data(cluster_sizes, G, W, m=1, n=2, noise_sd=0.1, is_torch_model=True):
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


def get_sbm_2blocks_data(nodes_num, weight, m=1, n=2, pin=0.5, pout=0.01, noise_sd=0.1, is_torch_model=True):
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
    cluster_sizes = [nodes_num, nodes_num]
    probs = np.array([[pin, pout], [pout, pin]])

    G = nx.stochastic_block_model(cluster_sizes, probs, seed=0)
    '''
    G: generated SBM graph with 2 clusters
    '''

    # define weight vectors for each cluster of the graph

    W1 = np.array([weight, weight])
    '''
    W1: the weigh vector for the first cluster
    '''
    W2 = np.array([-weight, weight])
    '''
    W2: the weigh vector for the second cluster
    '''

    W = [W1, W2]

    return get_sbm_data(cluster_sizes, G, W, m, n, noise_sd, is_torch_model)


def incidence_to_adjacency(incidence_matrix):
    num_edges, num_nodes = incidence_matrix.shape
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)

    # import pdb; pdb.set_trace()

    for edge_idx in range(num_edges):
        # 找到边的两个端点
        node_indices = np.where(incidence_matrix[edge_idx, :] != 0)[0]
        if len(node_indices) == 2:
            # 对应的邻接矩阵元素设为 1
            adjacency_matrix[node_indices[0], node_indices[1]] = 1
            adjacency_matrix[node_indices[1], node_indices[0]] = 1  # 对称地设置，如果是无向图

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


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def consensus_innovation(datapoints, adj_matrix, iterations=1000, learning_rate=0.11, lambda_reg=0, calculate_score=False):  # 原lr=0.01
    # datapoints = standardize_data(datapoints)
    num_nodes = adj_matrix.shape[1]  # Number of nodes
    weights = np.zeros((num_nodes, datapoints[0]['features'].shape[1]))  # Initialize local variables
    iteration_scores = []

    for i in range(num_nodes):
        weights[i] = np.zeros(datapoints[i]['features'].shape[1])

    for _ in range(iterations):
        weights_new = np.copy(weights)

        for i in range(num_nodes):
            consensus_sum = np.zeros(weights[i].shape)
            for j in range(num_nodes):
                if adj_matrix[i, j] != 0:
                    consensus_sum += adj_matrix[i, j] * weights[j]

            # 假设 np.dot(datapoints[i]['features'], weights[i]) 是一个 numpy.ndarray
            features_product = torch.tensor(np.dot(datapoints[i]['features'], weights[i]))  # 将 numpy.ndarray 转换为 Tensor
            gradient = np.dot(datapoints[i]['features'].T, (features_product - datapoints[i]['label']))
            # gradient = np.dot(datapoints[i]['features'].T, (np.dot(datapoints[i]['features'], weights[i]) - datapoints[i]['label']))
            regularization_term = lambda_reg * weights[i]
            gradient += regularization_term

            weights_new[i] = consensus_sum - learning_rate * gradient

        weights = np.copy(weights_new)

        if calculate_score:
            Y_pred = np.array([datapoints[i]['features'] @ weights[i] for i in range(num_nodes)])
            true_labels = np.array([datapoints[i]['label'] for i in range(num_nodes)])
            mse = mean_squared_error(true_labels, Y_pred)
            iteration_scores.append(mse)

    # print(weights)

    return iteration_scores, weights


# def eig(A):
#     # 计算特征值
#     eigenvalues = np.linalg.eigvals(A)
#
#     # 绘制特征值
#     plt.figure()
#     plt.scatter(eigenvalues.real, eigenvalues.imag, color='red', label='Eigenvalues')
#
#     # 绘制单位圆
#     circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--', label='Unit Circle')
#     plt.gca().add_artist(circle)
#
#     # 设置图形属性
#     plt.axhline(0, color='black', linewidth=0.5)
#     plt.axvline(0, color='black', linewidth=0.5)
#     plt.xlabel('Real Part')
#     plt.ylabel('Imaginary Part')
#     plt.title('Eigenvalues and Unit Circle')
#     plt.legend()
#     plt.grid()
#     plt.gca().set_aspect('equal', adjustable='box')
#     plt.xlim(-1, 1)
#     plt.ylim(-1, 1)
#
#     # 显示图形
#     plt.show()


if __name__ == '__main__':
    node = 100
    weights = [0.2, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0, 20.0]
    for i in weights:
        B, _, _, datapoints = get_sbm_2blocks_data(node, i)
        adjacency_matrix = incidence_to_adjacency(B)
        degrees = [sum(row) for row in adjacency_matrix]
        normalized_matrix = create_a_matrix(adjacency_matrix, degrees)
        count, ratio, avg, avg_no_diagonal = visualisation(normalized_matrix, node, i)
        # error, _ = consensus_innovation(datapoints, adjacency_matrix)
        print(f"weight: {i}")
        print(f"num of edge between blocks: {count}")
        print(f"ratio: {ratio}")
        print(f"average probability: {avg}")
        print(f"average probability except diagonal: {avg_no_diagonal}")
        # print(f"error: {error}")
        print("----------------------------------------------------------------")
    # eig(normalized_matrix)