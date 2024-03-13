# Primal Update
# %load algorithm/optimizer.py
import torch
import abc
import numpy as np

from abc import ABC


# The linear model which is implemented by pytorch
class TorchLinearModel(torch.nn.Module):
    def __init__(self, n):
        super(TorchLinearModel, self).__init__()
        self.linear = torch.nn.Linear(n, 1, bias=False)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


# The abstract optimizer model which should have model, optimizer, and criterion as the input
class Optimizer(ABC):
    def __init__(self, model, optimizer, criterion):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion

    @abc.abstractmethod
    def optimize(self, x_data, y_data, old_weight, regularizer_term):
        torch_old_weight = torch.from_numpy(np.array(old_weight, dtype=np.float32))
        self.model.linear.weight.data = torch_old_weight
        for iterinner in range(40):
            self.optimizer.zero_grad()
            y_pred = self.model(x_data)
            loss1 = self.criterion(y_pred, y_data)
            loss2 = 1 / (2 * regularizer_term) * torch.mean((self.model.linear.weight - torch_old_weight) ** 2)  # + 10000*torch.mean((model.linear.bias+0.5)**2)#model.linear.weight.norm(2)
            loss = loss1 + loss2
            loss.backward()
            self.optimizer.step()

        return self.model.linear.weight.data.numpy()


# The linear model in Networked Linear Regression section of the paper
class LinearModel:
    def __init__(self, degree, features, label):
        mtx1 = 2 * degree * np.dot(features.T, features).astype('float64')
        mtx1 += 1 * np.eye(mtx1.shape[0])
        mtx1_inv = np.linalg.inv(mtx1)

        mtx2 = 2 * degree * np.dot(features.T, label).T

        self.mtx1_inv = mtx1_inv
        self.mtx2 = mtx2

    def forward(self, x):
        mtx2 = x + self.mtx2
        mtx_inv = self.mtx1_inv

        return np.dot(mtx_inv, mtx2)


# The Linear optimizer in Networked Linear Regression section of the paper
class LinearOptimizer(Optimizer):

    def __init__(self, model):
        super(LinearOptimizer, self).__init__(model, None, None)

    def optimize(self, x_data, y_data, old_weight, regularizer_term):
        return self.model.forward(old_weight)


# The Linear optimizer model which is implemented by pytorch
class TorchLinearOptimizer(Optimizer):
    def __init__(self, model):
        criterion = torch.nn.MSELoss(reduction='mean')
        optimizer = torch.optim.RMSprop(model.parameters())
        super(TorchLinearOptimizer, self).__init__(model, optimizer, criterion)

    def optimize(self, x_data, y_data, old_weight, regularizer_term):
        return super(TorchLinearOptimizer, self).optimize(x_data, y_data, old_weight, regularizer_term)


# The Logistic optimizer model which is implemented by pytorch
class TorchLogisticOptimizer(Optimizer):
    def __init__(self, model):
        criterion = torch.nn.BCELoss(reduction='mean')
        optimizer = torch.optim.RMSprop(model.parameters())
        super(TorchLogisticOptimizer, self).__init__(model, optimizer, criterion)

    def optimize(self, x_data, y_data, old_weight, regularizer_term):
        return super(TorchLogisticOptimizer, self).optimize(x_data, y_data, old_weight, regularizer_term)

# Dual Update
# %load algorithm/penalty.py
import abc
import numpy as np

from abc import ABC


# The abstract penalty function which has a function update
class Penalty(ABC):
    def __init__(self, lambda_lasso, weight_vec, Sigma, n):
        self.lambda_lasso = lambda_lasso
        self.weight_vec = weight_vec
        self.Sigma = Sigma

    @abc.abstractmethod
    def update(self, new_u):
        pass


# The norm2 penalty function
class Norm2Pelanty(Penalty):
    def __init__(self, lambda_lasso, weight_vec, Sigma, n):
        super(Norm2Pelanty, self).__init__(lambda_lasso, weight_vec, Sigma, n)
        self.limit = np.array(lambda_lasso * weight_vec)

    def update(self, new_u):
        normalized_u = np.where(np.linalg.norm(new_u, axis=1) >= self.limit)
        new_u[normalized_u] = (new_u[normalized_u].T * self.limit[normalized_u] / np.linalg.norm(new_u[normalized_u], axis=1)).T
        return new_u


# The MOCHA penalty function
class MOCHAPelanty(Penalty):
    def __init__(self, lambda_lasso, weight_vec, Sigma, n):
        super(MOCHAPelanty, self).__init__(lambda_lasso, weight_vec, Sigma, n)
        self.normalize_factor = 1 + np.dot(2 * self.Sigma, 1/(self.lambda_lasso * self.weight_vec))

    def update(self, new_u):
        for i in range(new_u.shape[1]):
            new_u[:, i] /= self.normalize_factor

        return new_u


# The norm1 penalty function
class Norm1Pelanty(Penalty):
    def __init__(self, lambda_lasso, weight_vec, Sigma, n):
        super(Norm1Pelanty, self).__init__(lambda_lasso, weight_vec, Sigma, n)
        self.limit = np.array([np.zeros(n) for i in range(len(weight_vec))])
        for i in range(n):
            self.limit[:, i] = lambda_lasso * weight_vec

    def update(self, new_u):
        normalized_u = np.where(abs(new_u) >= self.limit)
        new_u[normalized_u] = self.limit[normalized_u] * new_u[normalized_u] / abs(new_u[normalized_u])
        return new_u

# Creat SBM Graph
from algorithm.optimizer import *
from torch.autograd import Variable


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
                model = TorchLinearModel(n)
                optimizer = TorchLinearOptimizer(model)
                features = Variable(torch.from_numpy(features)).to(torch.float32)
                label = Variable(torch.from_numpy(label)).to(torch.float32)

            else:

                model = LinearModel(node_degrees[i], features, label)
                optimizer = LinearOptimizer(model)
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


def get_consensus_innovation_MSE():
    iterations = 1000
    experiments = 1
    mu = 0.01

    w_ci = np.zeros((M, K, iterations, experiments))
    w_extra = np.zeros((M, K, iterations, experiments))
    w_tracking = np.zeros((M, K, iterations, experiments))

    psi_extra = np.zeros((M, K, iterations, experiments))
    g_tracking = np.zeros((M, K, iterations, experiments))

    error_ci = np.zeros((K, iterations, experiments))
    error_extra = np.zeros((K, iterations, experiments))
    error_tracking = np.zeros((K, iterations, experiments))

    for run in range(experiments):
        # Consensus + Innovations
        for k in range(K):
            for l in range(K):
                w_ci[:, k, 1, run] += B[l, k] * w_ci[:, l, 0, run]
        for k in range(K):
            w_ci[:, k, 1, run] += -np.true_divide(mu, N) * rho * w_ci[:, k, 0, run] + np.true_divide(mu, N) * (
                        d_k[:, k] - np.dot(H_k[:, :, k].T, w_ci[:, k, 0, run]))
            error_ci[k, 0, run] = np.square(np.linalg.norm(w_ci[:, k, 0, run] - w_star))

        # EXTRA
        for k in range(K):
            for l in range(K):
                w_extra[:, k, 1, run] += B[l, k] * w_extra[:, l, 0, run]
        for k in range(K):
            w_extra[:, k, 1, run] += -np.true_divide(mu, N) * rho * w_extra[:, k, 0, run] + np.true_divide(mu, N) * (
                        d_k[:, k] - np.dot(H_k[:, :, k].T, w_extra[:, k, 0, run]))
            w_extra[:, k, 1, run] = psi_extra[:, k, 1, run]
            error_extra[k, 0, run] = np.square(np.linalg.norm(w_extra[:, k, 0, run] - w_star))

        # NEXT
        for k in range(K):
            for l in range(K):
                w_tracking[:, k, 1, run] += B[l, k] * w_tracking[:, l, 0, run]
                g_tracking[:, k, 1, run] += B[l, k] * g_tracking[:, l, 0, run]
        for k in range(K):
            w_tracking[:, k, 1, run] += -np.true_divide(mu, N) * rho * w_tracking[:, k, 0, run] + np.true_divide(mu,
                                                                                                                 N) * (
                                                    d_k[:, k] - np.dot(H_k[:, :, k].T, w_tracking[:, k, 0, run]))
            g_tracking[:, k, 1, run] += - rho * w_tracking[:, k, 0, run] + (
                        d_k[:, k] - np.dot(H_k[:, :, k].T, w_tracking[:, k, 0, run]))

            error_tracking[k, 0, run] = np.square(np.linalg.norm(w_tracking[:, k, 0, run] - w_star))

        for i in range(1, iterations):
            # Consensus + Innovations
            for k in range(K):
                for l in range(K):
                    w_ci[:, k, i, run] += B[l, k] * w_ci[:, l, i - 1, run]

            for k in range(K):
                w_ci[:, k, i, run] += -np.true_divide(mu, N) * rho * w_ci[:, k, i - 1, run] + np.true_divide(mu, N) * (
                            d_k[:, k] - np.dot(H_k[:, :, k].T, w_ci[:, k, i - 1, run]))
                error_ci[k, i, run] = np.square(np.linalg.norm(w_ci[:, k, i, run] - w_star))

            # EXTRA
            for k in range(K):
                for l in range(K):
                    psi_extra[:, k, i, run] += B[l, k] * w_extra[:, l, i - 1, run]

            for k in range(K):
                w_extra[:, k, i, run] += psi_extra[:, k, i, run]
                psi_extra[:, k, i, run] += -np.true_divide(mu, N) * rho * w_extra[:, k, i - 1, run] + np.true_divide(mu,
                                                                                                                     N) * (
                                                       d_k[:, k] - np.dot(H_k[:, :, k].T, w_extra[:, k, i - 1, run]))
                w_extra[:, k, i, run] += psi_extra[:, k, i, run] - psi_extra[:, k, i - 1, run]
                error_extra[k, i, run] = np.square(np.linalg.norm(w_extra[:, k, i, run] - w_star))

            # NEXT
            for k in range(K):
                for l in range(K):
                    w_tracking[:, k, i, run] += B[l, k] * w_tracking[:, l, i - 1, run]
                    g_tracking[:, k, i, run] += B[l, k] * g_tracking[:, l, i - 1, run]
            for k in range(K):
                w_tracking[:, k, i, run] += g_tracking[:, k, i - 1, run]
                g_tracking[:, k, i, run] += -np.true_divide(mu, N) * rho * w_tracking[:, k, i - 1,
                                                                           run] + np.true_divide(mu, N) * (
                                                        d_k[:, k] - np.dot(H_k[:, :, k].T, w_tracking[:, k, i - 1,
                                                                                           run])) + np.true_divide(mu,
                                                                                                                   N) * rho * w_tracking[
                                                                                                                              :,
                                                                                                                              k,
                                                                                                                              i - 2,
                                                                                                                              run] - np.true_divide(
                    mu, N) * (d_k[:, k] - np.dot(H_k[:, :, k].T, w_tracking[:, k, i - 2, run]))

                error_tracking[k, i, run] = np.square(np.linalg.norm(w_tracking[:, k, i, run] - w_star))
    return w_ci, w_extra, w_tracking, error_ci, error_extra, error_tracking


# Two Clusters
# from sparsebm import generate_SBM_dataset
import networkx as nx


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


import random
import matplotlib.pyplot as plt
from collections import defaultdict


PENALTY_FUNCS = ['norm1', 'norm2', 'mocha']

LAMBDA_LASSO = {'norm1': 0.01, 'norm2': 0.01, 'mocha': 0.05}

sigma_h_squared = 1
sigma_v_squared = 1
sigma_w_squared = 1
sigma_w_variation_squared = 1

K = 2000
M = 2
N = 5

rho = 0

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

w_star = np.linalg.solve(H + rho*np.eye(M), d)


B, weight_vec, true_labels, datapoints = get_sbm_2blocks_data(pin=0.5, pout=0.01, is_torch_model=False)
E, N = B.shape


w_ci1, w_extra1, w_tracking1, error_ci1, error_extra1, error_tracking1 = get_consensus_innovation_MSE()
print(w_ci1, w_extra1, w_tracking1, error_ci1, error_extra1, error_tracking1)