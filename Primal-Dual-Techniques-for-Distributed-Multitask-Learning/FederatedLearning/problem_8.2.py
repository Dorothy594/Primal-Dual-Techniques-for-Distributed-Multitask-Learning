import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import networkx as nx
import random

np.random.seed(0)

sigma_h_squared = 1
sigma_v_squared = 1
sigma_w_squared = 1
sigma_w_variation_squared = 1

M = 2
K = 4
N = 15

rho = 0.01

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

# Parameters
K = 2  # Number of blocks
n_per_block = 100  # Number of nodes per block
p_edge = 0.5  # Probability of an edge within blocks
p_out = 0.01  # Probability of an edge between blocks

# Generate block adjacency matrix
block_adjacency = np.zeros((K, K))
for i in range(K):
    for j in range(i, K):
        if i == j:
            block_adjacency[i, j] = p_edge
        else:
            block_adjacency[i, j] = p_out
            block_adjacency[j, i] = p_out

# Generate adjacency matrix for the whole graph
adjacency = np.zeros((K * n_per_block, K * n_per_block))
# for i in range(K):
#     for j in range(i, K):
#         start_i = i * n_per_block
#         end_i = (i + 1) * n_per_block
#         start_j = j * n_per_block
#         end_j = (j + 1) * n_per_block
#         adjacency[start_i:end_i, start_j:end_j] = np.random.binomial(1, block_adjacency[i, j], size=(n_per_block, n_per_block))
#         adjacency[start_j:end_j, start_i:end_i] = adjacency[start_i:end_i, start_j:end_j].T  # Symmetric


for i in range(K * n_per_block):
  for j in range(K * n_per_block):
    adjacency[i, j] = random.randint(0, 1)
    adjacency[j, i] = adjacency[i, j]

# Create empty graph
G = nx.Graph()

# Add nodes
G.add_nodes_from(range(K * n_per_block))

# Add edges based on adjacency matrix
for i in range(K * n_per_block):
    for j in range(i+1, K * n_per_block):
        if adjacency[i, j] == 1:
            G.add_edge(i, j)

# Define node colors based on block membership
node_colors = ['skyblue'] * (n_per_block * K)
for i in range(1, K):
    node_colors[i * n_per_block: (i + 1) * n_per_block] = ['salmon'] * n_per_block

# Calculate the degrees of nodes
degrees = [sum(row) for row in adjacency]


def modified_laplacian_rule(adjacency_matrix, eta_max=max(degrees)):
    n = adjacency_matrix.shape[0]
    a = np.zeros((n, n), dtype=float)

    for k in range(n):
        nk = np.sum(adjacency_matrix[k])  # Count of neighbors for node k
        row_sum = 0

        for l in range(n):
            if k != l and adjacency_matrix[k, l] == 1:
                a[k, l] = 1 / eta_max
            # if k == l:
            #   a[k, l] = 1 - (nk - 1) / eta_max  # Diagonal elements
            # elif adjacency_matrix[k, l] == 1:
            #   a[k, l] = 1 / eta_max  # Off-diagonal elements for neighbors
            row_sum += a[k, l]

        a[k, k] = 1 - row_sum  # Adjust diagonal to make row sum to 1

    return a

new_matrix = modified_laplacian_rule(adjacency)


iterations = 200
experiments = 1
# mu = 0.73
mu = 0.68

M = 2
K = 4
N = 15

w_ci = np.zeros((M, K, iterations, experiments))
psi_ci = np.zeros((M, K, iterations, experiments))

error_ci = np.zeros((K, iterations, experiments))

new_matrix = np.array(new_matrix)


for run in range(experiments):
    # Consensus + Innovations
    for k in range(K):
        for l in range(K):
            w_ci[:, k, 1, run] += new_matrix[l, k] * w_ci[:, l, 0, run]
    for k in range(K):
        w_ci[:, k, 1, run] += - np.true_divide(mu, N) * rho * w_ci[:, k, 0, run] + np.true_divide(mu, N) * (d_k[:, k] - np.dot(H_k[:, :, k].T, w_ci[:, k, 0, run]))

        error_ci[k, 0, run] = np.square(np.linalg.norm(w_ci[:, k, 0, run] - w_star))
        error_ci[k, 1, run] = np.square(np.linalg.norm(w_ci[:, k, 1, run] - w_star))

    for i in range(2, iterations):
        # Consensus + innovations
        for k in range(K):
            for l in range(K):
                w_ci[:, k, i, run] += new_matrix[l, k] * w_ci[:, l, i-1, run]
        for k in range(K):
            w_ci[:, k, i, run] += - np.true_divide(mu, N) * rho * w_ci[:, k, i-1, run] + np.true_divide(mu, N) * (d_k[:, k] - np.dot(H_k[:, :, k].T, w_ci[:, k, i-1, run]))

            error_ci[k, i, run] = np.square(np.linalg.norm(w_ci[:, k, i, run] - w_star))

learning_curve_ci = np.mean(error_ci, axis=(0, 2))
plt.figure()
plt.semilogy(range(iterations), learning_curve_ci, linewidth=2)
plt.xlabel('Iteration',fontsize=12,fontname='times new roman')
plt.ylabel('MSD in dB',fontsize= 12,fontname='times new roman' )
plt.title(f'Consensus+innovations, lr={mu}')
plt.xlim(0,iterations)
plt.legend()
plt.grid()
plt.show()