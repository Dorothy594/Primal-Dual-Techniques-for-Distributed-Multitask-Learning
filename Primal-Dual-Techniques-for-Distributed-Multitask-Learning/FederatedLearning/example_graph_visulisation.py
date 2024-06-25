import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

K = 4
p_edge = 0.1

lambda_2 = 1
while lambda_2 > 0.99999999:
    C = np.eye(K)
    for k in range(K):
        for l in range(k+1, K):
            connected = np.random.binomial(1, p_edge)
            if connected == 1:
                C[l, k] = 1
                C[k, l] = 1

    n = C @ np.ones((K,))

    A = np.zeros((K, K))
    for k in range(K):
        for l in range(k+1, K):
            if C[k, l] == 1:
                A[k, l] = np.true_divide(1, np.max([n[k], n[l]]))
                A[l, k] = A[k, l]

    degrees = A @ np.ones((K,))
    for k in range(K):
        A[k, k] = 1 - degrees[k]

    eigs = np.linalg.eigvalsh(A)
    lambda_2 = eigs[-2]

# 创建图结构
G = nx.Graph()
for i in range(K):
    G.add_node(i)
for i in range(K):
    for j in range(i + 1, K):
        if C[i, j] == 1:
            G.add_edge(i, j)

# 绘制图
pos = nx.spring_layout(G)  # 选择一个布局
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=1500, edge_color='black', linewidths=1, font_size=15)
plt.title('Graph Visualization')
plt.show()
