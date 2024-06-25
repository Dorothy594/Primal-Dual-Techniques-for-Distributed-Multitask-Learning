import pandas as pd
import numpy as np


def is_strongly_connected(adj_matrix):
    # 使用深度优先搜索算法
    def dfs(node, visited):
        visited[node] = True
        for neighbor in range(len(adj_matrix)):
            if adj_matrix[node][neighbor] == 1 and not visited[neighbor]:
                dfs(neighbor, visited)

    # 从第一个节点开始进行深度优先搜索
    visited = [False] * len(adj_matrix)
    dfs(0, visited)

    # 如果有任何一个节点未被访问到，则返回 False
    if False in visited:
        return False
    else:
        return True


def is_doubly_stochastic(adj_matrix):
    # 检查行和列的和是否都等于1
    row_sum = np.sum(adj_matrix, axis=1)
    col_sum = np.sum(adj_matrix, axis=0)

    if np.allclose(row_sum, 1) and np.allclose(col_sum, 1):
        return True
    else:
        return False


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


def calculate_a_kl(k, l, matrix):
    if k != l:
        if matrix[k][l] == 1:
            return 1 / max(degrees[k], degrees[l])
        else:
            return 0
    else:
        # When k == l
        return 1 - sum(calculate_a_kl(k, i, matrix) for i in range(len(matrix)) if i != k and matrix[k][i] == 1)


# Create a new matrix to store all a_kl values
def create_a_matrix(matrix):
    size = len(matrix)
    a_matrix = [[0] * size for _ in range(size)]
    for k in range(size):
        for l in range(size):
            a_matrix[k][l] = calculate_a_kl(k, l, matrix)
    return a_matrix


# 从 Excel 文件读取数据
excel_filename = "B.xlsx"
df = pd.read_excel(excel_filename)

# 将 DataFrame 转换为 NumPy 数组
matrix_data = df.values
adjacency_matrix = incidence_to_adjacency(matrix_data)
degrees = [sum(row) for row in adjacency_matrix]
new_matrix = create_a_matrix(adjacency_matrix)


if is_strongly_connected(new_matrix):
    print("The graph is a strongly connected graph.")
else:
    print("The graph is NOT a strongly connected graph.")


if is_doubly_stochastic(new_matrix):
    print("Satisfy doubly stochastic.")
else:
    print("NOT satisfy doubly stochastic.")