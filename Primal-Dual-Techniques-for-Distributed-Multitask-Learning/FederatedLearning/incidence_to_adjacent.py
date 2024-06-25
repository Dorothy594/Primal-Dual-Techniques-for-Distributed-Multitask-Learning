import numpy as np
import pandas as pd


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


# 从 Excel 文件读取数据
excel_filename = "B_5.xlsx"
df = pd.read_excel(excel_filename)

# 将 DataFrame 转换为 NumPy 数组
matrix_data = df.values

# 打印矩阵数据
print("从 Excel 文件读取到的矩阵数据：")
print(matrix_data)

adjacent = incidence_to_adjacency(matrix_data)
print(adjacent)