import algorithm.optimizer as opt
import torch
import numpy as np
from torch.autograd import Variable
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA


def get_sbm_data(cluster_sizes, G, W, m=1, n=2, noise_sd=0, is_torch_model=True):
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


def get_sbm_2blocks_data(node, weight, m=1, n=2, pin=0.5, pout=0.01, noise_sd=0, is_torch_model=True):
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
    cluster_sizes = [node, node]
    probs = np.array([[pin, pout], [pout, pin]])

    G = nx.stochastic_block_model(cluster_sizes, probs, seed=0)
    '''
    G: generated SBM graph with 2 clusters
    '''

    # define weight vectors for each cluster of the graph

    W1 = np.array([weight, 2])
    '''
    W1: the weigh vector for the first cluster
    '''
    W2 = np.array([-weight, 2])
    '''
    W2: the weigh vector for the second cluster
    '''

    W = [W1, W2]

    return get_sbm_data(cluster_sizes, G, W, m, n, noise_sd, is_torch_model)


def hyperplane_coefficient(data):
    """
    Plot a hyperplane fitted to the given 3D data along with a 3D scatter plot of data points.

    Parameters:
        data (array-like): 3D data points (x, y, z).
    """

    # Extract data points
    x = np.array(data[:, 0])
    y = np.array(data[:, 1])
    z = np.array(data[:, 2])

    # Fit a plane to the data using linear regression
    A = np.column_stack((x, y, np.ones_like(x)))
    coeff, _, _, _ = np.linalg.lstsq(A, z, rcond=None)
    # print(coeff)

    # Generate data points for the hyperplane
    xx, yy = np.meshgrid(np.linspace(min(x), max(x)), np.linspace(min(y), max(y)))
    zz = coeff[0] * xx + coeff[1] * yy + coeff[2]
    # print(f'xx: {xx}')
    # print(f'yy: {yy}')
    # print(f'zz: {zz}')

    return xx, yy, zz, coeff


def hyperplane_midpoint(x, y, z):
    # 计算超平面的中点
    midpoint_x = np.mean(x)
    midpoint_y = np.mean(y)
    midpoint_z = np.mean(z)

    return midpoint_x, midpoint_y, midpoint_z


# def visualize_graph(B, weight_vec):
#     G = nx.Graph()
#     edge_weights = {}
#
#     # Add edges to the graph
#     for i in range(len(B)):
#         u, v = np.nonzero(B[i])[0]
#         weight = weight_vec[i]
#         G.add_edge(u, v)
#         edge_weights[(u, v)] = weight
#
#     pos = nx.spring_layout(G)  # Layout for visualizing the graph
#
#     # Draw nodes
#     nx.draw_networkx_nodes(G, pos, node_size=200)
#
#     # Draw edges
#     nx.draw_networkx_edges(G, pos)
#
#     # Draw labels
#     nx.draw_networkx_labels(G, pos)
#
#     # Draw edge labels
#     # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_weights)
#
#     plt.title('SBM Graph Visualization')
#     plt.show()
#
#     return G


# def graph_analysis(G):
#     print(f"Density: {nx.density(G)}")
#     print(f"Degree: {nx.degree(G)}")
#     total_degree = 0
#     for i in nx.degree(G):
#         total_degree += i[1]
#     avg_degree = total_degree / len(nx.degree(G))
#     print(f"Average degree: {avg_degree}")
#     print(f"Clustering coefficient: {nx. average_clustering(G)}")
#     print(f"Transitivity: {nx.transitivity(G)}")


def angle_between_hyperplane(a, b):
    norm_a = a / np.linalg.norm(a)
    norm_b = b / np.linalg.norm(b)
    dot_product = np.dot(norm_a, norm_b)
    angle = np.arccos(np.clip(dot_product, -1.0, 1.0))

    return np.degrees(angle)


def distance_between_points(point1, point2):
    """
    Calculate the Euclidean distance between two points in 3D space.

    Parameters:
        point1 (tuple): Coordinates of the first point (x1, y1, z1).
        point2 (tuple): Coordinates of the second point (x2, y2, z2).

    Returns:
        float: Euclidean distance between the two points.
    """
    return np.linalg.norm(np.array(point1) - np.array(point2))


def angle_between_vectors(vector1, vector2):
    """
    Calculate the angle in degrees between two vectors.

    Parameters:
        vector1 (array-like): Coordinates of the first vector (x1, y1, z1).
        vector2 (array-like): Coordinates of the second vector (x2, y2, z2).

    Returns:
        float: Angle in degrees between the two vectors.
    """
    # Convert input vectors to numpy arrays
    v1 = np.array(vector1)
    v2 = np.array(vector2)

    # Calculate the dot product
    dot_product = np.dot(v1, v2)

    # Calculate the magnitudes of the vectors
    magnitude_v1 = np.linalg.norm(v1)
    magnitude_v2 = np.linalg.norm(v2)

    # Calculate the cosine of the angle
    cos_theta = dot_product / (magnitude_v1 * magnitude_v2)

    # Ensure the value is within the valid range for arccos (to handle numerical issues)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)

    # Calculate the angle in radians
    angle_rad = np.arccos(cos_theta)

    # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg


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
        return None  # 对角线元素的计算移到外部

def create_a_matrix(matrix, degrees):
    size = len(matrix)
    if size != len(degrees):
        raise ValueError("The size of the matrix and degrees array must be the same.")

    a_matrix = np.zeros((size, size))

    for k in range(size):
        for l in range(size):
            if k == l:
                # 计算对角线元素
                a_matrix[k][l] = 1 - sum(
                    1 / max(degrees[k], degrees[i]) for i in range(size) if i != k and matrix[k][i] == 1)
            else:
                a_matrix[k][l] = calculate_a_kl(k, l, matrix, degrees)

    return a_matrix

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


def intracluster_correlation_coefficient(B):
    adj = incidence_to_adjacency(B)
    degrees = [sum(row) for row in adj]
    norm = create_a_matrix(adj, degrees)

    # 提取子数组
    sub_array_1 = norm[:100, :100]
    # 展平成一维数组
    flattened_array_1 = sub_array_1.flatten()
    # 计算方差
    var_b = np.var(flattened_array_1)

    sub_array_2 = norm[100:, :100]
    flattened_array_2 = sub_array_2.flatten()
    var_w = np.var(flattened_array_2)

    rho = var_b / (var_b + var_w)

    return rho


def run(node, weight, pout):
    # Example usage
    B, weight_vec, _, datapoints = get_sbm_2blocks_data(node, weight, pout=pout)

    feature_1 = []
    feature_2 = []
    label = []
    for i in datapoints.keys():
        feature_1.append(datapoints[i]['features'][0][0])
        feature_2.append(datapoints[i]['features'][0][1])
        label.append(datapoints[i]['label'][0])

    feature_1 = np.array(feature_1)
    feature_2 = np.array(feature_2)
    label = np.array(label)

    data1 = np.column_stack((feature_1[:node], feature_2[:node], label[:node]))
    data2 = np.column_stack((feature_1[node:], feature_2[node:], label[node:]))
    xx1, yy1, zz1, coeff1 = hyperplane_coefficient(data1)
    x1, y1, z1 = hyperplane_midpoint(xx1, yy1, zz1)
    print("Hyperplane midpoint 1:", (x1, y1, z1))
    dx1, dy1, dz1 = hyperplane_midpoint(feature_1[:node], feature_2[:node], label[:node])
    print("Dataset midpoint 1:", (dx1, dy1, dz1))
    xx2, yy2, zz2, coeff2 = hyperplane_coefficient(data2)
    x2, y2, z2 = hyperplane_midpoint(xx2, yy2, zz2)
    print("Hyperplane midpoint 2:", (x2, y2, z2))
    dx2, dy2, dz2 = hyperplane_midpoint(feature_1[node:], feature_2[node:], label[node:])
    print("Dataset midpoint 2:", (dx2, dy2, dz2))

    hyper_angle = angle_between_hyperplane(coeff1, coeff2)
    print(f"angel between hyperplanes: {hyper_angle}")
    hyper_angles.append(hyper_angle)
    # print(f"hyper plane coefficient:\nxx1: {xx1}\nyy1: {yy1}\nzz1: {zz1}\nxx2: {xx2}\nyy2: {yy2}\nzz2: {zz2}")

    # ax = plt.axes(projection="3d")
    # # Creating plot
    # ax.scatter3D(feature_1[:node], feature_2[:node], label[:node], color="blue")
    # ax.scatter3D(feature_1[node:], feature_2[node:], label[node:], color='red')
    #
    # ax.set_xlabel("Feature 1")
    # ax.set_ylabel("Feature 2")
    # ax.set_zlabel("Label")
    # ax.set_title(f"weight = {weight}")
    # ax.plot_surface(xx1, yy1, zz1, alpha=0.5)
    # ax.plot_surface(xx2, yy2, zz2, alpha=0.5)
    # ax.scatter([x1], [y1], [z1], color='green', s=100, label='Midpoint 1')  # 绘制超平面的中点
    # ax.text(x1, y1, z1, f'  {x1:.2f}, {y1:.2f}, {z1:.2f}', color='green')  # 标注中点
    # ax.scatter([x2], [y2], [z2], color='green', s=100, label='Midpoint 2')
    # ax.text(x2, y2, z2, f'  {x2:.2f}, {y2:.2f}, {z2:.2f}', color='green')
    # plt.show()

    point1 = (x1, y1, z1)
    point2 = (x2, y2, z2)
    distance = distance_between_points(point1, point2)
    print(f"distance = {distance}")
    distances.append(distance)

    d_distance = distance_between_points((dx1, dy1, dz1), (dx2, dy2, dz2))
    print(f"dataset distance = {d_distance}")
    d_distances.append(d_distance)

    angle = angle_between_vectors(point1, point2)
    print(f"hyper angle = {angle}")
    angles.append(angle)

    d_angle = angle_between_vectors((dx1, dy1, dz1), (dx2, dy2, dz2))
    print(f"dataset angle = {d_angle}")
    d_angles.append(d_angle)

    # angle = anagle_between_hyperplane(xx1, yy1, zz1, xx2, yy2, zz2)
    # print(angle)

    ICC = intracluster_correlation_coefficient(B)
    print(f"intracluster correlation coefficient: {ICC}")
    icc.append(ICC)


if __name__ == '__main__':
    node = 100
    # weight = [0.2, 0.5, 1.0, 1.5, 2.0, 5.0, 10.0, 15.0, 20.0]
    weight = list(range(0, 201))
    # weight = [i * 0.01 for i in range(0, 101)]
    # weight = 2
    pout = 0.01
    # pout = [0.01, 0.1, 0.5]
    # pout = [i * 0.01 for i in range(0, 51)]
    distances = []
    angles = []
    hyper_angles = []
    d_distances = []
    d_angles = []
    icc = []
    for i in weight:
        print(f"weight = {i}")
        run(node, i, pout)
        print("---------------------------------------------------------------------------------")

    modify_angle = []
    for i in angles:
        if i > 90:
            modify_angle.append(180 - i)
        else:
            modify_angle.append(i)

    # 文件路径
    file_path = r'C:\Users\DELL\Desktop\20 Jun\pout-weight_error.xlsx'
    # 读取Excel文件的Sheet2工作表
    data = pd.read_excel(file_path, sheet_name='Sheet2')
    # 打印前几行数据以确认读取成功
    print(data.head())
    weights = data['weight']
    pouts = data['pout']

    # # Calculate the line of best fit
    # coefficients = np.polyfit(weight, distances, 1)  # 1st degree polynomial (linear regression)
    # polynomial = np.poly1d(coefficients)
    # regression_line = polynomial(weight)
    #
    # # Plot the data points
    # plt.plot(weight, distances, label='Data points')
    #
    # # Plot the regression line
    # plt.plot(weight, regression_line, label='Regression line')
    # plt.xlabel('Weight')
    # plt.ylabel('Distance')
    # plt.title('Midpoint Distance vs Weight')
    # plt.show()
    # plt.close()

    # plt.plot(weight, modify_angle)
    # plt.xlabel('Weight')
    # plt.ylabel('Angles')
    # plt.title('Midpoint Angles vs Weight')
    # plt.show()
    # plt.close()

    # plt.plot(weight, hyper_angles)
    # plt.xlabel('Weight')
    # plt.ylabel('Hyper Angles')
    # plt.title('Hyperplane Angles vs Weight')
    # plt.show()
    # plt.close()

    # plt.plot(weight, icc)
    # plt.xlabel('Weight')
    # plt.ylabel('ICC')
    # plt.title('Intracluster Correlation Coefficient vs Weight')
    # plt.show()
    # plt.close()


    fig, ax1 = plt.subplots()

    # 在第一个 y 轴上绘制 hyper_angles
    ax1.plot(weight[:5], hyper_angles[:5], label='Hyper Angles', color='royalblue')
    ax1.set_xlabel('Weight')
    ax1.set_ylabel('Hyper Angles')
    ax1.tick_params(axis='y')

    # 创建第二个 y 轴，并绘制 pouts
    ax2 = ax1.twinx()
    ax2.plot(weights[:5], pouts[:5], label='pouts', color='orange')
    ax2.set_ylabel('pouts')
    ax2.tick_params(axis='y')

    plt.axvline(x=2, color='black', linestyle='--')

    plt.show()