import random
import numpy as np
from tqdm import tqdm

def find_root(clusters, i):
    if clusters[i] == i:
        return i
    return find_root(clusters, clusters[i])

def union(clusters, cluster_sizes, x, y):
    x_root = find_root(clusters, x)
    y_root = find_root(clusters, y)

    if x_root != y_root:
        if cluster_sizes[x_root] < cluster_sizes[y_root]:
            clusters[x_root] = y_root
            cluster_sizes[y_root] += cluster_sizes[x_root]
        elif cluster_sizes[x_root] > cluster_sizes[y_root]:
            clusters[y_root] = x_root
            cluster_sizes[x_root] += cluster_sizes[y_root]
        else:
            clusters[y_root] = x_root
            cluster_sizes[x_root] += cluster_sizes[y_root]

def site_percolation_cluster_distribution(nodes, edges, p):
    active_nodes = set()
    for node in nodes:
        if random.random() <= p:
            active_nodes.add(node)

    clusters = {}
    cluster_sizes = {}

    for u, v in edges:
        if u not in active_nodes or v not in active_nodes:
            continue

        if u not in clusters:
            clusters[u] = u
            cluster_sizes[u] = 1
        if v not in clusters:
            clusters[v] = v
            cluster_sizes[v] = 1
        union(clusters, cluster_sizes, u, v)

    cluster_size_distribution = []
    for key in clusters.keys():
        if clusters[key] == key:
            cluster_size_distribution.append(cluster_sizes[key])

    return cluster_size_distribution

def cont_bouchand_model(clusters, pa, pb):
    price_change = 0
    for cluster in clusters:
        if random.random() <= pa:
            if random.random() <= pb:
                price_change += cluster
            else:
                price_change -= cluster
    num_of_nodes = sum(cluster for cluster in clusters)
    return (price_change/num_of_nodes)

def bond_percolation_cluster_distribution_ignore_percolation(top_nodes, bottom_nodes, edges, p):
    clusters = {}
    cluster_sizes = {}

    for u, v in edges:
        if u not in clusters:
            clusters[u] = u
            cluster_sizes[u] = 1
        if v not in clusters:
            clusters[v] = v
            cluster_sizes[v] = 1
        if random.random() <= p:
            union(clusters, cluster_sizes, u, v)

    percolates = False
    for u in top_nodes:
        for v in bottom_nodes:
            if find_root(clusters, u) == find_root(clusters, v):
                percolates = True
                break

    cluster_size_distribution = []
    for key in clusters.keys():
        if clusters[key] == key:
            cluster_size_distribution.append(cluster_sizes[key])

    cluster_size_distribution.sort(reverse=True)
    if percolates:
        cluster_size_distribution.pop(0)

    return cluster_size_distribution

def cont_bouchand_model_ignore_percolation(top_nodes, bottom_nodes, edges, p, pa, pb):
    clusters = bond_percolation_cluster_distribution_ignore_percolation(top_nodes, bottom_nodes, edges, p)
    price_change = 0
    for cluster in clusters:
        coin1 = random.choices([0,1], [1 - pa, pa])[0]
        if coin1 == 1:
            coin2 = random.choices([0,1], [1 - pb, pb])[0]
            if coin2 == 1:
                price_change += cluster
            else:
                price_change -= cluster
    num_of_nodes = sum(cluster for cluster in clusters)
    return (price_change/num_of_nodes)

def basic_affine_jump_diffusion(
        mean,
        adjustment_speed,
        volatility,
        jump_intensity,
        jump_mean,
        total_time,
        time_step
    ):
    num_of_time_steps = int(total_time / time_step)
    time_vector = np.linspace(0., total_time, num_of_time_steps)

    n_jumps = np.random.poisson(lam = jump_intensity)
    jump_times = np.sort(np.random.uniform(low = 0, high = total_time, size = n_jumps))
    jump_sizes = np.random.exponential(scale = jump_mean, size = n_jumps)
    J = np.zeros(num_of_time_steps) # Jumps.
    for i in range(num_of_time_steps):
        J[i] = np.sum(jump_sizes[jump_times < time_vector[i]])
    dJ = np.insert(np.diff(J),0,0) # Jump increments.

    dW = np.sqrt(time_step) * np.random.randn(num_of_time_steps) # Brownian increments.
    x = np.zeros(num_of_time_steps)
    x[0] = mean
    for i in range(1, num_of_time_steps):
        x[i] = x[i - 1] + adjustment_speed * (mean - x[i - 1]) * time_step + volatility * np.sqrt(x[i - 1]) * dW[i - 1] + dJ[i - 1]
    return x

def dynamic_cont_couchand_model(edges, pa, pb, mean, adjustment_speed, volatility, jump_intensity, jump_mean, total_time, time_step):
    num_of_time_steps = int(total_time / time_step)
    price_changes = np.zeros(num_of_time_steps)
    edge_probabilities = basic_affine_jump_diffusion(mean, adjustment_speed, volatility, jump_intensity, jump_mean, total_time, time_step)
    for i in tqdm(range(len(edge_probabilities))):
        price_changes[i] = cont_bouchand_model(edges, edge_probabilities[i], pa, pb)
    return price_changes