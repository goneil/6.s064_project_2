import numpy as np

# x is n x d numpy array
# init is initial centroid guesses (k x d)
def kMeans(X, init):
    def distance(x1, x2):
        return np.linalg.norm(x1 - x2)
    def cost(X, cluster_assignments, centroids):
        k = centroids.shape[0]
        return sum(
            [sum([distance(X[i], centroids[j]) for i in np.where(cluster_assignments == j)[0]])
            for j in range(k)])

    centroids = np.copy(init)
    n = X.shape[0]
    k = centroids.shape[0]
    cluster_assignments = np.array([0.] * n);
    last_cost = float("inf")

    while cost(X, cluster_assignments, centroids) != last_cost:
        last_cost = cost(X, cluster_assignments, centroids)
        # find which x is closest to which centroid
        for i in range(n):
            min_x = float("inf")
            arg_min = -1
            for j in range(k):
                dist = distance(X[i], centroids[j])
                if dist < min_x:
                    min_x = dist
                    arg_min = j
            cluster_assignments[i] = arg_min
        for j in range(k):
            cluster_j = np.where(cluster_assignments == j)[0]
            centroids[j] = ((float(1)/cluster_j.size) * 
                           sum([X[i] for i in cluster_j]))

    return (centroids, cluster_assignments)

