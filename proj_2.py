from pca import pca as pca
import numpy as np

# note that my PCA might be a bit off

# returns (exemplars, exemplarIndices, clusterAssignments)
# X1     : (n x d) = data points
# init   : (k x d) = initial mediod guesses
# metric : function defining distance measure (distance between 2 points)
def kMedoids(X, init, metric):
    def cost(X, cluster_assignments, centroids):
        k = centroids.shape[0]
        return sum(
            [sum([metric(X[i], centroids[j]) for i in np.where(cluster_assignments == j)[0]])
            for j in range(k)])
    def mediod(cluster_j):
        min_index = cluster_j[0]
        min_sum = sum([metric(X[min_index], X[c_j]) for c_j in cluster_j])
        for i in range(cluster_j.size):
            index = cluster_j[i]
            new = sum([metric(X[index], X[c_j]) for c_j in cluster_j])
            if new < min_sum:
                min_sum = new
                min_index = index
        return min_index
    def setClusters():
        for i in range(n):
            min_x = float("inf")
            arg_min = -1
            for j in range(k):
                dist = metric(X[i], exemplars[j])
                if dist < min_x:
                    min_x = dist
                    arg_min = j
            cluster_assignments[i] = arg_min
       
            

    exemplars = np.copy(init)
    n = X.shape[0]
    k = exemplars.shape[0]
    exemplar_indices = np.zeros(k)
    cluster_assignments = np.array([0.] * n);
    last_cost = float("inf")
    setClusters()
    while cost(X, cluster_assignments, exemplars) != last_cost:
        last_cost = cost(X, cluster_assignments, exemplars)
        for j in range(k):
            cluster_j = np.where(cluster_assignments == j)[0]
            cluster_j_indices = np.where(cluster_assignments == j)
            exemplar_indices[j] = mediod(cluster_j)
            exemplars[j] = X[exemplar_indices[j]]
        setClusters()
   
    return (exemplars, exemplar_indices, cluster_assignments)
                           



# returns m x d array (high dimensional resonstruction of low dimensional data
# X : (n x d) = original data
# E : (d x d) = eiganvectors (columns)
# P : (m x c) = new vector
def reconstruction(X, E, P):
    n = X.shape[0]
    d = X.shape[1]
    c = P.shape[1]
    m = P.shape[0]

    # use X to compute the mean (origin)
    origin = ((float(1)/n) * sum([x for x in X]))

    # tot = reconstructed points?
    part1 = np.zeros((m, d))
    for i in range(m):
        for j in range(c):
            part1[i] += P[i][j] * E[:,j]
        
    return part1 + origin


# I am assuming that E arrives sorted from greatest to least eiganvalue
def projection(X, E, l):
    # E = axes
    # mean of X = center point
    num_points = X.shape[0]
    num_features = X.shape[1]
    origin = ((float(1)/num_points) * sum([x for x in X]))

    # rerepresent X in rotated coordinate frame by subtracting mean and
    # computing dot products with the eigenvectors
    new_coords = np.zeros((num_points, l))
    X = X - origin
    for i in range(num_points):
        for j in range(l):
            new_coords[i][j] = np.dot(X[i], E[:,j])
    return new_coords
    # discard vectors arrising from dot products corresponding to smallest
    # eigenvalues
    # I didnt do this b/c I am assuming E is sorted.


l2Sq = lambda x, y: np.linalg.norm(x - y)
X2 = np.array([[ -2.,  -2.], [ -1.,  -1.], [ -0.01,  -0.01], [  1.,
1.],  [  2.,   2.]])
kMedoids(X2, np.vstack([X2[0,:], X2[-1,:]]), l2Sq)
