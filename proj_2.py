from pca import pca as pca
import numpy as np

# note that my PCA might be a bit off

# I am assuming that E arrives sorted from greatest to least eiganvalue
def projection(X, E, l):
    # E = axes
    # mean of X = center point
    num_points = X.shape[0]
    num_features = X.shape[1]
    mean = ((float(1)/num_points) * sum([x for x in X]))

    # rerepresent X in rotated coordinate frame by subtracting mean and
    # computing dot products with the eigenvectors
    new_coords = np.zeros((num_points, l))
    X = X - mean
    for i in range(num_points):
        for j in range(l):
            new_coords[i][j] = np.dot(X[i], E[:,j])
    return new_coords
    # discard vectors arrising from dot products corresponding to smallest
    # eigenvalues
    # I didnt do this b/c I am assuming E is sorted.




X1 = np.array([[ -2.,  -2., 0., 1.], [ -1.,  -1., 1., 2.], [ -0.01,  -0.01, 2.,
3.]])

projection(X1, pca(X1), 1)
