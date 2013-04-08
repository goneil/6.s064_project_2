import numpy as np
# returns n x m array in which each column is an eiganvector
# returns small eigenvectors (u_i)
def pcahd(X):
    n = X.shape[0]
    mean = np.array(np.average(X, axis=0))
    D = X - mean
    D_t = np.transpose(D)
    D = np.mat(D)
    D_t = np.mat(D_t)

    mat = np.zeros(shape=(n, n))
    mat += (D * D_t)

    (l, u_i) = np.linalg.eig(mat)

    tolerance = (10. ** -10)
    arg_list = np.argsort(l)
    arg_list = arg_list[::-1]
    principal_values = []
    for i in arg_list:
        if l[i] >= tolerance:
            principal_values.append(u_i[:,i])

    # format solution
    return np.transpose(np.vstack(principal_values))
