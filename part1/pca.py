import numpy as np
def pca(X):
    u = np.array([np.average(X, axis=0)])
    n = X.shape[0]
    d = X.shape[1]

    # Sigma = covariance matrix
    Sigma = np.zeros(shape=(d, d))
    for i in range(n):
        Sigma += np.mat(np.transpose(X[i] - u)) * np.mat(X[i] - u)
    Sigma /= n

    # choose principal eigenvalues of Sigma (cov)
    (l, v) = np.linalg.eig(Sigma)
    tolerance = (10. ** -10)
    arg_list = np.argsort(l)
    arg_list = arg_list[::-1]
    principal_values = []
    for i in arg_list:
        if l[i] >= tolerance:
            principal_values.append(v[:,i])
    # format solution
    return np.transpose(np.vstack(principal_values))


# identidy D
def D(u, x, Sigma):
    first = np.mat(np.transpose(u - x)) * np.mat(np.linalg.inv(Sigma))
    second = np.mat((u - x))
    return np.sqrt(first * second).item(0)

# identity C
#def C(sigma):
    #d = sigma.shape[0]
    #two_pi = (2 * np.pi)
    #return np.sqrt((two_pi ** d) * np.linalg.det(sigma))

def C(Sigma):
    d = Sigma.shape[0]
    two_pi = (2 * np.pi)
    prod = 1
    for i in range(d):
        prod *= two_pi * Sigma[i][i]
    return np.sqrt(prod)

def p(x, u, Sigma):
    return 1./C(Sigma) * np.exp(-1./2. * (D(u, x, Sigma) ** 2))
def R(theta):
    return np.mat([[np.cos(theta), -np.sin(theta)],
                   [np.sin(theta), np.cos(theta)]])

# Sigma usually stdev(i, j) ** 2 * I for each index i, j
# Rotated sigma has todo with Lambda
Lambda = np.array([[2., 0.],
                  [0., 4.]])
#theta = np.pi / 4
#Sigma = R(theta) * np.mat(Lambda) * np.transpose(R(theta))
# u is the average of the x values
# sigma^2 is the max liklihood estimator for the variance
Sigma = Lambda
u = np.array([[8], [10]])
x = np.array([[8], [10]])
