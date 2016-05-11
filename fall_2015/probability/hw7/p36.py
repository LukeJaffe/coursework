import numpy as np
from math import sqrt
import matplotlib.pyplot as plt


def normalscatter(n):
    # Mean and covariance of X
    mean = np.array([0.0, 0.0])
    cov = np.array([[2.0, -1.5],[-1.5, 2.0]])

    # Generate X1, X2
    X = np.random.multivariate_normal(mean, cov, n)
    X1, X2 = X.T[0], X.T[1]

    # Do whitening transform
    w, v = np.linalg.eig(cov)
    Y = np.dot(v.T, X.T)
    coef = w**(-0.5)
    Z = coef*Y.T

    # Plot Z
    plt.scatter(Z.T[0], Z.T[1])
    plt.title("Whitened Gaussian RVs Z")
    plt.xlabel("Z1")
    plt.ylabel("Z2")
    plt.show()

    # Plot X
    plt.scatter(X1, X2)
    plt.title("Correlated Gaussian RVs X")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


if __name__=="__main__":
    normalscatter(5000)
