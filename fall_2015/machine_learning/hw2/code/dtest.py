import numpy as np

eps = 1e-4

def g(z):
    return 1.0 / (1.0 + np.exp(-z))

def d(z):
    return g(z)*(1.0-g(z))

def dg(z):
    top = g(z+eps) - g(z-eps)
    bot = 2.0*eps
    return top/bot

print d(.01), dg(.01)
print d(.1), dg(.1)
print d(1.0), dg(1.0)
print d(10.0), dg(10.0)
