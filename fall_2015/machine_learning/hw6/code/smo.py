import numpy as np


def K_rbf(x1, x2):
    pass        


def K_linear(x1, x2):
    return np.dot(x1,x2)


def dual(x, y, a, b, K=K_linear):
    return a*y*K(x,x) + b


def train(X, Y, C=0.01, tol=0.01, eps=1e-5, max_passes=3, K=K_linear):
    """
    Input:
        C: regularization parameter
        tol: numerical tolerance
        max-passes: max # of times to iterate over a's without changing
        {X,Y}: training data
    Output:
        a in R^m: Lagrange multipliers for solution
        b in R: threshold for solution
    """
    #print Y
    # Get parameter metadata
    m, d = len(X), len(X.T)
    # Initialize a_i = 0 for all i, b = 0
    a = np.zeros(m)
    b = 0
    # Initialize passes = 0
    passes = 0
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            # Calculate E_i using (2)
            E_i = dual(X[i], Y[i], a[i], b) - Y[i]
            # Horrific conditional statment
            if ((Y[i]*E_i < -tol and a[i] < C) or (Y[i]*E_i > tol and a[i] > 0)):
                # Select j != i randomly
                j = i
                while j == i:
                    j = np.random.randint(m)
                # Calculate E_j using (2)
                E_j = dual(X[j], Y[j], a[j], b) - Y[j]
                # Save old a's
                a_i, a_j = a[i], a[j]
                # Compute L and H by (10) or (11)
                if Y[i] == Y[j]:
                    L = max(0, a[i]+a[j]-C)
                    H = min(C, a[i]+a[j])
                else:
                    L = max(0, a[j]-a[i])
                    H = min(C, C+a[j]-a[i])
                # Compare L and H
                if L == H:
                    #print "L == H: continue"
                    continue
                # Compute n by (14)
                n = (2.0*K(X[i], X[j]) - K(X[i], X[i])) - K(X[j], X[j])
                # Compare n to threshold
                if n >= 0:
                    #print "n >= 0: continue"
                    continue
                # Compute new value for a_j using (12)
                a[j] -= Y[j]*(E_i-E_j)/n
                # Clip a_j using (15)
                a[j] = np.clip(a[j], L, H)
                # Compare a_j to its old value
                if (abs(a[j] - a_j)) < eps:
                    #print "(abs(a[j] - a_j)) < 1e-5: continue"
                    continue
                # Determine value for a_i using (16)
                a[i] += Y[i]*Y[j]*(a_j - a[j])
                # Compute b1 using (17)
                b1 = b - E_i - Y[i]*(a[i]-a_i)*K(X[i],X[i]) - Y[j]*(a[j]-a_j)*K(X[i],X[j])
                # Compute b2 using (18)
                b2 = b - E_j - Y[i]*(a[i]-a_i)*K(X[i],X[j]) - Y[j]*(a[j]-a_j)*K(X[j],X[j])
                # Compute b using (19)
                if 0 < a[i] < C:
                    b = b1
                elif 0 < a[j] < C:
                    b = b2
                else:
                    b = (b1+b2)/2.0
                # Update num_changed alphas
                num_changed_alphas += 1
        # Check if no alphas changed
        if num_changed_alphas == 0:
            passes += 1
            #print "Passes:",passes
        else:
            #print num_changed_alphas
            passes = 0
    w = np.zeros(d)
    for i in range(m):
        w += a[i]*Y[i]*X[i]
    return w, b


def hypothesis(X, w, b):
    m = len(X)
    h = np.zeros(m)
    for i in range(m):
        r = np.dot(w,X[i])+b
        if r > 0:
            h[i] = 1.0
    return h


def test(X, Y, w, b):
    c = 0
    m = len(X)
    for i in range(m):
        f = np.dot(w,X[i])+b 
        if f < 0 and Y[i] < 0 or f > 0 and Y[i] > 0:
            c += 1
    return float(c)/float(m)
