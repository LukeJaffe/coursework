from math import factorial
import matplotlib.pyplot as plt

inf = int(1e3)

def f(x):
    return float(x)

def product(iterable):
    prod = 1
    for n in iterable:
        prod *= n
    return prod

def npr(n, r):
    assert 0 <= r <= n
    return product(range(n - r + 1, n + 1))

def ncr(n, r):
    assert 0 <= r <= n
    if r > n // 2:
        r = n - r
    return npr(n, r) // factorial(r)

def rolodex(n):
    so = 0.0
    for r in range(n, inf):
        si = 0.0
        for i in range(n):
            p1 = f(ncr(n,i))
            p2 = f((-1)**i)
            p3 = (1.0 - f((i+1))/f(n))**f((r-1))
            si += p1*p2*p3
        so += f(r)*si 
    return so

if __name__=="__main__":
    X,Y = [],[]
    for i in range(1,51):
        X.append(i)
        Y.append(rolodex(i))
    plt.scatter(X, Y)
    plt.axis([0,50,0,250])
    plt.grid()
    plt.title('Average number of Rolodex tries to call all students at least once')
    plt.xlabel('Number of students in the class')
    plt.ylabel('Expected number of Rolodex tries')
    plt.show()
