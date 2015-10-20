import numpy as np

class Gaussian:
    def __init__(self, mu, sigma, pi):
        self.mu = mu
        self.sigma = sigma
        self.pi = pi

    def evidence(self, x):
        c = (2*np.pi)**(float(len(self.mu))/2)
        det = np.linalg.det(self.sigma)**(0.5)
        p1 = 1.0/(c*det)
        d = (x-self.mu)[:,None]
        p3 = np.dot(d.T,np.linalg.pinv(self.sigma))
        p4 = -0.5*np.dot(p3,d)
        return p1*np.exp(p4)


class EMGM:
    def __init__(self, data_file, k):
        # Get the data
        dmat = []
        f = open(data_file, "r")
        for line in f:
            x = line.split()
            x = [float(e) for e in x]
            dmat.append(x)
        self.D = np.array(dmat)
        m,d = self.D.shape
        self.k = k

        # Initialize <Zim>
        self.Z = np.zeros((k,m), dtype=np.int)
        idx = np.random.randint(m,size=k)
        m = self.D[idx]
        a1 = np.dot(m,self.D.T)
        a2 = np.array([np.dot(m[i], m[i]) for i in range(k)]).T/2
        a3 = a1.T-a2
        for i,z in enumerate(self.Z.T):
            self.Z[a3[i].argmax()][i] = 1

        # Initialize model
        self.model = [None for i in range(k)]

    def maximization(self):
        # Initialize model parameters
        m,d = self.D.shape
        print
        for i in range(self.k):
            print sum(self.Z[i]),
            pi = float(sum(self.Z[i]))/float(m)
            mu = self.D[self.Z[i]].mean(0)
            sigma = np.zeros((d,d))
            for j in range(m):
                a = self.D[j]-mu
                sigma += np.dot(a[:,None],a[None,:]) 
            sigma /= float(m)
            self.model[i] = Gaussian(mu, sigma, pi)

    def expectation(self):
        m,d = self.D.shape
        for i in range(m):
            e = []
            for j in range(self.k):
                e.append( self.model[j].evidence(self.D[i]) )
            z = np.array([0 for idx in range(self.k)])
            # Calculate likelihood denominator
            e_sum = 0.0
            for j in range(self.k):
                e_sum += e[j]*self.model[j].pi
            for j in range(self.k):
                self.Z[j][i] = ((e[j]*self.model[j].pi)/e_sum)[0][0]
            # Reassign Zim based on the result
            #print z
            #for j in range(self.k):
            #    if j == z.argmax():
            #        self.Z[j][i] = 1
            #    else:
            #        self.Z[j][i] = 0
        for j in range(self.k):
            print sum(self.Z[j]),


if __name__=="__main__":
    fname1 = '../data/mog/2gaussian.txt'
    fname2 = '../data/mog/3gaussian.txt'
    emgm = EMGM(fname1, 2) 
    for i in range(1):
        emgm.maximization()
        emgm.expectation()
