import numpy as np


class Perceptron:
    def __init__(self, n):
        self.W = np.random.random(n)
        self.bias = np.random.random()
        print self.bias

    def label(self, y):
        self.y = y

    def input(self, X):
        self.I = np.array(X)
        self.net = np.dot(self.W, X)# + self.bias
        #print self.W, X, self.net
        #print 

    def output(self):
        self.O = 1.0 / (1.0 + np.exp(-self.net))
        return self.O

    def update(self, dW, dB):
        self.W += dW
        self.bias += dB


class Layer:
    def __init__(self, n_p, n_in):
        self.P = []
        for i in range(n_p):
            self.P.append(Perceptron(n_in))

    def label(self, Y):
        for i in range(len(Y)):
            self.P[i].label(Y[i])

    def propagate(self, x):
        output = []
        for p in self.P:
            p.input(x)
            output.append(p.output())
        return output

    def backpropagate(self, err=[]):
        if err == []:
            for p in self.P:
                p.err = p.O*(1.0 - p.O)*(p.y - p.O)
                err.append( p.err )
            return err
        else:
            #print "hidden error:",sum(err), "\n"
            for p in self.P:
                #p.err = p.O*(1.0 - p.O)*(sum(err))
                p.err = p.O*(1.0 - p.O)*np.dot(err,p.W)

    def update(self):
        r = 0.001
        for p in self.P:
            dB = r*p.err
            dW = dB*p.O 
            p.update(dW, dB)

    def answer(self):
        g = []
        for p in self.P:
            if p.O < .5:
                g.append(0)
            else:
                g.append(1)
        return np.array(g)

    def show(self, Y):
        g = []
        for p in self.P:
            if p.O < .5:
                g.append(0.0)
            else:
                g.append(1.0)
        g = np.array(g)
        if np.array_equal(g,Y) or True:
            print g,Y

    def true(self):
        for p in self.P:
            print p.O,
        print


class Autoencoder:
    P = 3 
    N = 2**P

    def __init__(self):
        self.X = np.eye(self.N)
        self.Y = np.eye(self.N)
        self.H = Layer(self.P, self.N)
        self.O = Layer(self.N, self.P)

    def train(self):
        # While terminating condition is not satisfied
        for i in range(100000):
            tot_err = 0.0
            act_err = 0.0
            # For each training tuple X in D
            for j in range(len(self.X)): 
            #j = np.random.randint(0,7)
            #if True:
                # Program the true values into the output layer
                self.O.label(self.Y[j])
                # Propagate the inputs forward
                hidden_outputs = self.H.propagate(self.X[j])
                self.O.propagate(hidden_outputs)
                # Backpropagate the errors
                err = self.O.backpropagate(err=[])
                act_err += sum(err)
                self.H.backpropagate(err=err)
                # Update the weights and biases
                self.O.update()
                self.H.update()
                tot_err += sum(abs(self.Y[j]-self.O.answer()))
                # Print he outputs
                if i%1000 == 0:
                    #self.O.show(self.Y[j])
                    self.H.show(None)
            if i%1000 == 0:
                #print tot_err/8.
                print act_err
        

if __name__=="__main__":
    # Initialize all weights and biases in network
    ac = Autoencoder()
    # Train the network
    ac.train()
