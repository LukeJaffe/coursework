class Evaluator:
    def __init__(self, X, Y, W):
        self.X, self.Y, self.W = X, Y, W
            
    def MSE(self):
        mse = 0.0
        for i in range(len(self.X)):
            # Test solution against the testing data
            Xi, Yi, Wi = self.X[i], self.Y[i], self.W[i]
            ktotal = len(Yi)
            kmse = 0.0
            for j in range(len(Xi)):
                s = 0.0
                for e in range(len(Xi[j])):
                    # Basic predictor
                    s += Xi[j][e]*Wi[e]
                kmse += (Yi[j] - s)**2.0
            kmse /= float(ktotal)
            mse += kmse
        mse /= float(len(self.X))
        print "Average MSE:",mse

    def accuracy(self):
        acc = 0.0
        for i in range(len(self.X)):
            # Test solution against the testing data
            Xi, Yi, Wi = self.X[i], self.Y[i], self.W[i]
            ktotal = len(Yi)
            kcorrect = 0
            for j in range(len(Xi)):
                s = 0.0
                for e in range(len(Xi[j])):
                    # Basic predictor
                    s += Xi[j][e]*Wi[e]
                    # Logistic predictor
                    #s = 1.0 / (1.0 + exp(-s))
                if s >= 0.5:
                    s = 1.0
                else:
                    s = 0.0
                if s == Yi[j]:
                    kcorrect += 1
            kacc = float(kcorrect)/float(ktotal)
            acc += kacc
            #print "Correct:",str(kcorrect)+str('/')+str(ktotal)+' =',kacc
        acc /= float(len(self.X))
        print "Average accuracy:", acc
