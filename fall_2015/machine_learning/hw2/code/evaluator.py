import numpy as np
import matplotlib.pyplot as plt

def logit(p):
    return np.log(p/(1-p))

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
                s = np.dot(Xi[j], Wi)
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

    def confusion(self):
        TP, FP, FN, TN = 0, 0, 0, 0
        for i in range(len(self.X)):
            # Test solution against the testing data
            Xi, Yi, Wi = self.X[i], self.Y[i], self.W[i]
            ktotal = len(Yi)
            for j in range(len(Xi)):
                s = np.dot(Xi[j], Wi)
                if s >= 0.5:
                    s = 1.0
                else:
                    s = 0.0
                if s == 1.0 and Yi[j] == 1.0:
                    TP += 1 
                elif s == 1.0 and Yi[j] == 0.0:
                    FP += 1
                elif s == 0.0 and Yi[j] == 1.0:
                    FN += 1
                elif s == 0.0 and Yi[j] == 0.0:
                    TN += 1
        print "TP:",TP,"FP:",FP
        print "TN:",TN,"FN:",FN
        #print "Correct:",str(kcorrect)+str('/')+str(ktotal)+' =',kacc

    def roc(self):
        X,Y,W = self.X[0], self.Y[0], self.W[0] 
        TPR_list, FPR_list = [], []
        h = np.array(sorted([np.dot(X[j], W) for j in range(len(X))]))
        inc = (h.max() - h.min())/100
        thresh = h.min()
        while thresh <= h.max():
            TP, FP, FN, TN = 0, 0, 0, 0
            for j in range(len(X)):
                s = np.dot(X[j], W)
                if s >= thresh:
                    s = 1.0
                else:
                    s = 0.0
                if s == 1.0 and Y[j] == 1.0:
                    TP += 1 
                elif s == 1.0 and Y[j] == 0.0:
                    FP += 1
                elif s == 0.0 and Y[j] == 1.0:
                    FN += 1
                elif s == 0.0 and Y[j] == 0.0:
                    TN += 1
            TPR = float(TP) / (float(TP)+float(FN)) 
            FPR = float(FP) / (float(FP)+float(TN)) 
            TPR_list.append(TPR), FPR_list.append(FPR)
            thresh += inc
        s = 0.0
        for i in range(1, len(FPR_list)):
            s += abs(FPR_list[i]-FPR_list[i-1])*(TPR_list[i]+TPR_list[i-1])
        print s/2.0
        return FPR_list, TPR_list
