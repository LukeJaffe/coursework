import sys
import numpy as np
from sklearn import preprocessing
from svmutil import *

from load import load_mnist
from haar import sample, genbounds, genfeatures

if __name__=="__main__":
    # Load data using specialized script
    train_dataset = load_mnist(path="../data/mnist/", dataset="training")
    test_dataset = load_mnist(path="../data/mnist/", dataset="testing")

    # Take a fraction of the data to speed computation
    train_images, train_labels = sample(train_dataset, 10000)
    test_images, test_labels = test_dataset

    # Get the bounds of the haar rectangles
    bounds = genbounds(28, 28, 100)

    # Create data, using same rectangles for training and testing
    train_data = genfeatures(train_images, bounds)
    test_data = genfeatures(test_images, bounds)

    # Normalize the data
    zmscaler = preprocessing.StandardScaler()
    train_data = zmscaler.fit_transform(train_data)
    test_data = zmscaler.transform(test_data)

    # Train
    px = svm_problem(train_labels.ravel().tolist(), train_data.tolist())
    pm = svm_parameter()
    pm.kernel_type = LINEAR
    m = svm_train(px, pm)
    
    # Test
    p_label, p_acc, p_val = svm_predict(train_labels.ravel().tolist(), train_data.tolist(), m)
    h_label, h_acc, h_val = svm_predict(test_labels.ravel().tolist(), test_data.tolist(), m)
    
    # Eval
    ACC, MSE, SCC = evaluations(train_labels.ravel().tolist(), p_label)
    ACC, MSE, SCC = evaluations(test_labels.ravel().tolist(), h_label)
