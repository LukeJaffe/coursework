import sys
import argparse
import numpy as np
from sklearn import preprocessing
from kfolder import KFolder

from load import load_mnist
from haar import sample, genbounds, genfeatures
from nn import wnn, euclidean_distance, cosine_distance, gaussian_distance, polyd2_distance

parser = argparse.ArgumentParser()
parser.add_argument('-d', help='Dataset')
args = parser.parse_args(sys.argv[1:])

if args.d == "s":
    data_file = "../data/spambase/spambase.data"
    dmat = []
    f = open(data_file, "r")
    for line in f:
        x = line.split(',')
        x = [float(e) for e in x]
        dmat.append(x)
    data = np.array(dmat)

    # k-folds 
    folds = 10 
    kfolder = KFolder(data, folds, standard=True, shuffle=False)
    for i in range(1):
        print "Fold:", i+1
        # Get data and labels at fold k
        X,Y = kfolder.training(i)
        # Get the testing data
        Xi,Yi = kfolder.testing(i)

        # Run knn
        H = wnn(X, Xi, Y, d=euclidean_distance, r=4.6)
        c = np.sum(Yi.ravel()==H)
        print "r=%f:" % 4.6, float(c)/float(len(Yi))
elif args.d == "d":
    # Load data using specialized script
    train_dataset = load_mnist(path="../data/mnist/", dataset="training")
    test_dataset = load_mnist(path="../data/mnist/", dataset="testing")

    # Take a fraction of the data to speed computation
    train_images, train_labels = sample(train_dataset, 10000)
    test_images, test_labels = sample(test_dataset, 1000)

    # Get the bounds of the haar rectangles
    bounds = genbounds(28, 28, 100)

    # Create data, using same rectangles for training and testing
    train_data = genfeatures(train_images, bounds)
    test_data = genfeatures(test_images, bounds)

    # Normalize the data
    zmscaler = preprocessing.StandardScaler()
    train_data = zmscaler.fit_transform(train_data)
    test_data = zmscaler.transform(test_data)

    # Run knn
    H = wnn(train_data, test_data, train_labels, d=cosine_distance, r=0.325)
    c = np.sum(test_labels.ravel()==H)
    print "r=%f:" % 0.325, float(c)/float(len(test_labels))
else:
    pass

