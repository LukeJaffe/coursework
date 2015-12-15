import sys
import numpy as np
from sklearn import preprocessing

from load import load_mnist
from haar import sample, genbounds, genfeatures

from nn import knn, cosine_distance, gaussian_distance, polyd2_distance

if __name__=="__main__":
    # Load data using specialized script
    train_dataset = load_mnist(path="../data/mnist/", dataset="training")
    test_dataset = load_mnist(path="../data/mnist/", dataset="testing")

    # Take a fraction of the data to speed computation
    train_images, train_labels = sample(train_dataset, 5000)
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
    for d in [cosine_distance, gaussian_distance, polyd2_distance]:
        for k in [1,3,7]:
            H = knn(train_data, test_data, train_labels, d=d, k=k)
            c = np.sum(test_labels.ravel()==H)
            print "k=%d:" % k, float(c)/float(len(test_labels))
