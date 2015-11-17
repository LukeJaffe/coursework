from polluted import PollutedSpambase
from evaluator import Evaluator
from sklearn import linear_model
import sys
import argparse

if __name__=="__main__":
    # Parse cmdline arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', help='Method of regularization. {lasso, ridge}')
    args = parser.parse_args(sys.argv[1:])

    # Get data
    dataset = PollutedSpambase()
    train_data, train_labels = dataset.training()
    test_data, test_labels = dataset.testing()

    # Ridge
    if args.m == 'ridge':
        clf = linear_model.Ridge(alpha=10, max_iter = 1000000)
    elif args.m == 'lasso':
        clf = linear_model.Lasso(alpha=0.001, max_iter=1000000, selection='cyclic', positive=True, tol=2)
    clf.fit(train_data, train_labels)
    W = clf.coef_.ravel()
    print len(W[W>0])

    # Evaluate solution
    evaluator = Evaluator([test_data], [test_labels], [W])
    evaluator.accuracy()
