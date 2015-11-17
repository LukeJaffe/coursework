from polluted import PollutedSpambase
from evaluator import Evaluator
from descent import GradientDescent
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

    # Do ridge regularization
    gd = GradientDescent(train_data, train_labels)
    W = gd.logreg_ridge_stoch(it=5000001)

    # Evaluate solution
    evaluator = Evaluator([test_data], [test_labels], [W])
    evaluator.accuracy()

