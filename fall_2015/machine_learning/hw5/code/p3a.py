from polluted import PollutedSpambase
from evaluator import Evaluator
from descent import GradientDescent

if __name__=="__main__":
    # Get data
    dataset = PollutedSpambase()
    train_data, train_labels = dataset.training()
    test_data, test_labels = dataset.testing()

    # Do Logistic Regression
    gd = GradientDescent(train_data, train_labels)
    # 200,000 iterations gives ~85% acc
    W = gd.logreg_stoch(it=200001)

    # Evaluate solution
    evaluator = Evaluator([test_data], [test_labels], [W])
    evaluator.accuracy()
