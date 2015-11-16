from booster import Booster
from polluted import PollutedSpambase

if __name__=="__main__":
    dataset = PollutedSpambase()
    booster = Booster()
    train_data, train_labels = dataset.training()
    test_data, test_labels = dataset.testing()
    h_train, h_test = booster.boost(train_data, train_labels, test_data, booster.thresh(train_data), 10000, random=True)

    # Evaluate testing accuracy 
    c_test = len((h_test!=test_labels.ravel()).nonzero()[0])
    test_error = float(c_test)/float(len(test_labels))
    print "Test accuracy:", test_error
