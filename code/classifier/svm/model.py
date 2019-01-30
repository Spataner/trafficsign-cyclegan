'''
Model operations for SVM based multi-class classifier.

Dominic Spata,
Real-Time Computer Vision,
Institut fuer Neuroinformatik,
Ruhr University Bochum.
'''


import sklearn.svm
import sklearn.metrics
import classifier.common.report as report


def train(features, labels, c, gamma):
    '''
    Trains a multi-class SVM model.

    Arguments:
        features -- The feature vector.
        labels   -- The label vector.
        c        -- C parameter for the SVM.
        gamma    -- Gamma parameter for the SVM.

    Returns:
        The model object.
    '''

    model = sklearn.svm.SVC(C = c, gamma = gamma)
    model.fit(features, labels)

    return model

def test(features, labels, model):
    '''
    Tests a model on a dataset and returns the zero-one-loss.

    Arguments:
        features    -- The feature vector.
        labels      -- The label vector.
        model       -- The model to test.

    Returns:
        The zero-one-loss or misclassification percentage.
    '''

    predicted_labels = model.predict(features)
    return report.Report(labels, predicted_labels, report.LABEL_NAMES)