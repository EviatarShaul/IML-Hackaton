# Task 1.2.1
from typing import NoReturn

from sklearn.metrics import f1_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
import pandas as pd
import numpy as np
LABEL_NAME = "cancellation_datetime"


class AdaBoost:
    """
    class

    Attributes
    ----------
    self.var_: type
        notes
    """

    def __init__(self, a):
        """
        Instantiate an ...

        Parameters
        ----------
        a: type
            notes
        """
        pass
        # super().__init__()
        # self.wl_ = wl

    def fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        Fit

        Parameters
        ----------
        a: type
            notes
        """
        pass

    def predict(self, X):
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples
        """
        pass

    def loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under ... loss

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under ... loss function
        """
        pass


def test_classify_cancellation_prediction(train, test):
    X_test, y_test = test.loc[:, test.columns != LABEL_NAME].values, test[LABEL_NAME].values
    pred = classify_cancellation_prediction(train, X_test)


def classify_cancellation_prediction(train: pd.DataFrame, to_predict):
    # defining models to predict:
    models = [
        LogisticRegression(penalty=f1_score),
        DecisionTreeClassifier(max_depth=5),
        KNeighborsClassifier(n_neighbors=5),
        SVC(kernel='linear', probability=True),
        LinearDiscriminantAnalysis(store_covariance=True),
        QuadraticDiscriminantAnalysis(store_covariance=True)
    ]
    model_names = ["Logistic regression", "Desicion Tree (Depth 5)", "KNN",
                   "Linear SVM", "LDA", "QDA"]

    # splitting the pre-processed train data into X_train (samples) and y_train (labels)
    X_train, y_train = train.loc[:, train.columns != LABEL_NAME].values, train[LABEL_NAME].values

    # TODO: this part should be done in the pre-processing part!
    # changing y_train: where 1 indicating that a cancellation is predicted, and 0 otherwise
    y_train = np.where(pd.Series(y_train).isnull(), 0, 1)

    # TODO: testing logistic regression:
    return models[0].fit(X_train, y_train).predict(to_predict)

    # training regressors on the model:
    model_errors = []
    for i in range(len(models)):
        predicted = models[i].fit(X_train, y_train).predict(to_predict)

    return None





