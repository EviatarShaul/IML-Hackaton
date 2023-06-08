# Task 1.2.1
from typing import NoReturn

from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
    QuadraticDiscriminantAnalysis
import pandas as pd
import numpy as np

from project.task_1.code.hackathon_code.utils.preprocess import *


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


LABEL_NAME = "cancellation_datetime"
DATAFRAME_IMPORTANT_COLS = ["guest_is_not_the_customer",
                            "no_of_adults",
                            "no_of_children",
                            "no_of_extra_bed",
                            "no_of_room"]


def temp_classify_cancellation_prediction(raw_data: pd.DataFrame):
    train, test = train_test_split(raw_data, test_size=0.2)
    X_train, y_train = create_x_y_df(train, DATAFRAME_IMPORTANT_COLS,
                                     LABEL_NAME)
    X_test, y_test = create_x_y_df(test, DATAFRAME_IMPORTANT_COLS,
                                   LABEL_NAME)

    # TODO: this part should be done in the pre-processing part!
    # changing y_train: where 1 indicating that a cancellation is predicted, and 0 otherwise
    y_train = np.where(pd.Series(y_train).isnull(), 0, 1)
    y_test = np.where(pd.Series(y_test).isnull(), 0, 1)

    classify_cancellation_prediction(X_train, y_train, X_test, y_test)


def classify_cancellation_prediction(X_train, y_train, X_test, y_test):
    # defining models to predict:
    models = [
        LogisticRegression(),
        # LogisticRegression(penalty=f1_score),
        DecisionTreeClassifier(max_depth=5),
        KNeighborsClassifier(n_neighbors=5),
        SVC(kernel='linear', probability=True),
        LinearDiscriminantAnalysis(store_covariance=True),
        QuadraticDiscriminantAnalysis(store_covariance=True)
    ]
    model_names = ["Logistic regression", "Desicion Tree (Depth 5)", "KNN",
                   "Linear SVM", "LDA", "QDA"]

    # training regressors on the model:
    for i in range(len(models)):
        models[i].fit(X_train, y_train)
        pred = models[i].predict(X_test)
        model_f1_train_error = f1_score(y_test, pred)
        print(f"Model: {model_names[i]}:\n\tTrain Error: {model_f1_train_error}\n")


        # p = np.stack([y_test, pred], axis=1)
        # model_train_error = 1 - models[i].score(X_train, y_train)
        # model_test_error = 1 - models[i].fit(X_train, y_train).score(X_test, y_test)
        # print(f"Model: {model_names[i]}:\n\tTrain Error: {model_train_error}\n\tTest Error: {model_test_error}\n")


def write_answer(ids: pd.Series, predicted: np.ndarray, path: str):
    pass



