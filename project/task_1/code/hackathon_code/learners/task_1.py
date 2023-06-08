# Task 1.2.1
from typing import NoReturn

from sklearn.ensemble import RandomForestClassifier
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
from project.task_1.code.hackathon_code.utils.csv_helper import *
from project.task_1.code.hackathon_code.utils.preprocess import *

LABEL_NAME = "cancellation_datetime"
DATAFRAME_IMPORTANT_COLS = ["guest_is_not_the_customer",
                            "no_of_adults",
                            "no_of_children",
                            "no_of_extra_bed",
                            "no_of_room",
                            "cost",
                            "time_from_booking_to_checkin",
                            "stay_duration",
                            "is_weekday",
                            "checkin_month_sin",
                            "checkin_month_cos",
                            "hotel_age",
                            "hotel_star_rating",
                            "special_requests"]


def temp_classify_cancellation_prediction(raw_data: pd.DataFrame):
    # TODO: remove!
    raw_data["special_requests"].replace(np.nan, 0, inplace=True)

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
        KNeighborsClassifier(n_neighbors=1),
        SVC(kernel='poly', probability=True, max_iter=50),
        LinearDiscriminantAnalysis(store_covariance=True),
        QuadraticDiscriminantAnalysis(store_covariance=True),
        RandomForestClassifier()
    ]
    model_names = ["Logistic regression", "Desicion Tree (Depth 5)", "KNN",
                   "Linear SVM",
                   "LDA", "QDA", "Random Forest"]

    # training regressors on the model:
    for i in range(len(models)):
        models[i].fit(X_train, y_train)
        pred = models[i].predict(X_test)
        model_f1_train_error = f1_score(y_test, pred)
        print(
            f"Model: {model_names[i]}:\n\tTrain Error: {model_f1_train_error}\n")

    helper_write_csv(None, pred, "agoda_cancellation_prediction.csv",
                     "cancellation")

"""
def classify_cancellation_prediction(X_train, y_train, X_test, y_test):
    # defining models to predict:
    models = [
        LogisticRegression(),
        # LogisticRegression(penalty=f1_score),
        DecisionTreeClassifier(max_depth=5),
        KNeighborsClassifier(n_neighbors=1),
        SVC(kernel='poly', probability=True, max_iter=50),
        LinearDiscriminantAnalysis(store_covariance=True),
        QuadraticDiscriminantAnalysis(store_covariance=True),
        RandomForestClassifier()
    ]
    model_names = ["Logistic regression", "Desicion Tree (Depth 5)", "KNN",
                   "Linear SVM",
                   "LDA", "QDA", "Random Forest"]

    # training regressors on the model:
    for i in range(len(models)):
        models[i].fit(X_train, y_train)
        pred = models[i].predict(X_test)
        model_f1_train_error = f1_score(y_test, pred)
        print(
            f"Model: {model_names[i]}:\n\tTrain Error: {model_f1_train_error}\n")

    helper_write_csv(None, pred, "agoda_cancellation_prediction.csv",
                     "cancellation")"""
