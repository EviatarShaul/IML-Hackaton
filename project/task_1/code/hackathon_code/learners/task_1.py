# Task 1.2.1
from typing import NoReturn

import joblib
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
import plotly.graph_objects as go

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


def temp_classify_cancellation_prediction(raw_data, validate, test):
    X_train, y_train = create_x_y_df(raw_data, DATAFRAME_IMPORTANT_COLS,
                                     LABEL_NAME)
    X_val, y_val = create_x_y_df(validate, DATAFRAME_IMPORTANT_COLS,
                                 LABEL_NAME)
    X_test, y_test = create_x_y_df(test, DATAFRAME_IMPORTANT_COLS,
                                   LABEL_NAME)

    # TODO: this part should be done in the pre-processing part!
    # changing y_train: where 1 indicating that a cancellation is predicted, and 0 otherwise
    y_train = np.where(pd.Series(y_train).isnull(), 0, 1)
    y_val = np.where(pd.Series(y_val).isnull(), 0, 1)
    y_test = np.where(pd.Series(y_test).isnull(), 0, 1)

    print("when picking randomly - error is: " +
          str(1 - f1_score(y_test,
                           np.round(np.random.random(y_test.shape[0])))))
    print("when picking randomly - error is: " +
          str(1 - f1_score(y_test, np.ones(y_test.shape[0]))))

    # classifier = KNeighborsClassifier
    # display_errors(X_test, X_train, X_val, classifier, list(range(1, 10)), y_test,
    #                y_train, y_val, "k-nn")
    #
    # classifier = RandomForestClassifier
    # display_errors(X_test, X_train, X_val, classifier, list(range(5, 25)), y_test,
    #                y_train, y_val, "Random Forest")

    classify_cancellation_prediction(X_train, y_train, X_test, y_test)


def display_errors(X_test, X_train, X_val, classifier, k_range, y_test,
                   y_train, y_val, name):
    train_errors, val_errors, test_errors = [], [], []
    for k in k_range:
        model = classifier(k).fit(X_train, y_train)
        train_errors.append(1 - f1_score(y_train, model.predict(X_train)))
        val_errors.append(1 - f1_score(y_val, model.predict(X_val)))
        test_errors.append(1 - f1_score(y_test, model.predict(X_test)))
    val_errors = np.array(val_errors)
    min_ind = np.argmin(val_errors)
    selected_k = np.array(k_range)[min_ind]
    selected_error = val_errors[min_ind]
    mean, std = val_errors, np.std(val_errors, axis=0)
    go.Figure([
        go.Scatter(name='Train Error', x=k_range, y=train_errors,
                   mode='markers+lines', marker_color='rgb(152,171,150)'),
        go.Scatter(name='Mean Validation Error', x=k_range, y=mean,
                   mode='markers+lines', marker_color='rgb(220,179,144)'),
        go.Scatter(name='Test Error', x=k_range, y=test_errors,
                   mode='markers+lines', marker_color='rgb(25,115,132)'),
        go.Scatter(name='Selected Model', x=[selected_k], y=[selected_error],
                   mode='markers',
                   marker=dict(color='darkred', symbol="x", size=10))
    ]).update_layout(
        title=name,
        xaxis_title=r"k",
        yaxis_title=r"$\text{f1_score}$").show()


def classify_cancellation_prediction(X_train, y_train, X_test, y_test):
    # defining models to predict:
    models = [
        LogisticRegression(),
        # LogisticRegression(penalty=f1_score),
        DecisionTreeClassifier(max_depth=5),
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=3),
        SVC(kernel='poly', probability=True, max_iter=50),
        LinearDiscriminantAnalysis(store_covariance=True),
        QuadraticDiscriminantAnalysis(store_covariance=True),
        RandomForestClassifier(n_estimators=7)
    ]
    model_names = ["Logistic regression", "Desicion Tree (Depth 5)", "KNN(1)",
                   "KNN(3)",
                   "Linear SVM",
                   "LDA", "QDA", "Random Forest (7)","What changed?"]
    all_methods_scores = pd.DataFrame()
    errors = []
    # training regressors on the model:
    for i in range(len(models)):
        models[i].fit(X_train, y_train)
        pred = models[i].predict(X_test)
        model_f1_train_error = 1 - f1_score(y_test, pred)
        errors.append(model_f1_train_error)
        print(
            f"Model: {model_names[i]}:\n\tTrain Error: {model_f1_train_error}\n")

    errors.append("this is what changed")
    d = {}
    for i in range(len(errors)):
        d[model_names[i]] = [errors[i]]
    temp_df = pd.DataFrame(d)
    df: pd.DataFrame = joblib.load('errors_df.sav')
    df = pd.concat([df, temp_df])
    joblib.dump(df, 'errors_df.sav')

    print(df)


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
