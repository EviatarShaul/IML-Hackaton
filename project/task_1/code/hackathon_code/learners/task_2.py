# Task 1.2.2
import plotly.graph_objects as go
from typing import NoReturn
import sklearn.linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, \
    QuadraticDiscriminantAnalysis
import pandas as pd
import numpy as np

import project.task_1.code.hackathon_code.utils.model_helper
from project.task_1.code.hackathon_code.utils.csv_helper import *
from project.task_1.code.hackathon_code.utils.preprocess import *
from project.task_1.code.hackathon_code.utils.model_helper import *
import joblib

MODEL_SAVE_PATH = "task_1/code/hackathon_code/task_2_model_weights.sav"
MODEL_LOAD_PATH = "task_1/code/hackathon_code/task_2_model_weights.sav"

TASK_1_LABEL_NAME = "cancellation_datetime"
TASK_1_DATAFRAME_IMPORTANT_COLS = ["guest_is_not_the_customer",
                                   "no_of_adults",
                                   "no_of_children",
                                   "no_of_extra_bed",
                                   "no_of_room",
                                   "time_from_booking_to_checkin",
                                   "stay_duration",
                                   "is_weekday",
                                   "checkin_month_sin",
                                   "checkin_month_cos",
                                   "hotel_age",
                                   "hotel_star_rating",
                                   "special_requests"]

LABEL_NAME = "original_selling_amount"
DATAFRAME_IMPORTANT_COLS = ["guest_is_not_the_customer",
                            "no_of_adults",
                            "no_of_children",
                            "no_of_extra_bed",
                            "no_of_room",
                            "original_selling_amount",
                            "time_from_booking_to_checkin",
                            "stay_duration",
                            "is_weekday",
                            "checkin_month_sin",
                            "checkin_month_cos",
                            "hotel_age",
                            "hotel_star_rating",
                            # "cancellation_datetime",  # TODO: uncomment!
                            "special_requests"]

CATEGORICAL_COLS = ["charge_option_", "accommadation_type_name", "original_payment_type_"]

DAYS_IDX = 0
REL_COST_IDX = 1
POLICY_COL = "cancellation_policy_code"


def cancellation_fit(raw_data: pd.DataFrame, temp) -> RandomForestClassifier:
    for col in raw_data.columns:
        for prefix in CATEGORICAL_COLS:
            if col.startswith(prefix):
                TASK_1_DATAFRAME_IMPORTANT_COLS.append(col)

    X_train, y_train = create_x_y_df(raw_data, TASK_1_DATAFRAME_IMPORTANT_COLS,
                                     TASK_1_LABEL_NAME)

    print(set(temp.columns) - set(X_train.columns))
    # changing y_train: where 1 indicating that a cancellation is predicted, and 0 otherwise
    y_train = np.where(pd.Series(y_train).isnull(), 0, 1)

    model = RandomForestClassifier(11)
    model.fit(X_train, y_train)
    return model


def explore_predict_selling_amount(raw_data: pd.DataFrame, validate: pd.DataFrame, test: pd.DataFrame):
    # Adding dummies to important features list
    for col in raw_data.columns:
        for prefix in CATEGORICAL_COLS:
            if col.startswith(prefix):
                DATAFRAME_IMPORTANT_COLS.append(col)

    X_train, y_train = create_x_y_df(raw_data, DATAFRAME_IMPORTANT_COLS,
                                     LABEL_NAME)
    X_val, y_val = create_x_y_df(validate, DATAFRAME_IMPORTANT_COLS,
                                 LABEL_NAME)
    X_test, y_test = create_x_y_df(test, DATAFRAME_IMPORTANT_COLS,
                                   LABEL_NAME)

    cancel_model: RandomForestClassifier = cancellation_fit(raw_data, X_test)
    cancel_pred = cancel_model.predict(X_test.drop(columns=[LABEL_NAME]))

    # TODO: this part should be done in the INTERNAL pre-processing part!
    # changing y_train: where 1 indicating that a cancellation is predicted, and 0 otherwise
    # y_train = np.where(pd.Series(y_train).isnull(), 0, 1)
    # y_val = np.where(pd.Series(y_val).isnull(), 0, 1)
    # y_test = np.where(pd.Series(y_test).isnull(), 0, 1)

    # TODO: f1_score of picking randomly and of choosing all 0

    print("when picking randomly - error is: " +
          str(mean_squared_error(y_test,
                                 np.round(np.random.random(y_test.shape[0]) * y_train.max()), squared=False)))
    print("when picking all 0 - error is: " +
          str(mean_squared_error(y_test, np.ones(y_test.shape[0]) * y_train.mean(), squared=False)))

    model = AdaBoostClassifier()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print(mean_squared_error(y_test, pred))

    # classifier = RandomForestClassifier
    # display_errors(X_test, X_train, X_val, classifier, list(range(1, 15)), y_test,
    #                y_train, y_val, r"Random Forest Classifier - "
    #                                r"Number of classifiers as a function of f1 score "
    #                                r"on train\validation\test data")

    # TODO: older classifiers (lesser than randomForest):
    # classifier = KNeighborsClassifier
    # display_errors(X_test, X_train, X_val, classifier, list(range(1, 20)), y_test,
    #                y_train, y_val, "k-nn")
    # classifier = lambda x: DecisionTreeClassifier(max_depth=x)
    # display_errors(X_test, X_train, X_val, classifier, list(range(1, 25)), y_test,
    #                y_train, y_val, "Decision Tree")

    # TODO: older version that uses simpler classifiers:
    # classify_cancellation_prediction(X_train, y_train, X_test, y_test)

    result = np.where(cancel_pred == 1, -1, pred)
    print(mean_squared_error(y_test, result))
    return result


def display_errors(X_test, X_train, X_val, classifier, k_range, y_test,
                   y_train, y_val, name):
    # Calculating error for each k in k_range on train, validate and test sets
    train_errors, val_errors, test_errors = [], [], []
    for k in k_range:
        model = classifier(k).fit(X_train, y_train)
        train_errors.append(mean_squared_error(y_train, model.predict(X_train), squared=False))
        val_errors.append(mean_squared_error(y_val, model.predict(X_val), squared=False))
        test_errors.append(mean_squared_error(y_test, model.predict(X_test), squared=False))

    # finding the argument of the maximal value
    val_errors = np.array(val_errors)
    max_ind = np.argmax(val_errors)
    selected_k = np.array(k_range)[max_ind]
    selected_error = val_errors[max_ind]

    # displaying the graph of the f1 score on the train\validate\test sets
    go.Figure([
        go.Scatter(name='Train Score', x=k_range, y=train_errors,
                   mode='markers+lines', marker_color='rgb(152,171,150)'),
        go.Scatter(name='Validation Score', x=k_range, y=val_errors,
                   mode='markers+lines', marker_color='rgb(220,179,144)'),
        go.Scatter(name='Test Score', x=k_range, y=test_errors,
                   mode='markers+lines', marker_color='rgb(25,115,132)'),
        go.Scatter(name='Selected Model', x=[selected_k], y=[selected_error],
                   mode='markers',
                   marker=dict(color='darkred', symbol="x", size=10))
    ]).update_layout(
        title=r"$\text{" + name + r"}$",
        xaxis_title=r"$\text{Number of base estimators}$",
        yaxis_title=r"$\text{f1 macro score}$").show()

    # After running 10-20 iterations, we saw that for k= ~11-13 we
    # maximize the f1 macro score:
    # TODO: Save classifier weights:
    # model = classifier(11).fit(X_train, y_train)
    # save_model(model, MODEL_SAVE_PATH)


def classify_cancellation_prediction(X_train, y_train, X_test, y_test):
    # defining models to predict:
    models = [
        LogisticRegression(max_iter=35),
        DecisionTreeClassifier(max_depth=5),
        KNeighborsClassifier(n_neighbors=1),
        KNeighborsClassifier(n_neighbors=3),
        # SVC(kernel='linear', probability=True, max_iter=5),
        LinearDiscriminantAnalysis(store_covariance=True),
        # QuadraticDiscriminantAnalysis(store_covariance=True),
        RandomForestClassifier(n_estimators=7)
    ]
    model_names = ["Logistic regression",
                   "Desicion Tree (Depth 5)", "KNN(1)",
                   "KNN(3)",
                   # "Linear SVM",
                   "LDA",
                   # "QDA",
                   "Random Forest (7)", "What changed?"]
    errors = []

    # training regressors on the model:
    for i in range(len(models)):
        models[i].fit(X_train, y_train.astype(int))
        pred = models[i].predict(X_test)
        model_train_error = mean_squared_error(y_test, pred, squared=False)
        errors.append(model_train_error)
        print(
            f"Model: {model_names[i]}:\n\tTrain Error: {model_train_error}\n")

    errors.append("added dummies")
    d = {}
    for i in range(len(errors)):
        d[model_names[i]] = [errors[i]]
    temp_df = pd.DataFrame(d)
    df: pd.DataFrame = joblib.load('errors_df.sav')
    df = pd.concat([df, temp_df])
    joblib.dump(df, 'errors_df.sav')

    print(df.to_string())


def parse_policies(policies: List[str], row: pd.Series) -> List:
    res = [[np.inf, 0]]
    for i in range(len(policies)):
        d_ind = policies[i].find("D")
        if d_ind != -1:  # if there is a D in the policy, it is a cancellation fee
            days = int(policies[i][:d_ind])
            rel_cost = int(policies[i][d_ind + 1:-1]) / 100 if policies[i][-1] == "P" else int(
                policies[i][d_ind + 1:-1]) / row['stay_duration']
            res.append([days, rel_cost])
    # res.append([0, res[-1][REL_COST_IDX]])
    res.sort(key=lambda x: x[DAYS_IDX], reverse=True)
    return res


def calculate_cancellation_fees(row: pd.Series) -> pd.Series:
    ret_val = {"no_show_cost": 0, "cancellation_fee": 0}
    if row['cancelled'] == 0:
        return pd.Series(ret_val)

    policies = row[POLICY_COL].split("_")
    if policies[-1] == "UNKNOWN":
        return pd.Series(ret_val)
    if row['time_from_cancellation_to_checkin'] < 0:  # if no show
        no_show_policy = policies[-1]  # get the last policy
        if no_show_policy.find(
                "D") != -1:  # if there is a D in the last policy, it is a cancellation fee, so no no_show cost
            return pd.Series(ret_val)
        else:  # if there is no D in the last policy, it is a no show cost
            if no_show_policy[-1] == "P":
                percentage = int(no_show_policy[:-1]) / 100
            else:
                percentage = min(1, (int(no_show_policy[:-1]) / row['stay_duration']))
            ret_val['no_show_cost'] = percentage * row['original_selling_amount']
            return pd.Series(ret_val)
    else:  # if cancellation on time
        cancellation_policies = parse_policies(policies, row)
        idx = -1
        fee_time = row['time_from_cancellation_to_checkin']

        for i in range(len(cancellation_policies)):
            if cancellation_policies[i][DAYS_IDX] > fee_time:
                idx = i
            else:
                break
        ret_val['cancellation_fee'] = cancellation_policies[idx][REL_COST_IDX] * row['original_selling_amount']
        return pd.Series(ret_val)


def preprocess_task_2(data: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    data = data.join(data.apply(calculate_cancellation_fees, axis=1))
    # todo add default values
    return data, {}


def task_2_routine(data: pd.DataFrame):
    """
    :param data:
    :return:
    """
    model = load_model(MODEL_LOAD_PATH)
    ids = data["h_booking_id"]
    data.drop(["h_booking_id"])
    data = preprocess_task_2(data)
    # data = internal_preprocess(data)
    pred = model.predict(data)

    helper_write_csv(data["h_booking_id"], pred, "agoda_cost_of_cancellation.csv", "predicted_selling_amount")
