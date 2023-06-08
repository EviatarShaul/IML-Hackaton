# Task 1.2.2
from typing import NoReturn

import sklearn.linear_model
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

import project.task_1.code.hackathon_code.utils.model_helper
from project.task_1.code.hackathon_code.utils.csv_helper import *
from project.task_1.code.hackathon_code.utils.preprocess import *
import joblib

MODEL_LOAD_PATH = ""  # todo add model path
MODEL_SAVE_PATH = ""  # todo add model path

LABEL_NAME = "predicted_selling_amount"


def task_2_routine(data: pd.DataFrame):
    """
    :param data:
    :return:
    """
    model = task_1.code.hackathon_code.utils.model_helper.load_model(MODEL_LOAD_PATH)
    ids = data["h_booking_id"]
    data.drop(["h_booking_id"])
    # Todo: add internal preprocess
    # data = internal_preprocess(data)
    pred = model.predict(data)

    helper_write_csv(data["h_booking_id"], pred, "agoda_cost_of_cancellation.csv", "predicted_selling_amount")
