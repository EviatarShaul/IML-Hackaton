import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import project.task_1.code.hackathon_code.utils.csv_helper
import project.task_1.code.hackathon_code.utils.preprocess as preprocess
from project.task_1.code.hackathon_code.learners.task_1 import \
    temp_classify_cancellation_prediction
from project.task_1.code.hackathon_code.learners.task_3 import \
    churn_prediction_model
from project.task_1.code.hackathon_code.learners.task_1 import \
    temp_classify_cancellation_prediction

SEED = 420420

DATA_25_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_50_PATH = r'../../../instructions/divisions/agoda_train_1_from_4.csv'
DATA_75_PATH = r'../../../instructions/divisions/agoda_train_2_from_4.csv'
DATA_100_PATH = r'../../../instructions/divisions/agoda_train_3_from_4.csv'
DATA_ORIG_PATH = r'../../../instructions/agoda_cancellation_train.csv'


def explore():
    np.random.seed(SEED)
    data1 = project.task_1.code.hackathon_code.utils.csv_helper.read_csv_to_dataframe(
        DATA_25_PATH)
    data2 = project.task_1.code.hackathon_code.utils.csv_helper.read_csv_to_dataframe(
        DATA_50_PATH)
    data1, default_values = preprocess.generic_preprocess(data1)
    data2, default_values = preprocess.generic_preprocess(data2)
    temp, test = train_test_split(data2, test_size=0.3)
    data2, validate = train_test_split(temp, test_size=0.5)
    data = pd.concat([data1, data2])

    # Q1:
    temp_classify_cancellation_prediction(data, validate, test)
    # classify_cancellation_prediction(data)

    # Q2:

    # Q3:
    # churn_prediction_model(data)

    # Q4:


if __name__ == "__main__":
    explore()
