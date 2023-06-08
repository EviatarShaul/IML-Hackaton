import numpy as np
from sklearn.model_selection import train_test_split

import project.task_1.code.hackathon_code.utils.preprocess as preprocess
from project.task_1.code.hackathon_code.learners.task_1 import \
    temp_classify_cancellation_prediction
from project.task_1.code.hackathon_code.learners.task_3 import \
    churn_prediction_model

SEED = 420420

DATA_25_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_50_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_75_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_100_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_ORIG_PATH = r'../../../instructions/agoda_cancellation_train.csv'


if __name__ == "__main__":
    np.random.seed(SEED)
    raw_data = preprocess.read_csv_to_dataframe(DATA_25_PATH)

    # Q1:
    # temp_classify_cancellation_prediction(raw_data)
    # classify_cancellation_prediction(raw_data)

    # Q2:

    # Q3:
    churn_prediction_model(raw_data)

    # Q4:
