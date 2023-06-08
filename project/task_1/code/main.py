import numpy as np
from sklearn.model_selection import train_test_split

import project.task_1.code.hackathon_code.utils.preprocess as preprocess
from project.task_1.code.hackathon_code.learners.cancellation_prediction_classifier import \
    test_classify_cancellation_prediction

SEED = 420420

DATA_25_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_50_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_75_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_100_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_ORIG_PATH = r'../../../instructions/agoda_cancellation_train.csv'


if __name__ == "__main__":
    np.random.seed(SEED)
    data = preprocess.read_csv_to_dataframe(DATA_25_PATH)
    # data = preprocess.create_additional_cols(data)
    train, test = train_test_split(data, test_size=0.2)

    test_classify_cancellation_prediction(train, test)
    # print(data.head(30))
