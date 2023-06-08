import numpy as np
from sklearn.model_selection import train_test_split

import project.task_1.code.hackathon_code.utils.csv_helper
import project.task_1.code.hackathon_code.utils.preprocess as preprocess
from project.task_1.code.hackathon_code.learners.task_1 import temp_classify_cancellation_prediction

SEED = 420420

DATA_25_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_50_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_75_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_100_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_ORIG_PATH = r'../../../instructions/agoda_cancellation_train.csv'

if __name__ == "__main__":
    np.random.seed(SEED)
    data = project.task_1.code.hackathon_code.utils.csv_helper.read_csv_to_dataframe(DATA_25_PATH)
    data, default_values = preprocess.generic_preprocess(data)
    temp_classify_cancellation_prediction(data)
    print(data.head(30))
