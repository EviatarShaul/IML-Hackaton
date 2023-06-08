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
    # # Task 1
    # try:
    #     task_1.code.hackathon_code.learners.task_1.task_1_routine(data)
    # except Exception as e:
    #     pass
    # # Task 2
    # try:
    #     task_1.code.hackathon_code.learners.task_2.task_2_routine(data)
    # except Exception as e:
    #     pass
    # # Task 3
    # try:
    #     task_1.code.hackathon_code.learners.task_3.task_3_routine(data)
    # except Exception as e:
    #     pass
    # # Task 4
    # try:
    #     task_1.code.hackathon_code.learners.task_4.task_4_routine(data)
    # except Exception as e:
    #     pass
