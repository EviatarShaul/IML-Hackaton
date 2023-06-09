import sys

import numpy as np

import os
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import pandas as pd
from sklearn.model_selection import train_test_split

from project.task_1.code.hackathon_code.learners import task_2
from project.task_1.code.hackathon_code.utils import csv_helper

from project.task_1.code.hackathon_code.utils.csv_helper import *
import project.task_1.code.hackathon_code.utils.preprocess as preprocess
from project.task_1.code.hackathon_code.learners.task_1 import temp_classify_cancellation_prediction, task_1_routine
from project.task_1.code.hackathon_code.learners.task_3 import churn_prediction_model
from project.task_1.code.hackathon_code.learners.task_1 import temp_classify_cancellation_prediction
from task_1.code.hackathon_code.learners.task_2 import explore_predict_selling_amount



# def explore():
#     data1 = csv_helper.read_csv_to_dataframe(DATA_ORIG_PATH)
#     data2 = csv_helper.read_csv_to_dataframe(DATA_50_PATH)
#     # data3 = csv_helper.read_csv_to_dataframe(DATA_75_PATH)
#     data = pd.concat([data1, data2])
#     data, default_values = preprocess.generic_preprocess(data)
#
#     # split data into 2 parts by row
#     data1, data2 = data.iloc[:int(len(data) / 2)], data.iloc[int(len(data) / 2):]
#     temp, test = train_test_split(data2, test_size=0.3)
#     data2, validate = train_test_split(temp, test_size=0.5)
#     data = pd.concat([data1, data2])
#
#     # Q1:
#     temp_classify_cancellation_prediction(data, validate, test)
#
#     # Q2:
#     explore_predict_selling_amount(data, validate, test)
#
#     # Q3:
#     churn_prediction_model(data)
#
#     # Q4:


if __name__ == "__main__":
    SEED = 420420
    np.random.seed(SEED)
    if len(sys.argv) <= 3:
        exit("No input file!")
    input_file_1 = read_csv_to_dataframe(sys.argv[1])
    input_file_2 = read_csv_to_dataframe(sys.argv[2])
    input_file_3 = read_csv_to_dataframe(sys.argv[3])


    # exploration part
    # data = project.task_1.code.hackathon_code.utils.csv_helper.read_csv_to_dataframe(input_file)
    # temp_classify_cancellation_prediction(data)
    input_file_1 = preprocess.generic_preprocess(input_file_1)[0]
    input_file_2 = preprocess.generic_preprocess(input_file_2)[0]
    input_file_3 = preprocess.generic_preprocess(input_file_3)[0]

    # Task 1
    try:
        task_1_routine(input_file_1)
    except Exception as e:
        print(e)
    # Task 2
    task_2.task_2_routine(input_file_2)
    try:
        pass
    except Exception as e:
        pass
    # Task 3
    try:
        churn_prediction_model(input_file_3)
    except Exception as e:
        pass
    # Task 4
    try:
        # in future
        pass
    except Exception as e:
        pass
