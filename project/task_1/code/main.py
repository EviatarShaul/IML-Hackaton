import sys

import numpy as np

import os
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser
import pandas as pd
from sklearn.model_selection import train_test_split

from project.task_1.code.hackathon_code.learners import task_2
from project.task_1.code.hackathon_code.utils import csv_helper

import project.task_1.code.hackathon_code.utils.csv_helper
import project.task_1.code.hackathon_code.utils.preprocess as preprocess
from project.task_1.code.hackathon_code.learners.task_1 import temp_classify_cancellation_prediction
from project.task_1.code.hackathon_code.learners.task_3 import churn_prediction_model
from project.task_1.code.hackathon_code.learners.task_1 import temp_classify_cancellation_prediction
from task_1.code.hackathon_code.learners.task_2 import \
    explore_predict_selling_amount

SEED = 420420

DATA_25_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_50_PATH = r'../../../instructions/divisions/agoda_train_1_from_4.csv'
DATA_75_PATH = r'../../../instructions/divisions/agoda_train_2_from_4.csv'
DATA_100_PATH = r'../../../instructions/divisions/agoda_train_3_from_4.csv'
DATA_ORIG_PATH = r'../../../instructions/agoda_cancellation_train.csv'


def explore():
    data1 = csv_helper.read_csv_to_dataframe(DATA_25_PATH)
    data2 = csv_helper.read_csv_to_dataframe(DATA_50_PATH)
    # data3 = csv_helper.read_csv_to_dataframe(DATA_75_PATH)
    data = pd.concat([data1, data2])
    data, default_values = preprocess.generic_preprocess(data)

    # split data into 2 parts by row
    data1, data2 = data.iloc[:int(len(data) / 2)], data.iloc[int(len(data) / 2):]
    temp, test = train_test_split(data2, test_size=0.3)
    data2, validate = train_test_split(temp, test_size=0.5)
    data = pd.concat([data1, data2])

    # Q1:
    # temp_classify_cancellation_prediction(data, validate, test)

    # Q2:
    explore_predict_selling_amount(data, validate, test)

    # Q3:
    # churn_prediction_model(data)

    # Q4:


def main(input_file):
    data = project.task_1.code.hackathon_code.utils.csv_helper.read_csv_to_dataframe(input_file)
    data, default_values = preprocess.generic_preprocess(data)
    temp_classify_cancellation_prediction(data)
    # todo uncomment before submit

    # # Task 1
    # try:
    #     task_1.code.hackathon_code.learners.task_1.task_1_routine(data)
    # except Exception as e:
    #     pass
    # Task 2
    try:
        task_2.task_2_routine(data)
    except Exception as e:
        pass
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


if __name__ == "__main__":
    np.random.seed(SEED)
    # if len(sys.argv) != 2:
    #     exit("No input file!")
    # input_file = sys.argv[1]
    
    explore()
    exit(0)   
    main(input_file)

