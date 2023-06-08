import sys

import numpy as np
import os
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

import project.task_1.code.hackathon_code.utils.csv_helper
import project.task_1.code.hackathon_code.utils.preprocess as preprocess
from project.task_1.code.hackathon_code.learners.task_1 import temp_classify_cancellation_prediction

SEED = 420420

DATA_25_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_50_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_75_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_100_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_ORIG_PATH = r'../../../instructions/agoda_cancellation_train.csv'


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


if __name__ == "__main__":
    np.random.seed(SEED)
    # if len(sys.argv) != 2:
    #     exit("No input file!")
    # input_file = sys.argv[1]

    input_file = DATA_25_PATH  # todo delete
    main(input_file)
