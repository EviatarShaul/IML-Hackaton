import numpy as np
import project.task_1.code.hackathon_code.utils.preprocess as preprocess

SEED = 420420

DATA_25_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_50_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_75_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_100_PATH = r'../../../instructions/divisions/agoda_train_0_from_4.csv'
DATA_ORIG_PATH = r'../../../instructions/agoda_cancellation_train.csv'


if __name__ == "__main__":
    np.random.seed(SEED)
    data = preprocess.read_csv_to_dataframe(DATA_25_PATH)
    data = preprocess.create_additional_cols(data)
    print(data.head(30))
