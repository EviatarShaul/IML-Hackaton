import csv
import random
import os
import pandas as pd

import csv
import random
import os


def divide_csv_file(path, divisions, randomize):
    # Get the directory and filename from the input path
    directory = os.path.dirname(path)
    filename = os.path.basename(path)

    # Read the CSV file
    with open(path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # Separate the header row from the data rows
    header = rows[0]
    data_rows = rows[1:]

    # Shuffle the data rows if randomize is True
    if randomize:
        random.shuffle(data_rows)

    # Determine the number of data rows per division
    rows_per_division = len(data_rows) // divisions

    # Create divisions directory if it doesn't exist
    divisions_dir = os.path.join(directory, 'divisions')
    if not os.path.exists(divisions_dir):
        os.makedirs(divisions_dir)

    # Write rows to separate division files
    for i in range(divisions):
        division_path = os.path.join(divisions_dir, f'agoda_train_{i}_from_{divisions}.csv')
        with open(division_path, 'w', newline='') as file:
            writer = csv.writer(file)

            # Write the header row to each division file
            writer.writerow(header)

            start_index = i * rows_per_division
            end_index = (i + 1) * rows_per_division

            # Write the corresponding data rows to each division file
            writer.writerows(data_rows[start_index:end_index])


def read_csv_to_dataframe(path):
    return pd.read_csv(path)
