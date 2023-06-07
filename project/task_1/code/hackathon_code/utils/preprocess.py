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

    # Shuffle the rows if randomize is True
    if randomize:
        random.shuffle(rows)

    # Determine the number of rows per division
    rows_per_division = len(rows) // divisions

    # Create divisions directory if it doesn't exist
    divisions_dir = os.path.join(directory, 'divisions')
    if not os.path.exists(divisions_dir):
        os.makedirs(divisions_dir)

    # Write rows to separate division files
    for i in range(divisions):
        division_path = os.path.join(divisions_dir, f'agoda_train_{i}_from_{divisions}.csv')
        with open(division_path, 'w', newline='') as file:
            writer = csv.writer(file)
            start_index = i * rows_per_division
            end_index = (i + 1) * rows_per_division
            writer.writerows(rows[start_index:end_index])
