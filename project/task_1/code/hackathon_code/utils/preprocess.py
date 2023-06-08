import numpy as np
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


def read_csv_to_dataframe(path: str) -> pd.DataFrame:
    """
    :return: A pandas DataFrame object containing the data from the CSV file
    """
    return pd.read_csv(path)


def create_additional_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    create additional columns for the dataframe
    :param df: dataframe to be processed
    :return: the dataframe with additional columns
    """
    # Convert the columns to datetime objects
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])
    df['checkin_date'] = pd.to_datetime(df['checkin_date'])
    df['checkout_date'] = pd.to_datetime(df['checkout_date'])

    # Calculate the time difference and create a new column, in hours
    df['time_from_booking_to_checkin'] = (df['checkin_date'] - df['booking_datetime']) / pd.Timedelta(hours=1)
    df['stay_duration'] = (df['checkout_date'] - df['checkin_date']) / pd.Timedelta(hours=1)

    # Create a new column for whether the booking is on a weekday
    df['is_weekday'] = df['checkin_date'].dt.dayofweek.isin([0, 1, 2, 3, 4]).astype(int)

    # Create a new column for the month of the checkin date
    df['checkin_month'] = df['checkin_date'].dt.month

    # Make the month cyclic with sin and cos
    df['checkin_month_sin'] = np.sin((df['checkin_month'] - 1) * (2. * np.pi / 12))
    df['checkin_month_cos'] = np.cos((df['checkin_month'] - 1) * (2. * np.pi / 12))
    df = df.drop(columns=['checkin_month'])

    # Create a new column for the age of the hotel at the time of booking in years
    df['hotel_age'] = (pd.to_datetime(df['booking_datetime']) - pd.to_datetime(
        df['hotel_live_date'])) / pd.Timedelta(days=365)

    # Create a new column for the sum of special requests
    special_requests = ['request_nonesmoke', 'request_latecheckin', 'request_highfloor', 'request_largebed',
                        'request_twinbeds', 'request_airport']
    df['special_requests'] = 0
    for request in special_requests:
        df['special_requests'] += df[request]
    return df
