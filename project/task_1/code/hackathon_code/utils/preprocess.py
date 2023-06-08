import datetime
import numpy as np
import pandas as pd
import csv
import random
import os
from typing import List, Optional, Tuple, Dict
from currency_converter import CurrencyConverter

# Initializing a Currency Converter
currency_converter = CurrencyConverter(fallback_on_wrong_date=True, fallback_on_missing_rate=True)


def create_x_y_df(df: pd.DataFrame, x_columns: List[str], label_column: str = None) -> Tuple[
    pd.DataFrame, Optional[pd.Series]]:
    """
    :param df: Raw data split to
    :param label_column: the y feature
    :param x_columns: select row to design the matrix with
    :return:
            X: DataFrame of shape (n_samples, n_features)
                Data frame of samples and feature values.
            
            y: Series of shape (n_samples, )
            Responses corresponding samples in data frame.
    """
    return df[x_columns], df[label_column] if label_column is not None else df[x_columns]


def convert_currency_to_usd(amount: float, curr: str, date: datetime.date):
    """
    Converts a given currency to USD based on its value at a given date.

    Parameters:
    - amount (float): The amount of currency to be converted.
    - curr (str): The currency code of the currency to be converted.
    - date (datetime.date): The trade date for which the currency conversion should be performed.

    Returns:
    - float: The converted amount in USD.

    Note:
    - This function relies on an external currency converter library named `currency_converter`.
    - The `currency_converter` library should be installed and imported before using this function as requested
        the requirements file.
    """
    if curr in currency_converter.currencies:
        return currency_converter.convert(amount=amount, currency=curr, new_currency='USD', date=date)
    else:
        return np.mean([currency_converter.convert(amount=amount, currency=c, new_currency='USD', date=date) for c in
                        currency_converter.currencies], axis=0)


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


def create_dummies(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    :param df: Raw data split to
    :param columns: List of columns to create dummies for
    :return: A pandas DataFrame object containing the data from the CSV file
    """
    return pd.get_dummies(df, columns=columns)


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

    # todo - move function?
    # Create a new column for the sum of special requests
    special_requests = ['request_nonesmoke', 'request_latecheckin', 'request_highfloor', 'request_largebed',
                        'request_twinbeds', 'request_airport']
    df['special_requests'] = 0
    for request in special_requests:
        df['special_requests'] += df[request]
    return df


def generic_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    runs the generic preprocess on the data
    :param df: the dataframe to be processed
    :return: a tuple of the processed dataframe and the dictionary of the columns and their default values
    """
    df = create_additional_cols(df)
    df["cost"] = df.apply(lambda row: convert_currency_to_usd(row["original_selling_amount"],
                                                              row["original_payment_currency"],
                                                              pd.to_datetime(row["booking_datetime"])), axis=1)
    df = df.drop(columns=['original_selling_amount', 'original_payment_currency'])

    return df, {}
