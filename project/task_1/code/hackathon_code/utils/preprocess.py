import datetime
import numpy as np
import pandas as pd
import csv
import random
import os
from typing import List, Optional, Tuple, Dict, Union, Any
from currency_converter import CurrencyConverter
from pandas import DataFrame

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


def create_additional_cols(df: pd.DataFrame) -> tuple[DataFrame, dict[str, Any]]:
    """
    create additional columns for the dataframe
    :param df: dataframe to be processed
    :return: the dataframe with additional columns
    """
    default_values = {}
    # Convert the columns to datetime objects
    df['booking_datetime'] = pd.to_datetime(df['booking_datetime'])
    df['checkin_date'] = pd.to_datetime(df['checkin_date'])
    df['checkout_date'] = pd.to_datetime(df['checkout_date'])

    # Calculate the time difference and create a new column, in hours
    df['time_from_booking_to_checkin'] = (df['checkin_date'] - df['booking_datetime']) / pd.Timedelta(hours=1)
    default_values['time_from_booking_to_checkin'] = df['time_from_booking_to_checkin'].mean()

    df['stay_duration'] = (df['checkout_date'] - df['checkin_date']) / pd.Timedelta(hours=1)
    default_values['stay_duration'] = df['stay_duration'].mean()

    # Create a new column for whether the booking is on a weekday
    df['is_weekday'] = df['checkin_date'].dt.dayofweek.isin([0, 1, 2, 3, 4]).astype(int)
    default_values['is_weekday'] = df['is_weekday'].mean()

    # Create a new column for the month of the checkin date
    df['checkin_month'] = df['checkin_date'].dt.month

    # Make the month cyclic with sin and cos
    df['checkin_month_sin'] = np.sin((df['checkin_month'] - 1) * (2. * np.pi / 12))
    df['checkin_month_cos'] = np.cos((df['checkin_month'] - 1) * (2. * np.pi / 12))
    df = df.drop(columns=['checkin_month'])

    mean_month = df['checkin_month_sin'].mean()
    default_values['checkin_month_sin'] = np.sin((mean_month - 1) * (2. * np.pi / 12))
    default_values['checkin_month_cos'] = np.cos((mean_month - 1) * (2. * np.pi / 12))

    # Create a new column for the age of the hotel at the time of booking in years
    df['hotel_age'] = (pd.to_datetime(df['booking_datetime']) - pd.to_datetime(
        df['hotel_live_date'])) / pd.Timedelta(days=365)

    default_values['hotel_age'] = df['hotel_age'].mean()

    # Create a new column for the sum of special requests
    special_requests = ['request_nonesmoke', 'request_latecheckin', 'request_highfloor', 'request_largebed',
                        'request_twinbeds', 'request_airport', 'request_earlycheckin']
    df['special_requests'] = 0
    for request in special_requests:
        df[request] = df[request].fillna(df[request].mean())
        df['special_requests'] += df[request]
    default_values['special_requests'] = df['special_requests'].mean()

    return df, default_values


DUMMY_COLS = ['accommadation_type_name', 'charge_option', 'original_payment_type']


def get_dummies_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    runs the get dummies preprocess on the data
    :param df: the dataframe to be processed
    :return: a tuple of the processed dataframe and the dictionary of the columns and their default values
    """
    defualt_values = {}
    for col in DUMMY_COLS:
        dist = df[col].value_counts(normalize=True)
        defualt_values[col] = dist
        df = pd.concat([df, pd.get_dummies(df[col], prefix=col, dtype=float)], axis=1)
        if df[col].isnull().values.any():  # if there are null values in the column, fill dummies with approximate values
            df.loc[df[col].isnull(), df.columns.str.startswith(col)] = np.nan
            for val in dist.index:
                df[col + '_' + str(val)].fillna(dist[val], inplace=True)
        df = df.drop(columns=[col])

    return df, defualt_values


# todo - check of original_payment_type or original_payment_method
# todo - check langauge vs guest_nationality_country_name vs customer_nationality
# todo - check hotel_country_code vs (hotel_area_code and hotel_city_code)


cols_to_drop = ['booking_datetime', 'checkin_date', 'checkout_date', 'hotel_live_date', 'original_payment_currency',
                'original_payment_method','h_customer_id']


def generic_preprocess(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
    """
    runs the generic preprocess on the data
    :param df: the dataframe to be processed
    :return: a tuple of the processed dataframe and the dictionary of the columns and their default values
    """
    df, default_values = create_additional_cols(df)
    default_values['original_selling_amount'] = df['original_selling_amount'].mean()

    df, dummies_default_values = get_dummies_preprocess(df)
    default_values.update(dummies_default_values)
    df = df.drop(columns=cols_to_drop)

    return df, default_values
