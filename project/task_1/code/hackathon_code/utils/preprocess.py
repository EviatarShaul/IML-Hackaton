import datetime
import pandas as pd
import csv
import random
import os
from currency_converter import CurrencyConverter

# Initializing a Currency Converter
currency_converter = CurrencyConverter()


def convert_currency_to_usd(amount: float, curr: str, date: datetime.date) -> float:
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

    return currency_converter.convert(amount=amount, currency=curr, new_currency='USD', date=date)


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


# todo read this!
#  Data inspection and description - https://docs.google.com/spreadsheets/d/1tw4stK7GWiv9wh7VQ1cY5yMLxOtw4CbX7wdzLQl3JFQ/edit?usp=sharing

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
    # todo - this is where is stopped!
    df['time_from_booking_to_checkin'] = df['checkin_date'] - df['booking_datetime']
    df['stay_duration'] = df['checkout_date'] - df['checkin_date']

    # Create a new column for number of people in the booking
    df['num_of_people'] = df['num_of_adults'] + df['num_of_children']

    # Create a new column for the day of the week of the checkin date
    df['checkin_day_of_week'] = df['checkin_datetime'].dt.dayofweek

    # Create a new column for the month of the checkin date
    df['checkin_month'] = df['checkin_datetime'].dt.month

    # Create a new column for the age of the hotel
    df['hotel_age'] = df['checkin_datetime'].dt.year - df['hotel_live_datetime'].dt.year

    # Create a new column for the sum of special requests
    special_requests = ['request_nonesmoke', 'request_latecheckin', 'request_highfloor', 'request_largebed',
                        'request_twinbeds', 'request_airport']
    for request in special_requests:
        df['special_requests'] += df[request]

    return df
