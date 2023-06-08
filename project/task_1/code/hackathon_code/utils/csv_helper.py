import pandas as pd


def helper_write_csv(ids: pd.Series, predicted: pd.Series, output_name: str, predict_header: str) -> None:
    data = pd.DataFrame({"ID": ids, predict_header: predicted})
    data.to_csv(output_name, index=False)


def read_csv_to_dataframe(path: str) -> pd.DataFrame:
    """
    :return: A panda's DataFrame object containing the data from the CSV file
    """
    return pd.read_csv(path)
