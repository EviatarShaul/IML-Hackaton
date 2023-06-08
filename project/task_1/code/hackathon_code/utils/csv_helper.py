import pandas as pd


def helper_write_csv(ids: pd.Series, predicted: pd.Series, output_name: str, predict_header: str) -> None:
    data = pd.DataFrame({"ids": ids, predict_header: predicted})
    data.to_csv(output_name, index=False)
