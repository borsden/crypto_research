import pandas as pd

IS_RATIO = 0.8


def get_IS_OS(data: pd.DataFrame, ratio: float = IS_RATIO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into in-sample and out-of-sample datasets based on a specified ratio.
    """
    split_index = int(len(data) * ratio)
    in_sample_data = data.iloc[:split_index]
    out_of_sample_data = data.iloc[split_index:]
    return in_sample_data, out_of_sample_data
