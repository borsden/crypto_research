import pandas as pd


def resample(data: pd.DataFrame, window):
    """
    Resample pandas dataframe with 'time' and 'pair' multiindex.

    Parameters:
    data (pd.DataFrame): DataFrame to resample.
    window (str): Resampling frequency.

    Returns:
    pd.DataFrame: Resampled DataFrame.
    """
    rules = {
        'open': 'first',
        'close': 'last',
        'high': 'max',
        'low': 'min',
        'number_of_trades': 'sum',
        'quote_asset_volume': 'sum',
        'taker_buy_base_asset_volume': 'sum',
        'taker_buy_quote_asset_volume': 'sum',
        'volume': 'sum'
    }

    data = data.reset_index()
    data = data.set_index('time')
    # # Group by 'pair', resample and aggregate according to rules
    resampled_data = data.groupby('pair').apply(
        lambda x: x.resample(window).agg(rules)
    )
    return resampled_data
