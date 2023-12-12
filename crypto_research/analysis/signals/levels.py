# import pandas as pd
# import numpy as np
# from typing import Generator
# from crypto_research.analysis.signals import Indicator
#
# def technical_analysis_indicators(data: pd.DataFrame) -> Generator[Indicator, None, None]:
#     """
#     Generates various technical analysis indicators including Trend Lines, Fibonacci Retracement, and more.
#
#     :param data: A pandas DataFrame with columns ['open', 'high', 'low', 'close', 'volume'].
#     :return: A generator yielding Indicator objects.
#     """
#
#     def rolling_max_min(data: pd.DataFrame, window: int) -> tuple[pd.Series, pd.Series]:
#         """
#         Calculate rolling maximum and minimum values to approximate resistance and support trend lines.
#
#         :param data: DataFrame with 'high' and 'low' columns.
#         :param window: Window size for rolling calculation.
#         :return: Tuple of rolling maximum and minimum series.
#         """
#         rolling_max = data['high'].rolling(window=window, min_periods=1).max()
#         rolling_min = data['low'].rolling(window=window, min_periods=1).min()
#         return rolling_max, rolling_min
#
#     def fibonacci_retracement(data: pd.DataFrame, window: int) -> pd.DataFrame:
#         """
#         Calculate Fibonacci Retracement levels based on rolling maximum and minimum.
#
#         :param data: DataFrame with 'high' and 'low' columns.
#         :param window: Window size for rolling calculation.
#         :return: DataFrame with Fibonacci Retracement levels.
#         """
#         rolling_high, rolling_low = rolling_max_min(data, window=window)
#         diff = rolling_high - rolling_low
#         levels = {
#             '0%': rolling_low,
#             '23.6%': rolling_low + diff * 0.236,
#             '38.2%': rolling_low + diff * 0.382,
#             '50%': rolling_low + diff * 0.5,
#             '61.8%': rolling_low + diff * 0.618,
#             '100%': rolling_high
#         }
#         return pd.DataFrame(levels)
#
#     # Window size for calculations
#     window_size = 30  # Can be adjusted as needed
#
#     # Generate Trend Lines
#     rolling_resistance, rolling_support = rolling_max_min(data, window=window_size)
#     yield Indicator("Rolling Resistance", rolling_resistance)
#     yield Indicator("Rolling Support", rolling_support)
#
#     # Generate Fibonacci Retracement levels
#     fib_levels = fibonacci_retracement(data, window=window_size)
#     for level_name, level_values in fib_levels.items():
#         yield Indicator(f"Fibonacci Retracement - {level_name}", level_values)

# """
# data1 = data0.copy()
#
# while len(data1)>3:
#
#     reg = linregress(
#                     x=data1['date_id'],
#                     y=data1['Adj. High'],
#                     )
#     data1 = data1.loc[data1['Adj. High'] > reg[0] * data1['date_id'] + reg[1]]
#
# reg = linregress(
#                     x=data1['date_id'],
#                     y=data1['Adj. High'],
#                     )
#
# data0['high_trend'] = reg[0] * data0['date_id'] + reg[1]
#
# # low trend line
#
# data1 = data0.copy()
#
# while len(data1)>3:
#
#     reg = linregress(
#                     x=data1['date_id'],
#                     y=data1['Adj. Low'],
#                     )
#     data1 = data1.loc[data1['Adj. Low'] < reg[0] * data1['date_id'] + reg[1]]
#
# reg = linregress(
#                     x=data1['date_id'],
#                     y=data1['Adj. Low'],
#                     )
#
# """