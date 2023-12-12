import pandas as pd


# def simple_strategy(predicted_returns, buy_threshold, sell_threshold):
#     """
#     Generate trading signals based on predicted returns.
#
#     :param predicted_returns: DataFrame with columns as return horizons and rows as predictions.
#     :param buy_threshold: Threshold above which a buy signal is generated.
#     :param sell_threshold: Threshold below which a sell signal is generated.
#     :return: DataFrame with trading signals (1 for buy, -1 for sell, 0 for hold).
#     """
#     signals = pd.DataFrame(index=predicted_returns.index, columns=predicted_returns.columns)
#
#     signals[predicted_returns > buy_threshold] = 1   # Buy Signal
#     signals[predicted_returns < sell_threshold] = -1 # Sell Signal
#     signals[(predicted_returns >= sell_threshold) & (predicted_returns <= buy_threshold)] = 0 # Hold
#
#     return signals