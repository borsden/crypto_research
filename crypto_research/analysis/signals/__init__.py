import pandas as pd
from loguru import logger


class Base:
    def __init__(self, name: str, data: pd.Series, **kwargs):
        self.name = name
        self.data = data
        self.kwargs = kwargs

    def __str__(self):
        """Return name with values from kwargs"""
        kwargs_str = ""
        if self.kwargs:
            kwargs_values = "_".join([str(val) for val in self.kwargs.values()])
            kwargs_str = f"_{kwargs_values}"

        return f"{self.name}{kwargs_str}"

    def __repr__(self):
        """Class representation."""
        kwargs_str = ""
        if self.kwargs:
            kwargs_values = ", ".join([f"{key}={value}" for key, value in self.kwargs.items()])
            kwargs_str = f", {kwargs_values} "

        return f"{self.__class__.__name__}({self.name}{kwargs_str})"


class Signal(Base):
    """Class to use for signals (alphas)."""


class Indicator(Base):
    """Class to use for indicators."""


def combine_signals(data: pd.DataFrame, signals_dict: dict[str, callable]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Combines multiple trading signals into a single DataFrame.
    """

    def process_data(name: str, data: pd.Series | pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Process the signal data by converting it to a DataFrame and assigning appropriate column names.

        :param name: The name of the signal.
        :param data: The data associated with the signal, can be a Series or DataFrame.
        :return: A DataFrame with processed data.
        """
        # If the data is a Series, rename it and convert to DataFrame
        if isinstance(data, pd.Series):
            data = data.to_frame(name=name)

        # If the data is already a DataFrame, rename its columns
        elif isinstance(data, pd.DataFrame):
            data.columns = [
                (f"{name}_{col}" if col != data.index.name else col)
                for col in data.columns
            ]

        return data

    # Initialize an empty DataFrame
    indicators = []
    signals = []
    for signal_common_name, signal_func in signals_dict.items():
        try:
            for result in signal_func(data):
                if isinstance(result, Signal):
                    signals.append(process_data(str(result), result.data))
                elif isinstance(result, Indicator):
                    indicators.append(process_data(str(result), result.data))
                else:
                    raise ValueError(f'Unrecognized signal: {type(result)}')
        except Exception as e:
            logger.error(f"{signal_common_name} raises the following: {e}")
    indicators = pd.concat(indicators, axis=1)
    signals = pd.concat(signals, axis=1)

    indicators = indicators.loc[:,~indicators.columns.duplicated()]
    signals = signals.loc[:,~signals.columns.duplicated()]

    return indicators, signals
