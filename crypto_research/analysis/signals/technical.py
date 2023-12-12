"""
Various of alphas generated on technical analysis.
"""

# Todo: split them into signals/indicators. Yield either Alpha or Indicator like Alpha(name, data, **params??),
import warnings

import pandas as pd
import pandas_ta as ta
import functools
from typing import Callable, Generator

from crypto_research.analysis.signals import Signal, Indicator

NORMALIZED_WINDOW_MULTIPLIERS = (1, 2, 4, 12)


SIGNAL_RETURN = Generator[Signal | Indicator, None, None]


def normalize_z_score(series: pd.Series, window: int) -> pd.Series:
    """Z score normalization with moving window."""
    mean_vals = series.rolling(window=window).mean()
    std_vals = series.rolling(window=window).std()
    normalized = (series - mean_vals) / std_vals
    return normalized


def signal(func: Callable) -> Callable:
    """ Decorator to process a signal function with keyword arguments. """

    @functools.wraps(func)
    def wrapper(*args, **kwargs) -> Callable:
        @functools.wraps(func)
        def inner_wrapper(data: pd.DataFrame) -> pd.Series:
            return func(data, *args, **kwargs)

        return inner_wrapper

    return wrapper

@signal
def atr(
    data: pd.DataFrame, *,
    periods: list[int],
    long_period_multipliers: tuple[int]
) -> SIGNAL_RETURN:
    """
    Generates multiple alpha signals based on Average True Range (ATR) for different periods.

    ATR Formula: ATR = (Prior ATR * (N-1) + Current TR) / N

    Alpha Signals for each period combination in `periods`:
        1. ATR Ratio: The ratio of the current ATR to a longer-term ATR average.
           It helps in identifying if the current volatility is unusual compared to historical volatility.
        2. ATR Crossover: A signal indicating whether the current ATR is above (1) or below (0) its longer-term average.
           This can suggest increasing or decreasing market volatility.

    """
    # returns = data['close'].pct_change()

    for period in periods:
        atr_value = data.ta.atr(length=period)

        for multiplier in long_period_multipliers:
            long_period = period * multiplier
            atr_long = data.ta.atr(length=long_period).rolling(window=long_period).mean()
            atr_ratio = atr_value / atr_long

            atr_crossover = atr_value > atr_long
            yield Signal("atr_ratio", atr_ratio, period=period, long_period=long_period)
            yield Signal("atr_crossover", atr_crossover, period=period, long_period=long_period)
        # vol_adj_returns = returns / atr_value
        # yield f"vol_adj_returns_{period}", vol_adj_returns
        # Todo: implement smart way to select threshold
        # atr_pct_change = atr_value.pct_change()
        # atr_breakout = atr > threshold


@signal
def sma_ratio(
    data: pd.DataFrame, *,
    periods: list[int],
    long_period_multipliers: tuple[int]
) -> SIGNAL_RETURN:
    """
    Generates multiple alpha signals based on the ratio of two Simple Moving Averages (SMA) for different period pairs.

    The Simple Moving Average (SMA) formula:
    SMA = (P1 + P2 + ... + Pn) / n
    Where:
    - P1, P2, ..., Pn are the price points.
    - n is the number of periods.

    Alpha Signals:
    1. SMA Ratio Level: Ratio of short-term SMA to long-term SMA.
    2. SMA Crossover Signal: 1 for bullish crossover, -1 for bearish crossover.
    3. SMA Ratio Rate of Change: Rate of change of the SMA ratio.
    4. SMA Ratio Z-Score: Normalized Z-score of the SMA ratio.
    5. SMA Convergence/Divergence: Difference between the short and long SMAs.
    6. SMA Ratio Slope: Slope of the SMA ratio line.
    """
    for period in periods:
        sma_value = data.ta.sma(length=period)

        for multiplier in long_period_multipliers:
            long_period = period * multiplier
            sma_long = data.ta.sma(length=long_period)
            sma_ratio_value = sma_value / sma_long
            # SMA Ratio Level
            yield Signal("sma_ratio", sma_ratio_value, period=period, long_period=long_period)

            # SMA Crossover Signal
            crossover_signal = sma_value > sma_long
            crossover_signal = crossover_signal.apply(lambda x: 1 if x else -1)
            yield Signal("sma_crossover", crossover_signal, period=period, long_period=long_period)

            # SMA Ratio Rate of Change
            roc = sma_ratio_value.pct_change()
            yield Signal("sma_ratio_roc", roc, period=period, long_period=long_period)

            # SMA Ratio Z-Score
            z_score = normalize_z_score(sma_ratio_value, window=long_period)
            yield Signal("sma_ratio_z_score", z_score, period=period, long_period=long_period)

            # SMA Convergence/Divergence
            convergence_divergence = sma_value - sma_long

            yield Indicator(
                "sma_convergence_divergence", convergence_divergence, period=period, long_period=long_period
            )

            # SMA Ratio Slope
            slope = sma_ratio_value.diff()

            yield Signal(
                "sma_ratio_slope", slope, period=period, long_period=long_period
            )


@signal
def rsi(
    data: pd.DataFrame, *,
    periods: list[int],
    overbought_threshold: int = 70,
    oversold_threshold: int = 70,
) -> SIGNAL_RETURN:
    """
    Generates multiple alpha signals based on the Relative Strength Index (RSI).

    RSI Formula:
    RSI = 100 - (100 / (1 + RS))
    Where RS = Average Gain of Up Periods / Average Loss of Down Periods over a specified period.

    Alpha Signals:
        1. RSI Level: Direct RSI values.
        2. RSI Overbought: 1 when RSI crosses above the overbought threshold, 0 otherwise.
        3. RSI Oversold: 1 when RSI crosses below the oversold threshold, 0 otherwise.
        4. RSI Mean Reversion: Indicates potential mean reversion opportunities.
           1 for potential buy (oversold), -1 for potential sell (overbought).
        5. RSI Trend: The slope of the RSI over a specified period, indicating the trend direction.
        6. RSI Divergence: Difference between the current RSI and its moving average.
    """
    for period in periods:
        rsi = data.ta.rsi(length=period)

        # RSI Level
        yield Indicator("rsi_level", rsi, period=period)
        yield Signal("rsi_level_scaled", rsi / 100, period=period)

        # RSI Overbought & Oversold Signals
        overbought = rsi > overbought_threshold
        oversold = rsi < oversold_threshold

        yield Signal("rsi_overbought", overbought.astype(int), period=period)
        yield Signal("rsi_oversold", oversold.astype(int), period=period)

        # RSI Mean Reversion
        mean_reversion = (
            overbought.apply(lambda x: -1 if x else 0) +
            oversold.apply(lambda x: 1 if x else 0)
        )
        yield Signal("rsi_mean_reversion", mean_reversion.astype(int), period=period)

        # RSI Trend
        rsi_trend = rsi.diff(periods=period)
        yield Indicator("rsi_trend", rsi_trend, period=period)

        for multiplier in NORMALIZED_WINDOW_MULTIPLIERS:
            window = period * multiplier
            rsi_normalized = normalize_z_score(rsi, window)
            yield Signal(f'rsi_normalized', rsi_normalized, period=period, window=window)


@signal
def macd(
    data: pd.DataFrame, *,
    params: list[tuple[int, int, int]]
) -> SIGNAL_RETURN:
    """
    Generates multiple alpha signals based on the Moving Average Convergence Divergence (MACD).

    MACD Formula:
    MACD Line = EMA(short_period) - EMA(long_period)
    Signal Line = EMA of the MACD Line over signal_period
    MACD Histogram = MACD Line - Signal Line

    Alpha Signals:
        1. MACD Level: The current value of the MACD line.
        2. Signal Line Level: The current value of the signal line.
        3. Histogram Level: The current value of the MACD histogram.
        4. MACD Crossover: 1 when MACD crosses above the signal line (bullish), -1 when it crosses below (bearish).
        5. MACD Divergence: The difference between the current MACD line and its moving average.
        6. MACD Trend: The slope of the MACD line over a specified period, indicating trend direction.
    """
    for short_period, long_period, signal_period in params:
        macd = data.ta.macd(fast=short_period, slow=long_period, signal=signal_period)
        macd_line, histogram, signal_line = [macd[col] for col in macd.columns]

        kwargs = dict(short_period=short_period, long_period=long_period, signal_period=signal_period)

        # MACD Level
        yield Indicator("macd_level", macd_line, **kwargs)
        # Signal Line Level
        yield Indicator("macd_signal_line_level", signal_line, **kwargs)
        # Histogram Level
        yield Indicator("macd_histogram_level", histogram, **kwargs)

        # MACD Crossover
        crossover_signal = (macd_line > signal_line).apply(lambda x: 1 if x else -1)
        yield Signal("macd_crossover", crossover_signal, **kwargs)

        # MACD Divergence

        divergence = macd_line - macd_line.rolling(window=long_period).mean()
        macd_level_normalized = normalize_z_score(macd_line, window=long_period)
        yield Indicator("macd_divergence", divergence, **kwargs, window=long_period)
        yield Signal("macd_level_normalized", macd_level_normalized, **kwargs, window=long_period)

        # MACD Trend
        trend = macd_line.diff(periods=signal_period)
        yield Indicator("macd_trend", trend, **kwargs)


@signal
def bollinger_bands(
    data: pd.DataFrame, *,
    periods: list[int],
    std: float = 2.0,
) -> SIGNAL_RETURN:
    """
    Generates multiple alpha signals based on Bollinger Bands (BBANDS).

    Bollinger Bands Formula:
    - Middle Band = SMA(close, period)
    - Upper Band = Middle Band + (nbdevup * std(close, period))
    - Lower Band = Middle Band - (nbdevdn * std(close, period))

    Alpha Signals:
        1. Bandwidth: The width of the bands relative to the middle band.
        2. %B Indicator: Position of the price in relation to the bands.
        3. Band Crossover: Signal when the price crosses above or below the bands.
        4. Band Squeeze: Identifies when the bands are closer together, indicating lower volatility.
        5. Band Expansion: Identifies when the bands are moving apart, indicating higher volatility.
    """
    for period in periods:
        bbands = data.ta.bbands(length=period, std=std)
        upper_band, middle_band, lower_band = (
            bbands[f'BBU_{period}_2.0'], bbands[f'BBM_{period}_2.0'], bbands[f'BBL_{period}_2.0']
        )

        # Bandwidth
        bandwidth = (upper_band - lower_band) / middle_band
        yield Signal("bbands_bandwidth", bandwidth, period=period)

        # %B Indicator
        percent_b = (data['close'] - lower_band) / (upper_band - lower_band)
        yield Signal("bbands_percent_b", percent_b, period=period)

        # Band Crossover
        crossover_upper = data['close'] > upper_band
        crossover_lower = data['close'] < lower_band
        crossover_signal = crossover_upper.apply(lambda x: 1 if x else 0) - crossover_lower.apply(
            lambda x: 1 if x else 0)
        yield Signal("bbands_crossover", crossover_signal, period=period)

        # Band Squeeze
        squeeze = bandwidth < bandwidth.rolling(window=period).mean()
        yield Signal("bbands_squeeze", squeeze, period=period)

        # Band Expansion
        expansion = bandwidth > bandwidth.rolling(window=period).mean()
        yield Signal("bbands_expansion", expansion, period=period)


@signal
def stochastic_oscillator(
    data: pd.DataFrame, *,
    params: list[tuple[int, int, int]],
    overbought_threshold: int = 80,
    oversold_threshold: int = 20
) -> SIGNAL_RETURN:
    """
    Generates multiple alpha signals based on the Stochastic Oscillator.

    Stochastic Oscillator Formula:
    %K = (Current Close - Lowest Low)/(Highest High - Lowest Low) * 100
    %D = Simple Moving Average of %K

    The %K line is the main line indicating momentum, while the %D line is the signal line.

    Each parameter set in `params` consists of (k_period, d_period, smooth_k):
        - k_period: The lookback period for %K.
        - d_period: The smoothing period for %D.
        - smooth_k: The smoothing period applied to %K.

    Alpha Signals:
        1. Stochastic Level: The current value of %K and %D.
        2. Stochastic Crossover: 1 when %K crosses above %D (bullish), -1 when it crosses below (bearish).
        3. Overbought/Oversold: Indicates potential reversal points. 1 for oversold, -1 for overbought.
        4. Stochastic Divergence: Difference between the current %K/%D and their moving averages.
        5. Stochastic Trend: The slope of the %K/%D lines over a specified period, indicating trend direction.
    """
    # Calculate %K and %D
    for k_period, d_period, smooth_k in params:
        low_min = data['low'].rolling(window=k_period).min()
        high_max = data['high'].rolling(window=k_period).max()
        k_line = ((data['close'] - low_min) / (high_max - low_min)) * 100
        d_line = k_line.rolling(window=d_period).mean()

        # Smooth %K if required
        if smooth_k > 1:
            k_line = k_line.rolling(window=smooth_k).mean()

        # Stochastic Level
        yield Indicator("stochastic_k", k_line, period=k_period, smooth=smooth_k)
        yield Indicator("stochastic_d", d_line, period=d_period)

        # Stochastic Crossover
        crossover_signal = (k_line > d_line).apply(lambda x: 1 if x else -1)
        yield Signal("stochastic_crossover", crossover_signal, k_period=k_period, d_period=d_period)

        # Overbought/Oversold Signals
        overbought = k_line > overbought_threshold
        oversold = k_line < oversold_threshold

        yield Signal("stochastic_overbought", overbought, period=k_period)
        yield Signal("stochastic_oversold", oversold, period=k_period)

        # Stochastic Divergence
        k_divergence = k_line - k_line.rolling(window=k_period).mean()
        d_divergence = d_line - d_line.rolling(window=d_period).mean()
        yield Indicator("stochastic_k_divergence", k_divergence, period=k_period)
        yield Indicator("stochastic_d_divergence", d_divergence, period=d_period)

        # Stochastic Trend
        k_trend = k_line.diff(periods=k_period)
        d_trend = d_line.diff(periods=d_period)
        yield Indicator("stochastic_k_trend", k_trend, period=k_period)
        yield Indicator("stochastic_d_trend", d_trend, period=d_period)


@signal
def ppo(data: pd.DataFrame, *, periods: list[tuple[int, int]]) -> SIGNAL_RETURN:
    """
    Generates multiple alpha signals based on the Percentage Price Oscillator (PPO) for different combinations of fast and slow periods.

    PPO Formula: PPO = ((Fast EMA - Slow EMA) / Slow EMA) * 100
    Where:
    - EMA is the Exponential Moving Average.

    Alpha Signals for each period combination in `periods`:
    1. PPO Value: The raw PPO value indicating momentum. Higher values suggest stronger bullish momentum, lower values suggest bearish momentum.
    3. PPO Histogram: The difference between the PPO line and its signal line, indicating acceleration or deceleration of momentum.
    """

    for fast_period, slow_period in periods:
        ppo_value = data.ta.ppo(fast=fast_period, slow=slow_period)
        ppo_signal_line = ppo_value.rolling(window=slow_period // 2).mean()
        ppo_histogram = ppo_value - ppo_signal_line

        yield Indicator("ppo_value", ppo_value, fast_period=fast_period, slow_period=slow_period)
        yield Indicator("ppo_histogram", ppo_histogram, fast_period=fast_period, slow_period=slow_period)


@signal
def ema_crossover(data: pd.DataFrame, *, periods: list[tuple[int, int]]) -> SIGNAL_RETURN:
    """
    Generates multiple alpha signals based on Exponential Moving Average (EMA) crossovers for different EMA period pairs.

    EMA Formula: EMA = Price(t) * k + EMA(y) * (1 - k)
    Where:
    - Price(t) is the price at time t.
    - k is the smoothing factor, calculated as 2 / (N + 1).
    - EMA(y) is the EMA of the previous period.

    Alpha Signals for each period pair in `ema_periods`:
    1.
        Buy Signal: Short-term EMA > Long-term EMA.
        Sell Signal: Short-term EMA < Long-term EMA.

    Each pair in `periods` should be a dict with keys 'short_period' and 'long_period'.
    """
    for short, long in periods:
        ema_short = data.ta.ema(length=short)
        ema_long = data.ta.ema(length=long)
        alpha = ema_short > ema_long
        yield Indicator("ema", ema_short, length=short)
        yield Indicator("ema", ema_long, length=long)
        yield Signal("ema_crossover", alpha, short=short, long=long)


@signal
def vwap(data: pd.DataFrame, *, periods: list[int]) -> SIGNAL_RETURN:
    """
    Generates multiple alpha signals based on Volume Weighted Average Price (VWAP) for different VWAP periods.

    VWAP Formula: VWAP = Cumulative(Typical Price * Volume) / Cumulative(Volume)
    Where:
    - Typical Price = (High + Low + Close) / 3
    - Cumulative(Typical Price * Volume) is the running total of Typical Price multiplied by Volume.
    - Cumulative(Volume) is the running total of Volume.

    Alpha Signals for each period in `vwap_periods`:
    1. VWAP Price Distance: Difference between the price and VWAP, normalized to the price.

    """

    high = data['high']
    low = data['low']
    close = data['close']
    volume = data['volume']
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        for period in periods:
            vwap_value = ta.vwap(high, low, close, volume, anchor=period)

            # Calculating VWAP Price Distance
            vwap_price_distance = (close - vwap_value) / close

            yield Indicator("vwap", vwap_value, period=period)
            yield Signal("vwap_price_distance", vwap_price_distance, period=period)


@signal
def cci(data: pd.DataFrame, *, periods: list[int]) -> SIGNAL_RETURN:
    """
    Generates multiple alpha signals based on the Commodity Channel Index (CCI) for different CCI periods.

    CCI Formula: CCI = (Typical Price - MA) / (0.015 * Mean Deviation)
    Where:
    - Typical Price = (High + Low + Close) / 3
    - MA is the moving average of the Typical Price.
    - Mean Deviation is the mean of the absolute deviations from the moving average.

    Alpha Signals for each period in `periods`:
    1. CCI Z-Score: Normalized value of CCI indicating how far the price is from its statistical mean.
    2. CCI Trend Strength: Strength of the trend based on the CCI value.
       (1 for strong positive trend, -1 for strong negative trend, 0 for weak trend)
    """

    for period in periods:
        cci_value = data.ta.cci(length=period)

        # Normalizing CCI values using Z-Score
        cci_z_score = normalize_z_score(cci_value, period)

        # Determining CCI Trend Strength
        cci_trend_strength = cci_value.apply(lambda x: 1 if x > 100 else (-1 if x < -100 else 0))

        yield Indicator("cci", cci_value, period=period)
        yield Signal("cci_z_score", cci_z_score, period=period)
        yield Signal("cci_trend_strength", cci_trend_strength, period=period)


@signal
def obv(data: pd.DataFrame, *, obv_ema_periods: list[int]) -> SIGNAL_RETURN:
    """
    Generates multiple alpha signals based on On-Balance Volume (OBV) for different OBV EMA periods.

    OBV Formula: If today's closing price is higher than yesterday's, then: OBV = yesterday's OBV + today's volume.
                 If today's closing price is lower than yesterday's, then: OBV = yesterday's OBV - today's volume.
                 Otherwise, OBV remains unchanged.

    Alpha Signals for each period in `obv_ema_periods`:
    1. OBV Rate of Change (ROC): Measures the rate of change in OBV. A high positive value indicates strong buying pressure.
    2. OBV Price Divergence: Measures the divergence between OBV and price. A divergence can signal potential trend reversals.
    """

    obv = data.ta.obv()
    close = data['close']

    for period in obv_ema_periods:
        # Calculate OBV Rate of Change
        obv_roc = ta.roc(obv, length=period)

        # Calculate OBV Price Divergence
        obv_ema = ta.ema(obv, length=period)
        price_ema = ta.ema(close, length=period)
        obv_price_divergence = obv_ema - price_ema

        yield Indicator("obv_roc", obv_roc, period=period)
        yield Indicator("obv_price_divergence", obv_price_divergence, period=period)
        for multiplier in NORMALIZED_WINDOW_MULTIPLIERS:
            window = multiplier * period
            yield Signal("obv_roc_normalized", normalize_z_score(obv_roc, window), period=period, window=window)
            yield Signal(
                "obv_price_divergence_normalized",
                normalize_z_score(obv_price_divergence, window), period=period, window=window
            )


@signal
def parabolic_sar(data: pd.DataFrame) -> SIGNAL_RETURN:
    """
    Calculates the Parabolic SAR (Stop and Reverse) signals with a continuous range between -1 and 1.
    The signal strength is based on the relative position of the price to the Parabolic SAR dots.

    Parabolic SAR Formula:
    SARn+1 = SARn + α * (EP - SARn)
    Where:
    - SARn is the current SAR value.
    - SARn+1 is the next SAR value.
    - α is the acceleration factor, typically starting at 0.02 and increasing by 0.02 each time a new EP (Extreme Point) is established, up to a maximum of 0.20.
    - EP (Extreme Point) is the highest high or lowest low recorded during the current trend.

    The closer the price is to the lower SAR (bullish), the closer the signal is to 1.
    The closer the price is to the upper SAR (bearish), the closer the signal is to -1.
    """
    psar = data.ta.psar()
    psar_long = psar['PSARl_0.02_0.2']
    psar_short = psar['PSARs_0.02_0.2']

    # Normalize the distance between price and PSAR dots to the range [-1, 1]
    distance_long = (data['close'] - psar_long) / data['close']
    distance_short = (data['close'] - psar_short) / data['close']

    # Use the distance to generate a signal value
    signal = distance_long.fillna(distance_short)

    yield Signal('parabolic_sar', signal)



SIGNALS_1H = {
    "atr": atr(periods=[6, 12, 24, 48], long_period_multipliers=(2, 4, 8, 20, 50)),
    "stochastic_oscillator": stochastic_oscillator(
        params=[
            (14, 3, 3),  # Standard settings
            (10, 5, 5),  # Shorter %K with more smoothing
            (20, 5, 5),  # Longer %K with more smoothing
            (14, 14, 3)  # Equal %K and %D periods with less smoothing
        ],
    ),
    "sma_ratio": sma_ratio(periods=[6, 12, 24, 48], long_period_multipliers=(2, 4, 8, 20, 50)),
    "bollinger_bands": bollinger_bands(periods=[6, 12, 24, 48]),

    "rsi": rsi(periods=[6, 12, 24, 48]),
    "macd": macd(params=[
        (12, 26, 9),  # Standard MACD parameters
        (5, 35, 5),  # Shorter fast period, longer slow period
        (10, 30, 8),  # Slightly different from standard
        (15, 45, 10)  # Longer periods
    ]),

    "ppo": ppo(periods=[
        (12, 26),
        (24, 48),
    ]),
    "ema_crossover": ema_crossover(periods=[
        (12, 26),
        (24, 48),
    ]),
    # "ema_crossover_12_48": ema_crossover(short_period=12, long_period=48),
    "vwap": vwap(periods=[6, 12, 24, 48]),
    "cci": cci(periods=[6, 12, 24, 48]),
    "obv": obv(obv_ema_periods=[6, 12, 24, 48]),
    "parabolic_sar": parabolic_sar(),
}

SIGNALS_1D = {
    "atr": atr(periods=[14, 28, 56], long_period_multipliers=(2, 6)),  # Increased periods for daily data
    "stochastic_oscillator": stochastic_oscillator(
        params=[
            (14, 3, 3),  # Standard settings, suitable for daily
            (21, 5, 5),  # Increased %K period with more smoothing
            (28, 5, 5),  # Further increased %K period with more smoothing
            (14, 14, 3)  # Equal %K and %D periods with less smoothing
        ],
    ),
    "sma_ratio": sma_ratio(periods=[14, 28, 56], long_period_multipliers=(2, 6)),  # Increased periods
    "bollinger_bands": bollinger_bands(periods=[14, 28, 56]),  # Increased periods

    "rsi": rsi(periods=[14, 28, 56]),  # Adjusted periods for daily data
    "macd": macd(params=[
        (12, 26, 9),  # Standard MACD parameters
        (21, 42, 9),  # Adjusted for daily data
        (28, 56, 14),  # Further adjusted for daily data
    ]),

    "ppo": ppo(periods=[
        (12, 26),  # Standard parameters, suitable for daily
        (28, 56),  # Adjusted for longer-term trends
    ]),
    "ema_crossover": ema_crossover(periods=[
        (12, 26),  # Standard EMA periods, suitable for daily
        (28, 56),  # Adjusted for longer-term trends
    ]),

    "vwap": vwap(periods=[14, 28, 56]),  # Increased periods for daily data
    "cci": cci(periods=[14, 28, 56]),  # Adjusted periods for daily data
    "obv": obv(obv_ema_periods=[14, 28, 56]),  # Adjusted OBV periods
    "parabolic_sar": parabolic_sar(),  # Suitable for daily as is
}