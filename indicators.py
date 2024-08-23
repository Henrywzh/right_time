import pandas as pd
import numpy as np


def calculate_rsi(data, window=14):  # 70 overbought, 30 oversold
    """
    Calculate the Relative Strength Index (RSI) for a given pandas DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame containing the price data with a 'Close' column.
    window (int): The window size for calculating RSI. Default is 14.

    Returns:
    pd.Series: A pandas Series containing the RSI values.
    or
    pd.DataFrame
    """
    # Calculate the price differences
    delta = data['Close'].diff()

    # Separate positive and negative gains
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the average gain and loss
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    # Calculate the Relative Strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_cci(data, window=20):  # 100 overbought, -100 oversold
    """
    Calculate the Commodity Channel Index (CCI) for a given pandas DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame containing the price data with 'High', 'Low', and 'Close' columns.
    window (int): The window size for calculating CCI. Default is 20.

    Returns:
    pd.Series: A pandas Series containing the CCI values.
    or
    pd.DataFrame
    """
    # Calculate the typical price
    tp = (data['High'] + data['Low'] + data['Close']) / 3

    # Calculate the moving average of the typical price
    tp_sma = tp.rolling(window=window, min_periods=1).mean()

    # Calculate the mean deviation
    deviation = (tp - tp_sma).abs()
    mean_deviation = deviation.rolling(window=window, min_periods=1).mean()

    # Calculate the CCI
    cci = (tp - tp_sma) / (0.015 * mean_deviation)

    return cci


def calculate_sma(data, window=20):
    """
    Calculate the Simple Moving Average (SMA) for a given pandas DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame containing the price data with a 'Close' column.
    window (int): The window size for calculating SMA. Default is 20.

    Returns:
    pd.Series: A pandas Series containing the SMA values.
    or
    pd.DataFrame
    """
    sma = data['Close'].rolling(window=window, min_periods=1).mean()
    return sma


def calculate_roc(data, window=12):
    """
    Calculate the Rate of Change (ROC) for a given pandas DataFrame.

    Parameters:
    data (pd.DataFrame): DataFrame containing the price data with a 'Close' column.
    window (int): The window size for calculating ROC. Default is 12.

    Returns:
    pd.Series: A pandas Series containing the ROC values.
    or
    pd.DataFrame
    """
    roc = data['Close'].diff(window) / data['Close'].shift(window) * 100
    return roc
