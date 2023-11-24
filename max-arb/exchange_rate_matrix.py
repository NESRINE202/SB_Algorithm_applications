from binance.client import Client
import pandas as pd
import numpy as np


# This file is simply used to quickly build an exchange rate matrix pulled from Binance


def exchange_rate_matrix():
    #
    # This function will return the current exchange rate matrix as well as the list of coins, in the righ order, so we can match each pair with the corresponding coins
    #

    # Replace 'YOUR_API_KEY' and 'YOUR_API_SECRET' with your Binance API credentials
    api_key = 'lyN8BYr6qep4Si75fd1rKSv6DRjMuTajuKyuNWx99bZxUJlKooAiStiWSkzL6qyr'
    api_secret = 'RHidZheo2U8el24kRNizPnsjnK4d2iXx7AWzq5IVG73MYNAkSvQeyVE412r4yalS'

    # Initialize the Binance client
    client = Client(api_key, api_secret)

    # # Fetch all trading pairs
    exchange_info = client.get_exchange_info()
    symbols = exchange_info['symbols']

    # Fetch the current exchange rates for all trading pairs in a single request
    all_ticker_prices = client.get_symbol_ticker()

    # Create a dictionary mapping trading pairs to their current exchange rates
    exchange_rates = {}
    for ticker in all_ticker_prices:
        exchange_rates[ticker['symbol']] = float(ticker['price'])

    # Identify the unique coins
    coins = set()
    for symbol_info in symbols:
        coins.add(symbol_info['baseAsset'])
        coins.add(symbol_info['quoteAsset'])
    coins = sorted(list(coins))

    # Construct the square matrix
    n = len(coins)
    exchange_rate_matrix = np.empty((n, n))
    exchange_rate_matrix[:] = np.nan  # Fill matrix with NaN initially

    # Populate the matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                exchange_rate_matrix[i, j] = 1.0  # The rate from a coin to itself is 1
            else:
                # Check if there's a direct pair for conversion, use uppercase symbols
                forward_pair = coins[i].upper() + coins[j].upper()
                if forward_pair in exchange_rates:
                    exchange_rate_matrix[i, j] = exchange_rates[forward_pair]

    # Throw away the ones in the diagonal because they mean nothing
    # np.fill_diagonal(exchange_rate_matrix, np.nan)
    
    return exchange_rate_matrix, coins