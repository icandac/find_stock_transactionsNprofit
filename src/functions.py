# Python built-in packages
import json
import os
from datetime import datetime

# Third-party packages
import numpy as np
import pandas as pd
from pathlib import Path

# Internal modules
# from src.utils import add_to_log, stop_model

def load_csv():
    """
    The method where the csv file Spot_Month_Input is read.

    Returns:
        (pd.DataFrame): csv file
    """

    csv_path = Path.cwd().joinpath("input", "data_v2.csv")

    with open(csv_path, "r", encoding="utf-8-sig") as read_file:
        input_csv = pd.read_csv(read_file)

    return input_csv


def greedy_algorithm(df):
    # Add a column for the PnL contribution of each trade
    df.loc[:, 'PnL_Contribution'] = df['Price'] * df['Quantity']

    # Sort the DataFrame based on PnL contribution
    df_sorted = df.sort_values(by='PnL_Contribution', ascending=False)

    # Initialize variables
    cumulative_position = 0
    total_pnl = 0

    # List to store selected trades
    selected_trades = []

    # Iterate through the sorted DataFrame
    for index, row in df_sorted.iterrows():
        # Check if adding the current trade breaches the position limit
        if abs(cumulative_position + row['Quantity']) <= 100000:
            cumulative_position += row['Quantity']
            total_pnl += row['PnL_Contribution']
            selected_trades.append(index)

    return selected_trades, total_pnl


def simulated_annealing(df, initial_temp, cooling_rate, num_iterations):
    # Initial solution (randomly select trades)
    current_solution = np.random.choice([0, 1], size=len(df))
    current_pnl = np.sum(current_solution * df['Price'] * df['Quantity'])

    current_temp = initial_temp

    for iteration in range(num_iterations):
        # Perturb the solution (flip a random trade)
        new_solution = current_solution.copy()
        random_index = np.random.randint(0, len(df))
        new_solution[random_index] = 1 - new_solution[random_index]

        # Calculate new PnL
        new_pnl = np.sum(new_solution * df['Price'] * df['Quantity'])

        # Change in PnL
        delta_pnl = new_pnl - current_pnl

        # Acceptance probability
        if delta_pnl > 0:
            acceptance_prob = 1
        else:
            acceptance_prob = np.exp(delta_pnl / current_temp)

        # Accept new solution based on acceptance probability
        if np.random.rand() < acceptance_prob:
            current_solution = new_solution
            current_pnl = new_pnl

        # Reduce temperature
        current_temp *= cooling_rate

    return current_solution, current_pnl
