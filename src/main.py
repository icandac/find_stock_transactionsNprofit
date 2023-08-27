# Python libraries
import os
from pathlib import Path

# Third-party packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Internal modules
from src.utils import add_to_log
from src.functions import load_csv, greedy_algorithm, simulated_annealing


def main():
    """
    This model...
    """
    # Read the input file
    df_input = load_csv()
    df = df_input.copy()

    # Clean and sort the data
    print(df.isna().sum())  # No NaN values in the dataset
    print(df.isnull().sum())  # No null values in the dataset

    df.describe().to_csv(path_or_buf="./outputs/data_summary.csv", sep=",", columns=df.columns)

    add_to_log("The given data is read and cleaned.")

    # Analysis and calculation
    # We assume the initial position of the "ideal trader(s)" is zero.

    # Since we checked and saw that there are positions quite large, let's first filter the dataset
    # by removing the positions exceeding 200000, since they always make the position larger than
    # 100000 or less than -100000.
    filtered_df = df[abs(df['Quantity']) <= 200000]

    # Multiply the positions with -1 since actually sells, i.e. negative positions, cumulatively
    # effect the PnL and we are going to optimize the PnL.
    filtered_df.loc[:, 'Quantity'] = filtered_df['Quantity'] * -1

    # Assume the "ideal strategy" means a strategy that makes the owner earns the most PnL
    filt_df = filtered_df.copy(deep=True)
    # Apply greedy algortihm
    # selected_trades, max_pnl = greedy_algorithm(filt_df)
    # print(f"Selected trades indices: {selected_trades}")
    # print(f"Number of selected trades: {len(selected_trades)}")
    # print(f"Maximum PnL: {max_pnl}")

    # Parameters for simulated annealing
    initial_temp = 1000
    cooling_rate = 0.995
    num_iterations = 10000

    solution, max_pnl = simulated_annealing(filt_df, initial_temp, cooling_rate,
                                            num_iterations)
    selected_trade_indices = np.where(solution == 1)[0]
    print(f"Number of selected trades indices: {len(selected_trade_indices)}")
    print(f"Maximum PnL: {max_pnl}")

    # # Visualization
    # # Plot all trades
    # sns.scatterplot(data=filt_df, x=filt_df.index, y='Quantity', label='All Trades')
    #
    # # Overlay the "ideally" traded ones with red circles
    # ideal_trades = filt_df.loc[selected_trades]
    # sns.scatterplot(data=ideal_trades, x=ideal_trades.index, y='Quantity', color='red',
    #                 label='Ideal Trades', s=50)
    #
    # plt.xlabel('Trade Index')
    # plt.ylabel('Quantity')
    # plt.title('All Trades vs. Ideal Trades')
    # plt.legend()
    # plt.show()
