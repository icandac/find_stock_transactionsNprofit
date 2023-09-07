import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm_func

import src.functions as func
from src.utils import add_to_log


def main():
    """
    Analyze the trades dataset to identify and evaluate the most profitable trades
    using the simulated annealing algorithm. This function also visualizes the analysis results.

    Returns:
    None
    """
    # Load the dataset
    file_path = './input/data_v2.csv'
    df = pd.read_csv(file_path)

    # Calculate the cumulative position over time
    df['Cumulative_Position'] = df['Quantity'].cumsum()

    # Add a column to mark potential trades for the ideal strategy
    # A trade is marked as potential if the absolute cumulative position does not exceed 100,000
    df['Potential_Trade'] = df['Cumulative_Position'].apply(lambda x: abs(x) <= 100000)

    # Show some rows where Potential_Trade is True
    print(df[df['Potential_Trade']].head())

    add_to_log("The dataset is successfully get.")

    # # Create all combinations of parameters for grid search
    # # This takes time but for the first run, it will make choose the correct set of parameters.
    # best_profit_grid, best_parameters_grid = func.simulated_annealing_w_parameter_search(df)

    # Initial parameters for simulated annealing
    initial_temp = 1000
    cooling_rate = 0.995
    num_iterations = 1000

    best_overall_trades = []
    best_overall_profit = float('-inf')

    # This can be passed by just getting num_runs = 1 for a single pass algorithm but the more the
    # number is increased, the more chance is get to obtain a higher profit
    num_runs = 100
    for _ in tqdm_func(range(num_runs), desc="Running Simulated Annealing"):
        trades, profit = func.simulated_annealing(df, initial_temp, cooling_rate, num_iterations)
        if profit > best_overall_profit:
            best_overall_profit = profit
            best_overall_trades = trades

    print(best_overall_profit, best_overall_trades[:10])

    # Extract the DataFrame for the best trades identified by simulated annealing
    best_trades_df = df.loc[best_overall_trades]

    # Show the first few rows of the DataFrame for the best trades
    print(best_trades_df.head())

    # Calculate the cumulative position for the best trades
    best_trades_df['Best_Trades_Cumulative_Position'] = best_trades_df['Quantity'].cumsum()

    # Calculate the overall profit for the best trades
    overall_profit = func.calculate_profit(best_trades_df)

    # Calculate the number of trades in the best set
    num_trades = len(best_trades_df)

    # Calculate the efficiency metric (Profit per Trade)
    efficiency = overall_profit / num_trades if num_trades > 0 else 0

    print(overall_profit, num_trades, efficiency)

    add_to_log("The analysis is finished, now the visuals will take part.")

    # Visualizations

    # Plotting
    plt.figure(figsize=(15, 6))
    plt.plot(best_trades_df['time_id'], best_trades_df['Best_Trades_Cumulative_Position'],
             marker='o', linestyle='-')
    plt.axhline(100000, color='r', linestyle='--', label='Max Position Limit')
    plt.axhline(-100000, color='r', linestyle='--', label='Min Position Limit')
    plt.title('Cumulative Position of Best Trades')
    plt.xlabel('Time ID')
    plt.ylabel('Cumulative Position')
    plt.legend()
    plt.grid(True)
    plt.savefig("./outputs/cumulative_positions.png", dpi=200, format='png', bbox_inches='tight')

    # Calculate the profit for each individual trade
    best_trades_df['Individual_Profit'] = best_trades_df['Quantity'] * best_trades_df['Price']

    # Plotting the distribution of profits for individual trades
    plt.figure(figsize=(15, 6))
    plt.hist(best_trades_df['Individual_Profit'], bins=50, color='blue', alpha=0.7,
             label='Individual Profit')
    plt.axvline(best_trades_df['Individual_Profit'].mean(), color='r', linestyle='--',
                label='Mean Profit')
    plt.title('Distribution of Profits for Individual Trades')
    plt.xlabel('Profit')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig("./outputs/profit_dist.png", dpi=200, format='png', bbox_inches='tight')

    # Calculate some statistics related to individual profits
    mean_profit = best_trades_df['Individual_Profit'].mean()
    median_profit = best_trades_df['Individual_Profit'].median()
    std_dev_profit = best_trades_df['Individual_Profit'].std()

    print(mean_profit, median_profit, std_dev_profit)

    # Group the best trades by time ID and calculate the sum of profits and the number of trades
    # for each time ID
    time_analysis_df = best_trades_df.groupby('time_id')['Individual_Profit'].agg(
        ['sum', 'count']).reset_index()

    # Plotting
    fig, axs = plt.subplots(2, 1, figsize=(15, 12))
    fig.suptitle('Time Analysis of Best Trades')

    # Plot total profit over time IDs using a line chart
    axs[0].plot(time_analysis_df['time_id'], time_analysis_df['sum'], marker='o', color='blue',
                alpha=0.7)
    axs[0].set_title('Total Profit by Time ID')
    axs[0].set_xlabel('Time ID')
    axs[0].set_ylabel('Total Profit')
    axs[0].grid(True)

    # Adjust y-axis limits for better visualization
    axs[0].set_ylim(min(time_analysis_df['sum']) * 1.1, max(time_analysis_df['sum']) * (- 1e4))

    # Plot number of trades over time IDs using a line chart
    axs[1].plot(time_analysis_df['time_id'], time_analysis_df['count'], marker='o', color='green',
                alpha=0.7)
    axs[1].set_title('Number of Trades by Time ID')
    axs[1].set_xlabel('Time ID')
    axs[1].set_ylabel('Number of Trades')
    axs[1].grid(True)

    # Adjust y-axis limits if needed
    axs[1].set_ylim(min(time_analysis_df['count']) - 1, max(time_analysis_df['count']) + 1)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("./outputs/trades_profits.png", dpi=200, format='png', bbox_inches='tight')

    # Plotting
    plt.figure(figsize=(15, 6))

    # Calculate how often the strategy approaches the maximum position size of 100,000
    approaching_max = best_trades_df[
        abs(best_trades_df['Best_Trades_Cumulative_Position']) >= 90000]

    # Use the index of approaching_max as x-values
    approaching_indices = list(approaching_max.index)

    # Highlight points where the position is near the maximum or minimum limit
    plt.scatter(approaching_indices, list(approaching_max['Best_Trades_Cumulative_Position']),
                color='red', label='Near Max Position')

    # Lines to indicate the max and min position limits
    plt.axhline(100000, color='r', linestyle='--', label='Max Position Limit')
    plt.axhline(-100000, color='r', linestyle='--', label='Min Position Limit')

    plt.title('Position Size Analysis of Best Trades')
    plt.xlabel('Trade Sequence')
    plt.ylabel('Cumulative Position')
    plt.legend()
    plt.grid(True)
    plt.savefig("./outputs/position_size_analysis.png", dpi=200, format='png', bbox_inches='tight')
