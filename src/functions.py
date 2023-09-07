import random
import math
from tqdm import tqdm as tqdm_func


def calculate_profit(trades_df):
    """
     Calculate the total profit from a set of trades.

     Parameters:
     - trades_df (pd.DataFrame): A DataFrame containing the trades data.
                                 It should have columns 'Quantity' and 'Price'.

     Returns:
     - float: The total profit from the set of trades.
     """
    buys = trades_df[trades_df['Quantity'] > 0]
    sells = trades_df[trades_df['Quantity'] < 0]
    buy_amount = (buys['Quantity'] * buys['Price']).sum()
    sell_amount = (-sells['Quantity'] * sells['Price']).sum()
    return sell_amount - buy_amount


def simulated_annealing(df, initial_temp, cooling_rate, num_iterations):
    """
    Perform simulated annealing to identify a set of trades maximizing profit.

    Parameters:
    - df (pd.DataFrame): The DataFrame containing trades data.
    - initial_temp (float): The initial temperature for the simulated annealing algorithm.
    - cooling_rate (float): Rate at which the temperature decreases.
    - num_iterations (int): Number of iterations for the simulated annealing process.

    Returns:
    - list: Indices of the best trades identified.
    - float: Profit of the best trades identified.
    """
    # Initialize with a random set of trades that meet the criteria
    potential_trades = df[df['Potential_Trade']].index.tolist()
    current_solution = random.sample(potential_trades, 10)
    current_solution_df = df.loc[current_solution]
    current_profit = calculate_profit(current_solution_df)

    # Initialize variables to keep track of the best solution found
    best_solution = current_solution
    best_profit = current_profit

    # Temperature parameter
    temp = initial_temp

    # Main loop for simulated annealing
    for i in range(num_iterations):
        # Generate a neighbor solution (add or remove a random trade)
        neighbor_solution = current_solution.copy()
        if random.choice([True, False]):
            # Add a random trade
            new_trade = random.choice(potential_trades)
            neighbor_solution.append(new_trade)
        else:
            # Remove a random trade
            if len(neighbor_solution) > 1:  # Ensure we don't empty the list
                neighbor_solution.remove(random.choice(neighbor_solution))

        # Calculate profit for the neighbor solution
        neighbor_solution_df = df.loc[neighbor_solution]
        neighbor_profit = calculate_profit(neighbor_solution_df)

        # Calculate the change in profit
        delta_profit = neighbor_profit - current_profit

        # Acceptance criteria
        if delta_profit > 0:
            current_solution = neighbor_solution
            current_profit = neighbor_profit
        else:
            # Accept with a probability of exp(-delta_profit / temp)
            if random.random() < math.exp(delta_profit / temp):
                current_solution = neighbor_solution
                current_profit = neighbor_profit

        # Update the best solution if needed
        if current_profit > best_profit:
            best_solution = current_solution
            best_profit = current_profit

        # Cooling schedule (exponential cooling)
        temp *= cooling_rate

        # Termination criteria can be added here (if temp is too low or best_profit is good enough)
        if temp < 1e-5:
            break

    return best_solution, best_profit


def simulated_annealing_w_parameter_search(df):
    """
    Performs a grid search over specified parameter spaces to find the best parameters
    for the simulated annealing algorithm based on the profit achieved.

    Parameters:
    - df (pd.DataFrame): The dataset containing trade information. Expected columns include
                         'Quantity' and 'Price'.

    Returns:
    - float: The highest profit achieved in the grid search.
    - tuple: The set of parameters (initial_temp, cooling_rate, num_iterations) that achieved
             the highest profit.

    Notes:
    - The grid search explores combinations of initial temperatures from 500 to 5500 (inclusive)
      in increments of 500, cooling rates from 0.990 to 0.999 (inclusive) in increments of 0.001,
      and number of iterations from 500 to 5500 (inclusive) in increments of 500.
    - The search progress is visualized using a tqdm progress bar.
    """
    # Define the parameter space for grid search
    initial_temp_space = list(range(500, 5501, 500))
    cooling_rate_space = [i / 1000 + 0.990 for i in range(10)]
    num_iterations_space = list(range(500, 5501, 500))

    # Create all combinations of parameters for grid search
    grid_search_combinations = [(it, cr, ni) for it in initial_temp_space
                                for cr in cooling_rate_space
                                for ni in num_iterations_space]

    # Evaluate each combination in the grid
    grid_search_results = {}
    for combo in tqdm_func(grid_search_combinations, desc="Evaluating parameter combinations"):
        _, profit = simulated_annealing(df, combo[0], combo[1], combo[2])
        grid_search_results[combo] = profit

    # Find the best combination from grid search
    best_parameters_grid = max(grid_search_results, key=grid_search_results.get)
    best_profit_grid = grid_search_results[best_parameters_grid]

    return best_profit_grid, best_parameters_grid
