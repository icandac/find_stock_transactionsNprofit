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
from src.functions import load_csv


def main():
    """
    This model...
    """
    # Read the input file
    df_input = load_csv()

    # Clean and sort the data
    print(df_input.isna().sum())  # No NaN values in the dataset
    print(df_input.isnull().sum())  # No null values in the dataset

    # df.describe().to_csv(path_or_buf="./outputs/data_summary.csv", sep=",", columns=df.columns)

    # Visualization

    add_to_log("The given data is read and cleaned.")
    a = 5
