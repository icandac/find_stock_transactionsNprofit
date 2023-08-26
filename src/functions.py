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
