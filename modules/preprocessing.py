##################################################################
#                       preprocessing.py                         #
#               Load dataset and preprocessing with              #
#                     interpolate and sampling                   #
##################################################################

import os
import pandas as pd
import numpy as np
import logging
import random
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

def uniform_sampling(data, interval):
    current_length = len(data)
    if interval <= 0 or interval > current_length:
        raise ValueError("Interval must be a positive integer and less than or equal to the length of the data.")
    # Calculate indices for uniform sampling based on the given interval
    sampled_indices = np.arange(0, current_length, interval)
    # Sample the data at the calculated indices
    sampled_data = data.iloc[sampled_indices].reset_index(drop=True)
    return sampled_data


def pad_data_to_max_length(data, max_length):
    padded_data = []
    for df in data:
        length = len(df)
        if length < max_length:
            # Create a DataFrame filled with NaNs for padding
            padding_length = max_length - length
            padding = pd.DataFrame(np.nan, index=range(padding_length), columns=df.columns)
            padded_df = pd.concat([df, padding], ignore_index=True)
        else:
            padded_df = df
        padded_data.append(padded_df)
    return padded_data


def preprocessing(dataset_path, interval, adjusted):
    pullup_path = os.path.join(dataset_path, 'pull_up_0913')
    non_pullup_path_BM1 = os.path.join(dataset_path, 'non_pull_up/BM1')
    
    # Get file lists
    pullup_files = [os.path.join(root, file) for root, dirs, files in os.walk(pullup_path) for file in files if file.endswith('.csv')]
    nonpullup_files_bm1 = [os.path.join(root, file) for root, dirs, files in os.walk(non_pullup_path_BM1) for file in files if file.endswith('.csv')]

    # Read and process data
    pullup_data = []
    pullup_cols = ['time', 'phase', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'lat', 'lon', 'alt']
    for file in tqdm(pullup_files):
        data = pd.read_csv(file, sep=',', header=0, names=pullup_cols)
        data.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z'}, inplace=True)
        pullup_data.append(data[['X', 'Y', 'Z', 'vx', 'vy', 'vz']]) 

    non_pullup_data_bm1 = []
    bm1_cols = ['time', 'lat', 'lon', 'alt', 'x', 'y', 'z', 'vx', 'vy', 'vz']
    for file in tqdm(nonpullup_files_bm1):
        data = pd.read_csv(file, header=None, names=bm1_cols)
        data.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z'}, inplace=True)
        non_pullup_data_bm1.append(data[['X', 'Y', 'Z', 'vx', 'vy', 'vz']]) 

    # Filter out short sequences
    non_pullup_data_bm1 = [df for df in non_pullup_data_bm1 if len(df) > 1]
    sampled_data, sampled_bm1, sampled_bm2 = [], [], []
    intervals = [interval / 0.1, interval / 0.1]
    for data in tqdm(pullup_data):
        init_data = data
        temp_data = uniform_sampling(init_data, intervals[0])
        sampled_data.append(temp_data)

    for data in tqdm(non_pullup_data_bm1):
        init_data = data
        temp_data = uniform_sampling(init_data, intervals[1])
        sampled_bm1.append(temp_data)

    pullup_data = sampled_data
    non_pullup_data_bm1 = sampled_bm1

    
    # Determine the maximum length of the sequences
    max_length = max(len(df) for df in pullup_data + non_pullup_data_bm1)
    logger.info(f"Maximum length for padding: {max_length}")
    # Adjust the max_length (default = 1.1)
    max_length = int(max_length * adjusted)
    logger.info(f"Adjusted maximum length for padding: {max_length}")

    # Pad the sequences with NaNs
    pullup_data = pad_data_to_max_length(pullup_data, max_length)
    non_pullup_data_bm1 = pad_data_to_max_length(non_pullup_data_bm1, max_length)
    
    # Convert DataFrames to NumPy arrays
    pullup_data_array = np.array([df.values for df in pullup_data])
    non_pullup_data_bm1_array = np.array([df.values for df in non_pullup_data_bm1])

    return pullup_data_array, non_pullup_data_bm1_array, max_length