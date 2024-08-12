##################################################################
#                       preprocessing.py                         #
#               Load dataset and preprocessing with              #
#                     interpolate and sampling                   #
##################################################################

import os
import pandas as pd
from scipy.interpolate import interp1d
import numpy as np

def interpolate_data(data, desired_length):
  """
  Interpolates the data to increase its length to the desired length.
  
  Parameters:
  data (pd.DataFrame): The input data as a pandas dataframe.
  desired_length (int): The desired length of the data.
  
  Returns:
  pd.DataFrame: The interpolated data.
  """
  current_length = len(data)
  if current_length >= desired_length:
      return data
  
  # Generate new indices for interpolation
  new_indices = np.linspace(0, current_length - 1, num=desired_length)
  
  # Interpolate each column
  interpolated_data = pd.DataFrame()
  for column in data.columns:
      interp_func = interp1d(np.arange(current_length), data[column], kind='linear')
      interpolated_data[column] = interp_func(new_indices)
  
  return interpolated_data


def uniform_sampling(data, desired_length):
  """
  Uniformly samples the data to increase its length to the desired length.
  Parameters:
  data (pd.DataFrame): The input data as a pandas dataframe.
  desired_length (int): The desired length of the data.
  Returns:
  pd.DataFrame: The uniformly sampled data.
  """

  current_length = len(data)
  if current_length <= desired_length:
      return data
  # Calculate indices for uniform sampling
  sampled_indices = np.linspace(0, current_length - 1, num=desired_length).astype(int)
  # Sample the data at the calculated indices
  sampled_data = data.iloc[sampled_indices].reset_index(drop=True)
  return sampled_data


def preprocessing(dataset_path):
  pullup_path = os.path.join(dataset_path, 'pull_up')
  non_pullup_path_BM1 = os.path.join(dataset_path, 'non_pull_up/BM1')
  non_pullup_path_BM2 = os.path.join(dataset_path, 'non_pull_up/bm2')

  pullup_file = os.walk(pullup_path)
  nonpullup_file_bm1 = os.walk(non_pullup_path_BM1)
  nonpullup_file_bm2 = os.walk(non_pullup_path_BM2)


  pullup_list = []
  nonpullup_list_bm1 = []
  nonpullup_list_bm2 = []

  for root, dirs, files in pullup_file:
    pullup_list.extend(files)

  for root, dirs, files in nonpullup_file_bm1:
    for f in files:
      nonpullup_list_bm1.append(os.path.join(root, f))
  for root, dirs, files in nonpullup_file_bm2:
    for f in files:
      nonpullup_list_bm2.append(os.path.join(root, f))

  pullup_list = [file for file in pullup_list if file.endswith('.csv')]
  nonpullup_list_bm1 = [file for file in nonpullup_list_bm1 if file.endswith('.csv')]
  nonpullup_list_bm2 = [file for file in nonpullup_list_bm2 if file.endswith('.csv')]


  pullup_data = []
  pullup_col = ['time', 'temp', 'x', 'y', 'z', 'vx', 'vy', 'vz', 'lat', 'lon', 'alt']

  for csv_name in pullup_list:
    file_path = os.path.join(pullup_path, csv_name)
    data = pd.read_csv(file_path, delim_whitespace=True, header=None, names=pullup_col)
    data.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z'}, inplace=True)
    pullup_data.append(data)


  updated_pullup = []
  columns_to_keep = ['X', 'Y', 'Z']
                    
  for df in pullup_data:
    df = df.filter(items=columns_to_keep)
    updated_pullup.append(df)

  pullup_data = updated_pullup


  non_pullup_data_bm1 = []
  non_pullup_data_bm2 = []

  bm1_col = ['time', 'lat', 'lon', 'alt', 'x', 'y', 'z', 'vx', 'vy', 'vz']
  bm2_col = ['time', 'x', 'y', 'z', 'lat', 'lon', 'alt', 'acc', 'vx', 'vy', 'vz']

  for csv_name in nonpullup_list_bm1:
    file_path = os.path.join(non_pullup_path_BM1, csv_name)
    data = pd.read_csv(file_path, header=None, names=bm1_col)
    data.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z'}, inplace=True)
    non_pullup_data_bm1.append(data)

  for csv_name in nonpullup_list_bm2:
    file_path = os.path.join(non_pullup_path_BM2, csv_name)
    data = pd.read_csv(file_path, header=None, names=bm2_col, skiprows=1)
    data.rename(columns={'x': 'X', 'y': 'Y', 'z': 'Z'}, inplace=True)
    non_pullup_data_bm2.append(data)


  updated_non_pullup_bm1 = []
  updated_non_pullup_bm2 = []
  columns_to_keep = ['X', 'Y', 'Z']
                    
  for df in non_pullup_data_bm1:
    df = df.filter(items=columns_to_keep)
    updated_non_pullup_bm1.append(df)

  for df in non_pullup_data_bm2:
    df = df.filter(items=columns_to_keep)
    updated_non_pullup_bm2.append(df)

  filtered_df_list_1 = [df for df in updated_non_pullup_bm1 if len(df) > 1]
  filtered_df_list_2 = [df for df in updated_non_pullup_bm2 if len(df) > 1]

  non_pullup_data_bm1 = filtered_df_list_1
  non_pullup_data_bm2 = filtered_df_list_2


  # Make length same with interpolate / sampling

  sampled_data, sampled_bm1, sampled_bm2 = [], [], []
  desired_length = 300

  for data in pullup_data:
    original_length = len(data)
    if original_length < desired_length:
      sampled_data.append(interpolate_data(data, desired_length))
    else:
      sampled_data.append(uniform_sampling(data, desired_length))

  for data in non_pullup_data_bm1:
    original_length = len(data)
    if original_length < desired_length:
      sampled_bm1.append(interpolate_data(data, desired_length))
    else:
      sampled_bm1.append(uniform_sampling(data, desired_length))

  for data in non_pullup_data_bm2:
    original_length = len(data)
    if original_length < desired_length:
      sampled_bm2.append(interpolate_data(data, desired_length))
    else:
      sampled_bm2.append(uniform_sampling(data, desired_length))

  pullup_data = sampled_data
  non_pullup_data_bm1 = sampled_bm1
  non_pullup_data_bm2 = sampled_bm2


  return pullup_data, non_pullup_data_bm1, non_pullup_data_bm2
