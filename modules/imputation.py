###############################################################
#                       imputation.py                         #
#   Predict/Imputate with trained C2SDI model and visualize   #
###############################################################

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import time
from pypots.utils.metrics import calc_mae, calc_rmse, calc_quantile_crps

def visualize_data(data_mean, data_min, data_max, ori_data, sma_data_list, saving_path, image_idx, predicted):
  fig = plt.figure(figsize=(8, 6))
  ax = fig.add_subplot(111, projection='3d')
  
  # Using mean value 
  x_mean = data_mean[:, 0]
  y_mean = data_mean[:, 1]
  z_mean = data_mean[:, 2]
  ax.plot(x_mean, y_mean, z_mean, label='Imputation Mean', color='b')

  # Shading the margin of error
  for i in range(len(x_mean)):
    ax.plot([x_mean[i], x_mean[i]], [y_mean[i], y_mean[i]], [data_min[i, 2], data_max[i, 2]], color='lightblue', alpha=0.3)

  # Draw 3 randomly selected Imputation results
  random_indices = random.sample(range(len(sma_data_list)), 3)
  color = 'darkblue'

  for idx in random_indices:
    sma_data = sma_data_list[idx]
    x_data = sma_data[:, 0]
    y_data = sma_data[:, 1]
    z_data = sma_data[:, 2]
    ax.plot(x_data, y_data, z_data, color=color, alpha=0.4)

  # Original dataset
  x_ori = ori_data[:, 0]
  y_ori = ori_data[:, 1]
  z_ori = ori_data[:, 2]
  ax.plot(x_ori, y_ori, z_ori, label='Ground Truth', color='r')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.legend()
  plt.title('Data Visualization')

  if predicted:
    file_path = os.path.join(saving_path, "images_gt", str(image_idx) + ".png")
  else:
    file_path = os.path.join(saving_path, "images_pr", str(image_idx) + ".png")

  os.makedirs(os.path.dirname(file_path), exist_ok=True)

  plt.savefig(file_path)
  plt.close()


def imputation(test_dataset, orig_data, csdi, saving_path, predicted=False):
  # Predict and check elapsed time
  start_time = time.time()
  csdi_results = csdi.predict(test_dataset, n_sampling_times=10)
  end_time = time.time()

  csdi_imputation = csdi_results["imputation"]
  data = csdi_imputation.mean(axis=1)

  # Calculate mean absolute error on the ground truth (artificially-missing values)
  testing_mae = calc_mae(
    data,
    orig_data['test_X_ori'],
    orig_data['test_X_indicating_mask'],
  )

  testing_rmse = calc_rmse(
    data,
    orig_data['test_X_ori'],
    orig_data['test_X_indicating_mask'],
  )

  reee = calc_quantile_crps(
    csdi_imputation,
    orig_data['test_X_ori'],
    orig_data['test_X_indicating_mask'],
  )

  if (not predicted):
    test_MAE = f"With GT Label) MAE: {testing_mae:.4f} / RMSE: {testing_rmse:.4f} / CRPS: {reee:.4f} / Elapsed time per sample: {(end_time - start_time) / 10:.4f} s"

  else:
    test_MAE = f"With Predicted Label) MAE: {testing_mae:.4f} / RMSE: {testing_rmse:.4f} / CRPS: {reee:.4f} / Elapsed time per sample: {(end_time - start_time) / 10:.4f} s"


  # Visualize results and save with images
  # for i in range(len(orig_data['test_X'])):
  for i in range(1):
    data_lst = [csdi_imputation[i][j] for j in range(10)]
    ori_data = orig_data['test_X_ori'][i]

    sma_data_list = []

    for data in data_lst:
      df = pd.DataFrame(data, columns=['X', 'Y', 'Z'])
      df = df.sort_values(by='X')
      
      # Calculate a Simple Moving Average (SMA)
      window_size = 20  # Window size to calculate the moving average
      sma_data = df.rolling(window=window_size).mean().to_numpy()
      sma_data_list.append(sma_data)

    # Calculate the average of data with a SMA
    sma_data_mean = np.mean(sma_data_list, axis=0)

    # Calculate the minimum and maximum values of data using a SMA
    sma_data_min = np.min(sma_data_list, axis=0)
    sma_data_max = np.max(sma_data_list, axis=0)

    visualize_data(sma_data_mean, sma_data_min, sma_data_max, ori_data, sma_data_list, saving_path, i, predicted)


  return test_MAE
