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
from datetime import datetime
from tqdm import tqdm
from pypots.utils.metrics import calc_mae, calc_rmse, calc_quantile_crps
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

def visualize_data(data_mean, data_min, data_max, ori_data, sma_data_list, saving_path, image_idx, predicted, indi_len, real_test):

  nan_mask = indi_len.copy()

  nan_array = np.where(nan_mask == False, np.nan, nan_mask)
  indi_len =  nan_array

  orig_nan_array = nan_array.copy()

  for idx in range(len(orig_nan_array)):
    if np.isnan(orig_nan_array[idx][0]):
      if not np.isnan(ori_data[:, 0][idx]):
          orig_nan_array[idx] = [1.0, 1.0, 1.0]

  for idx in range(len(orig_nan_array)):
    if orig_nan_array[idx][0] == nan_array[idx][0]:
      orig_nan_array[idx] = [np.nan, np.nan, np.nan]

  orig_indi_len = orig_nan_array


  init_mask_array = nan_array.copy()
  real_test_data = real_test[:, :3]
  init_flag = False

  for idx in range(len(init_mask_array)):
    if init_flag:
      init_mask_array[idx] = [np.nan, np.nan, np.nan]
    elif not init_flag and not np.isnan(ori_data[idx][0]):
      init_flag = True
    elif not init_flag and np.isnan(ori_data[idx][0]):
      init_mask_array[idx] = [1.0, 1.0, 1.0]

  init_indi_len = init_mask_array

  fig = plt.figure(figsize=(10, 8))
  ax = fig.add_subplot(111, projection='3d')
  
  # Using mean value 
  x_mean = data_mean[:, 0] * indi_len[:, 0]
  y_mean = data_mean[:, 1] * indi_len[:, 1]
  z_mean = data_mean[:, 2] * indi_len[:, 2]
  ax.plot(x_mean, y_mean, z_mean, label='Imputation Mean', color='b')
  

  # Shading the margin of error
  # for i in range(len(x_mean)):
    # ax.plot([x_mean[i], x_mean[i]], [y_mean[i], y_mean[i]], [data_min[i, 2], data_max[i, 2]], color='lightblue', alpha=0.3)

  # Draw 1 randomly selected Imputation results
  random_indices = random.sample(range(len(sma_data_list)), 1) ### 3개 -> 1개
  color = 'darkblue'

  for idx in random_indices:
    sma_data = sma_data_list[idx]
    x_data = sma_data[:, 0] * indi_len[:, 0]
    y_data = sma_data[:, 1] * indi_len[:, 1]
    z_data = sma_data[:, 2] * indi_len[:, 2]
    ax.plot(x_data, y_data, z_data, color=color, alpha=0.4)

  #! Initial part
  x_ori = real_test_data[:, 0] * init_indi_len[:, 0]
  y_ori = real_test_data[:, 1] * init_indi_len[:, 1]
  z_ori = real_test_data[:, 2] * init_indi_len[:, 2]
  ax.plot(x_ori, y_ori, z_ori, label='Initial Parts', color='black')

  #! Observation part
  x_ori = ori_data[:, 0] * orig_indi_len[:, 0]
  y_ori = ori_data[:, 1] * orig_indi_len[:, 1]
  z_ori = ori_data[:, 2] * orig_indi_len[:, 2]
  ax.plot(x_ori, y_ori, z_ori, label='Observation', color='green')

  # Original dataset
  x_ori = ori_data[:, 0] * indi_len[:, 0]
  y_ori = ori_data[:, 1] * indi_len[:, 1]
  z_ori = ori_data[:, 2] * indi_len[:, 2]
  ax.plot(x_ori, y_ori, z_ori, label='Ground Truth', color='r')

  ax.set_xlabel('X')
  ax.set_ylabel('Y')
  ax.set_zlabel('Z')
  ax.legend()
  plt.title('Data Visualization')


  ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
  ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))
  ax.zaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))


  ax.tick_params(axis='x', rotation=45)
  ax.tick_params(axis='y', rotation=45)
  ax.tick_params(axis='z', rotation=45)

  plt.tight_layout()
  

  if predicted:
    file_path = os.path.join(saving_path, f"images_pr", str(image_idx) + ".png")
  else:
    file_path = os.path.join(saving_path, f"images_gt", str(image_idx) + ".png") # not used anymore

  os.makedirs(os.path.dirname(file_path), exist_ok=True)

  plt.savefig(file_path, dpi=300, bbox_inches='tight') 
  plt.close()


def calculate_metrics_with_scaling(data, csdi_imputation, orig_data, data_idx_list, rate_list, scaler):
  mae_calc_list = []
  rmse_calc_list = []
  crps_calc_list = []

  for rate in tqdm(rate_list, desc="Calculating metrics"):
      rres_mae = 0
      rres_rmse = 0
      rres_crps = 0
      
      for i in range(data.shape[0]):
          rate_data = [data[i][j] for j in data_idx_list[i][:int(len(data_idx_list[i]) * rate)]]
          rate_data_for_crps = [csdi_imputation[i][0][j] for j in data_idx_list[i][:int(len(data_idx_list[i]) * rate)]]
          rate_orig = [orig_data['test_X_ori'][i][j] for j in data_idx_list[i][:int(len(data_idx_list[i]) * rate)]]
          rate_mask = [orig_data['test_X_indicating_mask'][i][j] for j in data_idx_list[i][:int(len(data_idx_list[i]) * rate)]]
          
          # Reshape data for inverse transform
          rate_data = np.array(rate_data)
          rate_data_for_crps = np.array(rate_data_for_crps)
          rate_orig = np.array(rate_orig)
          
          if rate_data.size == 0:
              break

          # Apply inverse transform & flatten
          rate_data_inverse = scaler.inverse_transform(rate_data).flatten()
          rate_data_inverse_for_crps = scaler.inverse_transform(rate_data_for_crps).flatten()
          rate_orig_inverse = scaler.inverse_transform(rate_orig).flatten()
          rate_mask = np.array(rate_mask).flatten()
          
          testing_mae = calc_mae(
              rate_data_inverse,
              np.nan_to_num(rate_orig_inverse, nan=0.0),
              rate_mask
          )

          testing_rmse = calc_rmse(
              rate_data_inverse.flatten(),
              np.nan_to_num(rate_orig_inverse, nan=0.0),
              rate_mask
          )
          
          testing_crps = calc_quantile_crps(
              rate_data_inverse_for_crps.reshape(1, -1, data.shape[-1]),
              np.nan_to_num(rate_orig_inverse, nan=0.0).reshape(1, -1, data.shape[-1]),
              rate_mask.reshape(1, -1, data.shape[-1])
          )

          rres_mae += testing_mae
          rres_rmse += testing_rmse
          rres_crps += testing_crps

      rres_mae /= data.shape[0]
      rres_rmse /= data.shape[0]
      rres_crps /= data.shape[0]

      mae_calc_list.append(rres_mae)
      rmse_calc_list.append(rres_rmse)
      crps_calc_list.append(rres_crps)

  return mae_calc_list, rmse_calc_list, crps_calc_list



def imputation(test_dataset, orig_data, real_test, csdi, saving_path, scaler, n_sampling_times, batch_size, predicted=False, viss=True):
  # Predict and check elapsed time
  start_time = time.time()
  csdi_results = csdi.predict(test_dataset, n_sampling_times=n_sampling_times)
  end_time = time.time()

  csdi_imputation = csdi_results["imputation"]
  data = csdi_imputation.mean(axis=1)
  data_idx_list = []
  for _ in range(data.shape[0]):
    data_idx_list.append([])

  for i in range(data.shape[0]):
    for idx in range(len(orig_data['test_X_indicating_mask'][i])):
      if orig_data['test_X_indicating_mask'][i][idx][0]: data_idx_list[i].append(idx)



  ############## Calculating metrics ##############

  rate_list = [0.2, 0.4, 0.6, 0.8, 1.0]
  mae_results, rmse_results, crps_results = calculate_metrics_with_scaling(data, csdi_imputation, orig_data,
                                                                           data_idx_list, rate_list, scaler)

  ############## Calculating MAE ##############
  print(" MAE with ", end="")
  for i in range(5):
    if i != 4:
      print(f"Rate {rate_list[i]:.1f}: {mae_results[i]:.4f} / ", end='')
    else:
      print(f"Rate {rate_list[i]:.1f}: {mae_results[i]:.4f}")


  ############## Calculating RMSE ##############
  print("RMSE with ", end="")
  for i in range(5):
    if i != 4:
      print(f"Rate {rate_list[i]:.1f}: {rmse_results[i]:.4f} / ", end='')
    else:
      print(f"Rate {rate_list[i]:.1f}: {rmse_results[i]:.4f}")


  ############## Calculating CRPS ##############
  print("CRPS with ", end="")
  for i in range(5):
    if i != 4:
      print(f"Rate {rate_list[i]:.1f}: {crps_results[i]:.4f} / ", end='')
    else:
      print(f"Rate {rate_list[i]:.1f}: {crps_results[i]:.4f}")

  logger.info(f"Elapsed time per sample: {(end_time - start_time) / (len(test_dataset) * 4 * batch_size):.4f} s")

  if (not predicted):  
    test_metric = "Imputation Done. Please check the outputs."
  else:
    test_metric = "Imputation Done. Please check the outputs."


  # Visualize results and save with images
  for i in tqdm(range(len(orig_data['test_X'])), total=len(orig_data['test_X']), desc="Visualization"):
    data_lst = [csdi_imputation[i][j] for j in range(n_sampling_times)]
    inverse_lst = []

    # Use the same scaler for both classes
    for data in data_lst:
      inverse_lst.append(scaler.inverse_transform(data))

    data_lst = inverse_lst

    # Use the same scaler for original data
    ori_data = scaler.inverse_transform(orig_data['test_X_ori'][i])

    sma_data_list = []

    for data in data_lst:
      df = pd.DataFrame(data, columns=['X', 'Y', 'Z'])
      
      # Calculate a Simple Moving Average (SMA)
      window_size = 10  # Window size to calculate the moving average 
      sma_data = df.rolling(window=window_size).mean().to_numpy()
      sma_data_list.append(sma_data)

    # Calculate the average of data with a SMA
    sma_data_mean = np.mean(sma_data_list, axis=0)

    # Calculate the minimum and maximum values of data using a SMA
    sma_data_min = np.min(sma_data_list, axis=0)
    sma_data_max = np.max(sma_data_list, axis=0)

    ### Check indicating_mask: False -> True
    indi_len = orig_data['test_X_indicating_mask'][i]
    
    if viss:
      visualize_data(sma_data_mean, sma_data_min, sma_data_max, ori_data, sma_data_list, saving_path, i, predicted, indi_len, real_test[i])


  return test_metric