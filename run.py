import argparse
import sys
import torch
import numpy as np
import random
import os
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


parser = argparse.ArgumentParser(description='Argparser')

parser.add_argument('path', type=str, help='Enter the path with C2SDI git repository')
parser.add_argument('dataset_path', type=str, help='Enter the path of dataset')
parser.add_argument('--result_save_path', type=str, default='results/', help="Path for saving model and imputation results")
parser.add_argument('--seq_len', type=int, default=300, help='Sequence length of timeseries data')
parser.add_argument('--seed', type=int, default=42, help='Seed for overall running')

args = parser.parse_args()

git_path = args.path
dataset_path = args.dataset_path
results_path = args.result_save_path
seq_len = args.seq_len
seed = args.seed


#-------------------- Load C2SDI and fix seeds --------------------#
sys.path.insert(0, git_path)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(seed)
random.seed(seed)


#-------------------- Load Dataset --------------------#
from modules import preprocessing
pullup_data, non_pullup_data_bm1, non_pullup_data_bm2 = preprocessing(dataset_path=dataset_path)


#-------------------- Data Augmentation --------------------#
from modules import augmentation
augmented_pullup = augmentation(base_data=pullup_data)


#-------------------- Prepare Dataset --------------------#
class_0_data = augmented_pullup
class_1_data = non_pullup_data_bm1 + non_pullup_data_bm2

def split_data(data):
  train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
  valid_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
  return train_data, valid_data, test_data
 
train_class_0, valid_class_0, test_class_0 = split_data(class_0_data)
train_class_1, valid_class_1, test_class_1 = split_data(class_1_data)


# 먼저 각 클래스의 데이터를 리스트에서 3D NumPy 배열로 변환
train_class_0 = np.array(train_class_0)
valid_class_0 = np.array(valid_class_0)
test_class_0 = np.array(test_class_0)

train_class_1 = np.array(train_class_1)
valid_class_1 = np.array(valid_class_1)
test_class_1 = np.array(test_class_1)

# 클래스 0에 대한 정규화
scaler_0 = StandardScaler()
train_class_0_reshaped = train_class_0.reshape(-1, 3)
scaler_0.fit(train_class_0_reshaped)

train_class_0_normalized = scaler_0.transform(train_class_0_reshaped).reshape(train_class_0.shape)
valid_class_0_normalized = scaler_0.transform(valid_class_0.reshape(-1, 3)).reshape(valid_class_0.shape)
test_class_0_normalized = scaler_0.transform(test_class_0.reshape(-1, 3)).reshape(test_class_0.shape)

# 클래스 1에 대한 정규화
scaler_1 = StandardScaler()
train_class_1_reshaped = train_class_1.reshape(-1, 3)
scaler_1.fit(train_class_1_reshaped)

train_class_1_normalized = scaler_1.transform(train_class_1_reshaped).reshape(train_class_1.shape)
valid_class_1_normalized = scaler_1.transform(valid_class_1.reshape(-1, 3)).reshape(valid_class_1.shape)
test_class_1_normalized = scaler_1.transform(test_class_1.reshape(-1, 3)).reshape(test_class_1.shape)

# DataFrame으로 변환하는 함수
def to_dataframe(normalized_data):
    columns = ['X', 'Y', 'Z']
    return [pd.DataFrame(data, columns=columns) for data in normalized_data]

# 각 정규화된 데이터를 DataFrame 리스트로 변환
train_class_0_normalized_df = to_dataframe(train_class_0_normalized)
valid_class_0_normalized_df = to_dataframe(valid_class_0_normalized)
test_class_0_normalized_df = to_dataframe(test_class_0_normalized)

train_class_1_normalized_df = to_dataframe(train_class_1_normalized)
valid_class_1_normalized_df = to_dataframe(valid_class_1_normalized)
test_class_1_normalized_df = to_dataframe(test_class_1_normalized)

# training, validation, test 세트를 합치기
train_data = train_class_0_normalized_df + train_class_1_normalized_df
valid_data = valid_class_0_normalized_df + valid_class_1_normalized_df
test_data = test_class_0_normalized_df + test_class_1_normalized_df

print("Number of Training/ Validation / Test set: ", len(train_data), '/', len(valid_data), '/', len(test_data))

training_label = [0 for i in range(len(train_class_0))]
valid_label = [0 for i in range(len(valid_class_0))]
test_label = [0 for i in range(len(test_class_0))]

for i in range(len(train_class_1)):
  training_label.append(1)

for i in range(len(valid_class_1)):
  valid_label.append(1)

for i in range(len(test_class_1)):
  test_label.append(1)

X_list = [df[['X', 'Y', 'Z']] for df in train_data]
X_array = np.stack([df.values for df in X_list], axis=0)

training_df = {
  'X': X_array,
  'class_label': training_label,
}

X_list_v = [df[['X', 'Y', 'Z']] for df in valid_data]
X_array_v = np.stack([df.values for df in X_list_v], axis=0)

validatn_df = {
  'X': X_array_v,
  'class_label': valid_label,
}

X_list_t = [df[['X', 'Y', 'Z']] for df in test_data]
X_array_t = np.stack([df.values for df in X_list_t], axis=0)

testingg_df = {
  'X': X_array_t,
  'class_label': test_label,
}

missile_data = {
  "n_features": training_df['X'].shape[-1],
  "train_X": training_df['X'],
  
  "val_X": validatn_df['X'],
  "val_X_ori": validatn_df['X'],

  "test_X": testingg_df['X'],
  "test_X_ori": testingg_df['X'],

  "train_class_label": training_df['class_label'],
  "val_class_label": validatn_df['class_label'],
  "test_class_label": testingg_df['class_label']
}


#-------------------- Masking exclude initial parts --------------------#
def mask_middle_portion_exclude_initial(data, rate, axis=1):
  data = data.copy()
  n = data.shape[axis]
  
  initial_exclude_rate = rate
  initial_exclude_count = int(n * initial_exclude_rate)
  
  start_idx = initial_exclude_count
  
  if axis == 1:
      data[:, start_idx:, :] = np.nan
  elif axis == 0:
      data[start_idx:, :, :] = np.nan
  else:
      raise ValueError("Axis should be 0 or 1.")
      
  return data

remaining_rate = 0.3

if remaining_rate > 0:
  # mask values in the validation set as ground truth
  val_X_ori = validatn_df['X']
  val_X = mask_middle_portion_exclude_initial(validatn_df['X'], remaining_rate)

  test_X_ori = testingg_df['X']
  test_X = mask_middle_portion_exclude_initial(testingg_df['X'], remaining_rate)

  missile_data["val_X"] = val_X
  missile_data["val_X_ori"] = val_X_ori

  missile_data["test_X"] = test_X
  missile_data["test_X_ori"] = np.nan_to_num(test_X_ori)
  missile_data["test_X_indicating_mask"] = ~np.isnan(test_X_ori) ^ ~np.isnan(test_X)
  
dataset_for_training = {
    "X": missile_data['train_X'],
    "class_label": missile_data['train_class_label']
}

dataset_for_validating = {
    "X": missile_data['val_X'],
    "X_ori": missile_data['val_X_ori'],
    "class_label": missile_data['val_class_label']
}

dataset_for_testing = {
    "X": missile_data['test_X'],
    "class_label": missile_data['test_class_label']
}


#-------------------- Start Training --------------------#
from modules import train
csdi = train(dataset_for_training, dataset_for_validating, n_features=missile_data['n_features'], saving_path=results_path)


#-------------------- Testing (Imputation) --------------------#
from modules import imputation
gt_res = imputation(dataset_for_testing, orig_data=missile_data, csdi=csdi, saving_path=results_path)


#-------------------- Predict Binary Class Label --------------------#
from modules import classification

predicted_label = classification(missile_data=missile_data, saving_path=results_path)

import copy
pred_test = copy.deepcopy(dataset_for_testing)
pred_test['class_label'] = predicted_label

pr_res = imputation(pred_test, orig_data=missile_data, csdi=csdi, saving_path=results_path, predicted=True)

print(gt_res)
print(pr_res)
