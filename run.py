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
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)


#-------------------- Entering arguments --------------------#

parser = argparse.ArgumentParser(description='Argparser')
parser.add_argument('--path', default="/mlainas/seonggyun/C2SDI_final/C2SDI_new/", type=str, help='Enter the path with C2SDI git repository')
parser.add_argument('--dataset_path', default="/data1/seonggyun/hanhwa/20240826_data/", type=str, help='Enter the path of dataset')
# parser.add_argument('--dataset_path', default="/data1/seonggyun/hanhwa/20240826_data_toy36/", type=str, help='Enter the path of dataset')
parser.add_argument('--result_save_path', type=str, default='results/', help="Path for saving model and imputation results")
parser.add_argument('--csdi_model_path', type=str, default=None, help='''Path for existing C2SDI model.
                                                                         If there are not existiing models, leave it empty.''')
parser.add_argument('--classifier_model_path', type=str, default=None, help='''Path for existing classifier model.
                                                                               If there are not existing models, leave it empty.''')
parser.add_argument('--csdi_scaler_path', type=str, default=None, help='''Path for existing csdi scaler (3 dimension - X, Y, Z).
                                                                               If there are not existing models, leave it empty.''')
parser.add_argument('--classifier_scaler_path', type=str, default=None, help='''Path for existing classifier scaler (6 dimension - X, Y, Z, vX, vY, vZ).
                                                                               If there are not existing models, leave it empty.''')
                                                                               

# hyperparameters
parser.add_argument('--time_interval', type=float, default=1.0, help='Sampling interval to shorten the sequence length')
parser.add_argument('--adjusted', type=float, default=1.1, help='Drainage to adjust maximum length')
parser.add_argument('--model_epochs', type=int, default=80, help='The number of epochs for training the C2SDI')
parser.add_argument('--batch_size', type=int, default=64, help='The batch size for training and evaluating the C2SDI')
parser.add_argument('--patience', type=int, default=10, help='''The patience for the early-stopping mechanism. Given a positive integer, the training process will be
                                                                stopped when the model does not perform better after that number of epochs.
                                                                Leaving it default as None will disable the early-stopping.''')
parser.add_argument('--classifier_epochs', type=int, default=50, help='The number of epochs for training pullup classifier')
parser.add_argument('--use_augmentation', action='store_true', help='Augmentation decision')
parser.add_argument('--num_aug', type=int, default=11, help='The number of augmented samples for each datum')
parser.add_argument('--missing_rate', type=float, default=0.05, help='Artificial missing rate for data')


# for data split
# example: train_ratio = 0.7 / val_ratio = 0.5 => tr/val/te: 0.7/0.15/0.15
parser.add_argument('--train_ratio', type=float, default=0.7, help='Training ratio among all data')
parser.add_argument('--val_ratio', type=float, default=0.5, help='Test/validation data ratio')
parser.add_argument('--seed', type=int, default=42, help='Seed for overall running')

# for test
# example: test_rate=0.2 => observed dataset ratio=0.2
parser.add_argument('--n_sampling_times', type=int, default=1, help='The number of sampling times for testing the model to sample from the diffusion process')
parser.add_argument('--test_rate', type=float, default=0.2, help='A rate of conditions for test')
parser.add_argument('--use_impact_point', action='store_true',help='Impact point given')


args = parser.parse_args()
current_path = os.getcwd()

git_path = args.path
dataset_path = args.dataset_path
results_path = args.result_save_path
csdi_scaler_path = args.csdi_scaler_path
classifier_scaler_path = args.classifier_scaler_path

csdi_inference_mode = False
classifier_inference_mode = False
csdi_model_path = args.csdi_model_path
classifier_model_path = args.classifier_model_path

if args.csdi_model_path is not None:
   csdi_inference_mode = True

if args.classifier_model_path is not None:
   classifier_inference_mode = True

time_interval = args.time_interval
adjusted = args.adjusted
model_epochs = args.model_epochs
batch_size = args.batch_size
classifier_epochs = args.classifier_epochs
patience = args.patience
use_augmentation = args.use_augmentation
num_aug = args.num_aug
missing_rate = args.missing_rate

train_ratio = args.train_ratio
val_ratio = args.val_ratio
seed = args.seed

n_sampling_times = args.n_sampling_times
test_rate = args.test_rate
use_impact_point = args.use_impact_point



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
logger.info("Preprocessing ... ")
pullup_data, non_pullup_data_bm1, max_length = preprocessing(dataset_path=dataset_path, interval=time_interval, adjusted=adjusted)
logger.info("Preprocessing done.")


#-------------------- Prepare Dataset --------------------#
logger.info("Whole dataset loading ...")
class_0_data = pullup_data
class_1_data = non_pullup_data_bm1

logger.info("Dataset prepared. split for train/val/test ...")


def split_data(data):
  train_data, temp_data = train_test_split(data, test_size=1-train_ratio, random_state=seed)
  valid_data, test_data = train_test_split(temp_data, test_size=1-val_ratio, random_state=seed)
  return train_data, valid_data, test_data

train_class_0, valid_class_0, test_class_0 = split_data(class_0_data)
train_class_1, valid_class_1, test_class_1 = split_data(class_1_data)



#-------------------- Initial data masking --------------------#
def initial_masking(data):
    rand_num = random.randint(45, 55) ### when interval=1.0
    for i in range(len(data)):
      data[i][:rand_num] = np.nan

    return data

# for visualization
real_test = np.concatenate([test_class_0, test_class_1], axis=0)

# initial data masking
train_class_0, valid_class_0, test_class_0 = initial_masking(train_class_0), initial_masking(valid_class_0), initial_masking(test_class_0)
train_class_1, valid_class_1, test_class_1 = initial_masking(train_class_1), initial_masking(valid_class_1), initial_masking(test_class_1)



#-------------------- Data Augmentation (recently for non-pullup data) --------------------#
from modules import augmentation
if use_augmentation:
  train_class_1 = augmentation(base_data=train_class_1, num_aug=num_aug)
  ## if you want to use augmentation for validation data, activate the line below.
  # valid_class_1 = augmentation(base_data=valid_class_1, num_aug=num_aug) 
  logger.info("Data Augmentation done.")
else:
  logger.info("Data Augmentation isn't applied.")


train_data = np.concatenate([train_class_0, train_class_1], axis=0)


#-------------------- Catching Impact Point --------------------#

def find_last_real_index(data):
    # index for non-NaN value
    real_indices = np.where(~np.isnan(data[:, 0]))[0]
    
    # if real_indices is not empty, return the last index.
    # if real_indices is empty, return -1.
    return real_indices[-1] if real_indices.size > 0 else -1

# saving the last index for real values
train_data_last_indices = [find_last_real_index(data) for data in train_data]
valid_class_0_last_indices = [find_last_real_index(data) for data in valid_class_0]
valid_class_1_last_indices = [find_last_real_index(data) for data in valid_class_1]
test_class_0_last_indices = [find_last_real_index(data) for data in test_class_0]
test_class_1_last_indices = [find_last_real_index(data) for data in test_class_0]

valid_data_last_indices = np.concatenate([valid_class_0_last_indices, valid_class_1_last_indices])
test_data_last_indices = np.concatenate([test_class_0_last_indices, test_class_1_last_indices])


#-------------------- Injecting Artificially Missing Values --------------------#


def artificial_missing_rows(data, missing_rate):
    if not isinstance(data, np.ndarray):
        raise TypeError("Input data must be a numpy array")
    
    if data.ndim != 3:
        raise ValueError("Input data must be 3-dimensional with shape (a, b, c)")
    
    data = data.copy()
    
    num_subarrays, num_rows, _ = data.shape
    num_missing_rows = int(num_rows * missing_rate)
    
    for i in range(num_subarrays): 
        missing_row_indices = np.random.choice(num_rows, num_missing_rows, replace=False)
        data[i, missing_row_indices, :] = np.nan
    
    return data

train_data = artificial_missing_rows(train_data, missing_rate)
valid_class_0 = artificial_missing_rows(valid_class_0, missing_rate)
valid_class_1 = artificial_missing_rows(valid_class_1, missing_rate)
test_class_0 = artificial_missing_rows(test_class_0, missing_rate)
test_class_1 = artificial_missing_rows(test_class_1, missing_rate)


#-------------------- Normalization --------------------#
from pickle import dump, load

if csdi_scaler_path is None:
  scaler_for_C2SDI = StandardScaler()
  scaler_for_C2SDI.fit(train_data[:, :, :3].reshape(-1, 3))

  # save the scaler
  dump(scaler_for_C2SDI, open('.scaler_for_C2SDI.pkl','wb'))
  logger.info("Scaler for C2SDI is saved.")

else:
  logger.info("Scaler for C2SDI received is used.")
  scaler_for_C2SDI = load(open(csdi_scaler_path,'rb'))
  scaler_for_C2SDI.fit(train_data[:, :, :3].reshape(-1, 3))


if classifier_scaler_path is None:
  scaler_for_classifier = StandardScaler()
  scaler_for_classifier.fit(train_data[:, :, :3].reshape(-1, 6))

  # save the scaler
  from pickle import dump
  dump(scaler_for_classifier, open('.scaler_for_classifier.pkl','wb'))
  logger.info("Scaler for pull-up classifier is saved.")

else:
  logger.info("Scaler for classifier received is used.")
  scaler_for_classifier = load(open(classifier_scaler_path,'rb'))
  scaler_for_classifier.fit(train_data[:, :, :3].reshape(-1, 6))


def normalize_and_convert(data, scaler, dim):
    data_array = np.array(data)
    data_reshaped = data_array.reshape(-1, dim)
    valid_indices = ~np.isnan(data_reshaped).any(axis=1)
    data_normalized = np.full(data_reshaped.shape, np.nan)
    data_normalized[valid_indices] = scaler.transform(data_reshaped[valid_indices])

    return data_normalized.reshape(data_array.shape)

# for C2SDI
train_data_normalized = normalize_and_convert(train_data[:, :, :3], scaler_for_C2SDI, 3)
valid_class_0_normalized = normalize_and_convert(valid_class_0[:, :, :3], scaler_for_C2SDI, 3)
valid_class_1_normalized = normalize_and_convert(valid_class_1[:, :, :3], scaler_for_C2SDI, 3)
test_class_0_normalized = normalize_and_convert(test_class_0[:, :, :3], scaler_for_C2SDI, 3)
test_class_1_normalized = normalize_and_convert(test_class_1[:, :, :3], scaler_for_C2SDI, 3)

# for pullup classifier
train_data_normalized_classifier = normalize_and_convert(train_data, scaler_for_classifier, 6)
valid_class_0_normalized_classifier = normalize_and_convert(valid_class_0, scaler_for_classifier, 6)
valid_class_1_normalized_classifier = normalize_and_convert(valid_class_1, scaler_for_classifier, 6)
test_class_0_normalized_classifier = normalize_and_convert(test_class_0, scaler_for_classifier, 6)
test_class_1_normalized_classifier = normalize_and_convert(test_class_1, scaler_for_classifier, 6)


def to_dataframe(normalized_data, target_model):

  if target_model == "C2SDI":
    columns = ['X', 'Y', 'Z']
  elif target_model == "pullup classifier":
    columns = ['X', 'Y', 'Z', 'vx', 'vy', 'vz']
  else:
    ValueError("Enter either C2SDI or classifier")

  return [pd.DataFrame(data, columns=columns) for data in normalized_data]


# for C2SDI
train_data = to_dataframe(train_data_normalized, "C2SDI")
valid_data = to_dataframe(valid_class_0_normalized, "C2SDI") + to_dataframe(valid_class_1_normalized, "C2SDI")
test_data = to_dataframe(test_class_0_normalized, "C2SDI") + to_dataframe(test_class_1_normalized, "C2SDI")

# for pullup classifier
train_data_classifier = to_dataframe(train_data_normalized_classifier, "pullup classifier")
valid_data_classifier = to_dataframe(valid_class_0_normalized_classifier, "pullup classifier") + to_dataframe(valid_class_1_normalized_classifier, "pullup classifier")
test_data_classifier = to_dataframe(test_class_0_normalized_classifier, "pullup classifier") + to_dataframe(test_class_1_normalized_classifier, "pullup classifier")

logger.info(f"Number of Train / Val / Test dataset: {len(train_data)} / {len(valid_data)} / {len(test_data)}")



#-------------------- Preparing dataset (dictionary) --------------------#

# pull-up label (training for both C2SDI & pullup classifier)
training_label = [0 for i in range(len(train_class_0))] + [1 for i in range(len(train_class_1))]
valid_label = [0 for i in range(len(valid_class_0))] + [1 for i in range(len(valid_class_1))] 
test_label = [0 for i in range(len(test_class_0))] + [1 for i in range(len(test_class_1))] 

# training
X_list = [df for df in train_data]
X_array = np.stack([df.values for df in X_list], axis=0)
X_classifier_list = [df for df in train_data_classifier]
X_classifier_array = np.stack([df.values for df in X_classifier_list], axis=0)

training_df = {
  'X': X_array,
  'X_classifier': X_classifier_array,
  'class_label': training_label,
}

# validation
X_list_v = [df for df in valid_data]
X_array_v = np.stack([df.values for df in X_list_v], axis=0)
X_list_v_classifier = [df for df in valid_data_classifier]
X_array_v_classifier = np.stack([df.values for df in X_list_v_classifier], axis=0)

validatn_df = {
  'X': X_array_v,
  'X_classifier': X_array_v_classifier,
  'class_label': valid_label,
}

# test
X_list_t = [df for df in test_data]
X_array_t = np.stack([df.values for df in X_list_t], axis=0)
X_list_t_classifier = [df for df in test_data_classifier]
X_array_t_classifier = np.stack([df.values for df in X_list_t_classifier], axis=0)

testingg_df = {
  'X': X_array_t,
  'X_classifier': X_array_t_classifier,
  'class_label': test_label,
}



#-------------------- Masking initial parts (for C2SDI) --------------------#

def masking_condition(data, rate=1.0, axis=1):
  data = data.copy()
  n = np.sum(~np.isnan(data), axis=1)[:,0] # the number of non-NaN values per row

  initial_exclude_rate = rate # making by rate
  initial_exclude_count = np.asarray(n * initial_exclude_rate).astype(int) 
  start_idx = list(initial_exclude_count)

  if axis == 1:
      indices = np.arange(data.shape[1])
      data[:, indices >= start_idx[0], :] = np.nan 
  elif axis == 0:
      data[start_idx:, :, :] = np.nan 
  else:
      raise ValueError("Axis should be 0 or 1.")
      
  return data



#-------------------- Building dataset whether using impact point --------------------#

def data_with_impact_point(original_data, masked_data, indices):
    restored_data = masked_data.copy()
    for i in range(len(restored_data)):
        restored_data[i, indices[i], :] = original_data[i, indices[i], :]
    return restored_data


if use_impact_point:
  logger.info("Impact Point is utilized.")

  missile_data = {
    "n_features": training_df['X'].shape[-1],

    # for training C2SDI
    "train_X": np.concatenate([data_with_impact_point(X_array, masking_condition(training_df['X'], 0.2), train_data_last_indices),
                              data_with_impact_point(X_array, masking_condition(training_df['X'], 0.4), train_data_last_indices),
                              data_with_impact_point(X_array, masking_condition(training_df['X'], 0.6), train_data_last_indices),
                              data_with_impact_point(X_array, masking_condition(training_df['X'], 0.8), train_data_last_indices)]),
    "train_X_ori": np.concatenate([training_df['X']] * 4),

    # for validating C2SDI
    "val_X": np.concatenate([data_with_impact_point(X_array_v, masking_condition(validatn_df['X'], 0.2), valid_data_last_indices),
                            data_with_impact_point(X_array_v, masking_condition(validatn_df['X'], 0.4), valid_data_last_indices),
                            data_with_impact_point(X_array_v, masking_condition(validatn_df['X'], 0.6), valid_data_last_indices),
                            data_with_impact_point(X_array_v, masking_condition(validatn_df['X'], 0.8), valid_data_last_indices)]),
    "val_X_ori": np.concatenate([validatn_df['X']] * 4),
    
    # for testing C2SDI
    "test_X": data_with_impact_point(X_array_t, masking_condition(testingg_df['X'], test_rate), test_data_last_indices),
    "test_X_ori": testingg_df['X'],

    # for pullup classifier
    "train_X_classifier": training_df['X_classifier'],
    "val_X_classifier": validatn_df['X_classifier'], 
    "test_X_classifier": testingg_df['X_classifier'],

    # for both C2SDI & pullup classifier
    "train_class_label": np.concatenate([training_df['class_label']] * 4),
    "val_class_label": np.concatenate([validatn_df['class_label']] * 4), 
    "train_class_label_classifier": training_df['class_label'],
    "val_class_label_classifier": validatn_df['class_label'],
    "test_class_label": testingg_df['class_label'],
    # "test_X_indicating_mask" will be made later

  }
  
else:
  
  logger.info("Impact Point is not utilized.")  
  missile_data = {

    "n_features": training_df['X'].shape[-1],

    # for training C2SDI
    "train_X": np.concatenate([masking_condition(training_df['X'], 0.2),
                              masking_condition(training_df['X'], 0.4),
                              masking_condition(training_df['X'], 0.6),
                              masking_condition(training_df['X'], 0.8)]),
    "train_X_ori": np.concatenate([training_df['X']] * 4),

    # for validating C2SDI
    "val_X": np.concatenate([masking_condition(validatn_df['X'], 0.2),
                            masking_condition(validatn_df['X'], 0.4),
                            masking_condition(validatn_df['X'], 0.6),
                            masking_condition(validatn_df['X'], 0.8)]),
    "val_X_ori": np.concatenate([validatn_df['X']] * 4),
    
    # for testing C2SDI
    "test_X": masking_condition(testingg_df['X'], test_rate),
    "test_X_ori": testingg_df['X'],

    # for pullup classifier
    "train_X_classifier": training_df['X_classifier'],
    "val_X_classifier": validatn_df['X_classifier'], 
    "test_X_classifier": testingg_df['X_classifier'],

    # for both C2SDI & pullup classifier
    "train_class_label": np.concatenate([training_df['class_label']] * 4),
    "val_class_label": np.concatenate([validatn_df['class_label']] * 4), 
    "train_class_label_classifier": training_df['class_label'],
    "val_class_label_classifier": validatn_df['class_label'],
    "test_class_label": testingg_df['class_label'],
    # "test_X_indicating_mask": None # will be made later

  }
  


# for imputation target : mask = 1. if not, mask = 0.
# imputation target : test_X_ori is not NaN & test_X is NaN
missile_data["train_X_indicating_mask"] = ~np.isnan(missile_data["train_X_ori"]) & np.isnan(missile_data["train_X"])
missile_data["val_X_indicating_mask"] = ~np.isnan(missile_data["val_X_ori"]) & np.isnan(missile_data["val_X"])
missile_data["test_X_indicating_mask"] = ~np.isnan(missile_data["test_X_ori"]) & np.isnan(missile_data["test_X"])
  
dataset_for_training = {
    "X": missile_data['train_X'],
    "X_ori": missile_data['train_X_ori'],
    "indicating_mask": missile_data["train_X_indicating_mask"],
    "class_label": missile_data['train_class_label']
}

dataset_for_validating = {
    "X": missile_data['val_X'],
    "X_ori": missile_data['val_X_ori'],
    "indicating_mask": missile_data["val_X_indicating_mask"],
    "class_label": missile_data['val_class_label']
}

dataset_for_testing = {
    "X": missile_data['test_X'],
    "class_label": missile_data['test_class_label']
}


#-------------------- Start Training --------------------#
from modules import train

csdi = train(dataset_for_training, dataset_for_validating, n_features=missile_data['n_features'], 
              saving_path=results_path, model_epochs=model_epochs, batch_size=batch_size, 
              patience=patience, inference_mode=csdi_inference_mode, existed_model_path=csdi_model_path)


#-------------------- Predict Binary Class Label --------------------#
from modules import imputation
from modules import classification
import copy

predicted_label = classification(missile_data=missile_data, saving_path=results_path, num_epochs=classifier_epochs, 
                                  inference_mode=classifier_inference_mode, existed_model_path=classifier_model_path)


pred_test = copy.deepcopy(dataset_for_testing)
pred_test['class_label'] = predicted_label

pr_res = imputation(pred_test, orig_data=missile_data, real_test=real_test, csdi=csdi, saving_path=results_path, 
                    scaler=scaler_for_C2SDI, predicted=True, n_sampling_times=n_sampling_times, batch_size=batch_size, viss=True)

print(pr_res)