# C2SDI: Conditional Score-based Diffusion Models with Classifier-free Guidance

This github repository is implementation of C2SDI: Conditional Score-based Diffusion Models with Classifier-free Guidance [[1]](#1), based on CSDI [[2]](#2).

The code in this GitHub is based on, and modified from, the CSDI code in the PyPOTS library [[3]](#3).


## Usage
### Compatibility

This repository has been tested with Python version 3.8.

```shell
git clone https://github.com/doongsae/C2SDI.git

pip install requirements.txt

python run.py [path] [data_path] (--other_optional_args)
```


## Arguments for whole pipeline
* `path` (Required): The path where the GitHub repository has been cloned.

* `dataset_path` (Required): The path to the directory containing the dataset.

* `--result_save_path` (_Optional_): Path for saving model and imputation results. (default=`'results/'`)

* `--seed` (_Optional_): Global seed value for running program. (Default: 42)

* `--csdi_model_path` (_Optional_): Path for existing C2SDI model. If there are not existiing models, leave it empty.

* `--classifier_model_path` (_Optional_): Path for existing classifier model. If there are not existiing models, leave it empty.

* `--csdi_scaler_path` (_Optional_): Path for existing csdi scaler (3 dimension - X, Y, Z). If there are not existing models, leave it empty.

* `--classifier_scaler_path` (_Optional_): Path for existing classifier scaler (6 dimension - X, Y, Z, vX, vY, vZ). If there are not existing models, leave it empty.

## Arguments of hyperparameters
* `--time_interval` (_Optional_): Sampling interval to shorten the sequence length. (default=1.0)

* `--adjusted` (_Optional_): Drainage to adjust maximum length` (default=1.1)

* `--model_epochs` (_Optional_): The number of epochs for training the C2SDI (default=80)

* `--batch_size` (_Optional_): The batch size for training and evaluating the C2SDI (default=64)

* `--patience` (_Optional_): The patience for the early-stopping mechanism. Given a positive integer, the training process will be stopped when the model does not perform better after that number of epochs. Leaving it default as None will disable the early-stopping.

* `--classifier_epochs` (_Optional_): The number of epochs for training pullup classifier. (default=50)

* `--use_augmentation` (_Optional_): Augmentation decision for non-pullup train data. If you don't want to use augmented data, leave this empty.

* `--num_aug` (_Optional_): The number of augmented samples for each datum in train data. (default=11)

* `--missing_rate` (_Optional_): Artificial missing rate for data (default=0.05)


## Argument for data split
* `--train_ratio` (_Optional_): Training ratio among all data. (default=0.7)

* `--val_ratio` (_Optional_): Test/validation data ratio. (default=0.5)

## Argument for testing
* `--n_sampling_times` (_Optional_): The number of sampling times for testing the model to sample from the diffusion process. (default=1)

* `--test_rate` (_Optional_): A rate of conditions for test. (default=0.2)

* `--use_impact_point` (_Optional_): Decision of using impact point (additive condition). If you don't want to utilze this, leave this empty.



## Notifications
We wrote the code assuming __binary__ classification. So if you want to change this, you can modify `n_classes` in the CSDI initialize part of `train.py`.


## References
<a id="1">[1]</a> 
Kang, J., Lee, S., Yeom, J., Song, K. (2024). C2SDI: Conditional Score-based Diffusion Models with 
Classifier-free Guidance. IJCAI 2024 Workshop on Large Knowledge-Enhanced Models.

<a id="2">[2]</a> 
Tashiro, Yusuke, et al. "CSDI: Conditional score-based diffusion models for probabilistic time series imputation." Advances in Neural Information Processing Systems 34 (2021): 24804-24816.

<a id="3">[3]</a> 
Wenjie Du. PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series. arXiv, abs/2305.18811, 2023.
