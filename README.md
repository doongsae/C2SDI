# C2SDI: Conditional Score-based Diffusion Models with Classifier-free Guidance

This github repository is implementation of C2SDI: Conditional Score-based Diffusion Models with Classifier-free Guidance, based on CSDI [[1]](#1).

The code in this GitHub is based on, and modified from, the CSDI code in the PyPOTS library [[2]](#2).


## Usage
### Compatibility

This repository has been tested with Python version 3.8.

```shell
git clone https://github.com/doongsae/C2SDI.git

pip install requirements.txt

python run.py [path] [data_path] (--other_optional_args)
```


## Arguments
* `path` (Required): The path where the GitHub repository has been cloned.

* `dataset_path` (Required): The path to the directory containing the dataset.

* `--result_save_path` (_Optional_): The directory path where the trained model / output images will be saved. (Default: `'results/'`)

* `seq_len` (_Optional_): The sequence length of the data. This specifies the number of time steps or elements in each input sequence of the dataset. (Default: 300)

* `seed` (_Optional_): Global seed value for running program. (Default: 42)


## Notifications
We wrote the code assuming __binary__ classification. So if you want to change this, you can modify `n_classes` in the CSDI initialize part of `train.py`.


## References
<a id="1">[1]</a> 
Tashiro, Yusuke, et al. "CSDI: Conditional score-based diffusion models for probabilistic time series imputation." Advances in Neural Information Processing Systems 34 (2021): 24804-24816.

<a id="2">[2]</a> 
Wenjie Du. PyPOTS: a Python toolbox for data mining on Partially-Observed Time Series. arXiv, abs/2305.18811, 2023.