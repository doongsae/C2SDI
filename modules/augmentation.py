############################################################
#                      augmentation.py                     #
#        Make augmentation by polynomial regression        #
############################################################

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s]: %(message)s')
logger = logging.getLogger(__name__)

def fit_polynomial(t, values, degree):
    # Choose non-NaN values
    mask = ~np.isnan(values)
    t_valid = t[mask]
    values_valid = values[mask]
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(t_valid.reshape(-1, 1), values_valid)
    return model

def augment_polyreg(data, num_augmentations):
    seq_length, num_features = data.shape
    t = np.linspace(0, 1, seq_length)
    X = data[:, 0]
    Y = data[:, 1]
    Z = data[:, 2]
    augmented_datasets = []

    # Fitting
    model_x = fit_polynomial(t, X, 8)
    model_y = fit_polynomial(t, Y, 8)
    model_z = fit_polynomial(t, Z, 8)

    for _ in range(num_augmentations):
        augmented_data = data.copy()
        adjust_x = np.random.uniform(-1, 1, 8 + 1)
        adjust_y = np.random.uniform(-1, 1, 8 + 1)
        adjust_z = np.random.uniform(-1, 1, 8 + 1)

        # predict for non-NaN values
        mask_x = ~np.isnan(X)
        mask_y = ~np.isnan(Y)
        mask_z = ~np.isnan(Z)

        x_pred = np.full_like(X, np.nan)
        y_pred = np.full_like(Y, np.nan)
        z_pred = np.full_like(Z, np.nan)

        x_pred[mask_x] = model_x.predict(t[mask_x].reshape(-1, 1)).flatten()
        y_pred[mask_y] = model_y.predict(t[mask_y].reshape(-1, 1)).flatten()
        z_pred[mask_z] = model_z.predict(t[mask_z].reshape(-1, 1)).flatten()
        
        augmented_data[:, 0] = np.where(mask_x, x_pred + np.polyval(adjust_x[::-1], t), X)
        augmented_data[:, 1] = np.where(mask_y, y_pred + np.polyval(adjust_y[::-1], t), Y)
        augmented_data[:, 2] = np.where(mask_z, z_pred + np.polyval(adjust_z[::-1], t), Z)
        augmented_datasets.append(augmented_data)

    return np.array(augmented_datasets)

def augmentation(base_data, num_aug):
    augmented = []
    logger.info(f"Augmenting {num_aug} samples for each datum ...")
    for datum in base_data:
        augmented.append(datum)
        augmented.extend(augment_polyreg(datum, num_augmentations=num_aug))
    return np.array(augmented)