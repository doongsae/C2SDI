############################################################
#                      augmentation.py                     #
#        Make augmentation by polynomial regression        #
############################################################

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
import numpy as np

def fit_polynomial(x, y, degree):
  model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
  model.fit(x.reshape(-1, 1), y)
  return model

def augment_polyreg(data, num_augmentations):
  # Assuming 'data' is a pandas DataFrame with columns 'x', 'y', 'z', and 'time'
  
  # Normalize time to range [0, 1]
  t = np.linspace(0, 1, len(data))
  #t = data['time'].values
  
  # Fit n-degree polynomials to x, y, and z
  model_x = fit_polynomial(t, data['X'], 8)
  model_y = fit_polynomial(t, data['Y'], 8)
  model_z = fit_polynomial(t, data['Z'], 8)
    
  augmented_datasets = []
  
  for i in range(num_augmentations):
    # Create augmented dataset
    augmented_data = data.copy()
    
    # Generate small random adjustments for each coefficient
    adjust_x = np.random.uniform(-1, 1, 8+1)
    adjust_y = np.random.uniform(-1, 1, 8+1)
    adjust_z = np.random.uniform(-1, 1, 8+1)
    
    # Apply the adjusted polynomials to modify x, y, z coordinates
    x_pred = model_x.predict(t.reshape(-1, 1)).flatten()
    y_pred = model_y.predict(t.reshape(-1, 1)).flatten()
    z_pred = model_z.predict(t.reshape(-1, 1)).flatten()
    
    # Apply adjustments using numpy's polynomial capabilities
    augmented_data['X'] = x_pred + np.polyval(adjust_x[::-1], t)
    augmented_data['Y'] = y_pred + np.polyval(adjust_y[::-1], t)
    augmented_data['Z'] = z_pred + np.polyval(adjust_z[::-1], t)
    
    augmented_datasets.append(augmented_data)
    
  return augmented_datasets


def augmentation(base_data):
  augmented_pullup = []

  for datum in base_data:
    augmented_pullup.append(datum)
    augmented_pullup.extend(augment_polyreg(datum, num_augmentations=3))

  return augmented_pullup