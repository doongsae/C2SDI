############################################################
#                       find_pullup.py                     #
#    Find Ground Truth Pullup Label using local minimum    #
############################################################

import numpy as np

def find_pullup(total_data):
  pullup_index = []
  k = 5
  n_data = len(total_data)

  for i in range(n_data):
    x = total_data[i]['X']
    y = total_data[i]['Y']
    z = total_data[i]['Z']

    # Calculate difference: value after - value before. 
    # The point where the difference goes from positive to negative is the extreme value.
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)

    # Detect change in difference (positive -> negative -> positive)
    extrema_indices = np.where(((dy[:-k] < 0) & (dy[k:] > 0)) | ((dz[:-k] < 0) & (dz[k:] > 0)))[0] + 1
    pullup_index.append(extrema_indices)


  pullup_list = []
  for pts in pullup_index:
    if len(pts) == 0:
        pullup_list.append(0)
    else:
        pullup_list.append(1)

  return pullup_list