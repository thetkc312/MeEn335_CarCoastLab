"""
Written by T.K.C. 10/13/2026
"""

import scipy.io
import numpy as np

RAW_DATA_FILENAME = 'data/LexusData.mat'

# Load the .mat file
data = scipy.io.loadmat(RAW_DATA_FILENAME)
print(data['theta_deg'])

header = list(data.keys())
skip = ['__header__', '__version__', '__globals__', 'units_note']
for item in skip:
    if item in header:
        header.remove(item)

data_array = np.zeros((max([len(data[key]) for key in header]), len(header)))

for i, key in enumerate(header):
    col_data = data[key].flatten()
    data_array[:len(col_data), i] = col_data

# Save the data to a CSV file
np.savetxt(CLEAN_DATA_FILENAME, data_array, delimiter=',', header=','.join(header), fmt='%s')