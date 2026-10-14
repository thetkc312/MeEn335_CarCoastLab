"""
Written by T.K.C. 10/13/2026
"""

import numpy as np
import scipy.io

RAW_DATA_FILENAME = 'data/LexusData.mat'

# Load the .mat file
data = scipy.io.loadmat(RAW_DATA_FILENAME)
angle = data['theta_deg'].flatten()[0]
#unit_note = data['units_note']
# time in seconds, v1 in m/s

coast_north = {}
coast_north['t'] = data['t1']
coast_north['v'] = data['v1']
coast_north['θ'] = angle

coast_south = {}
coast_south['t'] = data['t2']
coast_south['v'] = data['v2']
coast_south['θ'] = -angle

print(coast_north)
print(coast_south)
