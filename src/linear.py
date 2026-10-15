"""
Written by T.K.C. 10/13/2026
"""

import matplotlib.pyplot as plt
import numpy as np
import scipy

RAW_DATA_FILENAME = 'data/LexusData.mat'

# Load the .mat file
data = scipy.io.loadmat(RAW_DATA_FILENAME)
angle = np.deg2rad(data['theta_deg'].item())
#unit_note = data['units_note']
# time in seconds, v1 in m/s

# --- Formatting useful dictionaries of data
coast_north = {}
coast_north['t'] = data['t1'].flatten()
coast_north['v'] = data['v1'].flatten()
coast_north['θ'] = angle

coast_south = {}
coast_south['t'] = data['t2'].flatten()
coast_south['v'] = data['v2'].flatten()
coast_south['θ'] = -angle

# --- Sorting and removing duplicates based on time for time and velocity
north_order = np.argsort(coast_north['t'])
coast_north['t'] = coast_north['t'][north_order]
coast_north['v'] = coast_north['v'][north_order]
coast_north['t'], north_unique = np.unique(coast_north['t'], return_index=True)
coast_north['v'] = coast_north['v'][north_unique]

south_order = np.argsort(coast_south['t'])
coast_south['t'] = coast_south['t'][south_order]
coast_south['v'] = coast_south['v'][south_order]
coast_south['t'], south_unique = np.unique(coast_south['t'], return_index=True)
coast_south['v'] = coast_south['v'][south_unique]

# --- Reducing dataset density such that only pairs where the GPS data has updated are included
# This was implemented with the support of ChatGPT-5.6 Luna
if False: #Optional data-cleaning method prior to analysis
    north_updated = np.r_[True, np.diff(coast_north['v']) != 0]
    coast_north['t'] = coast_north['t'][north_updated]
    coast_north['v'] = coast_north['v'][north_updated]

    south_updated = np.r_[True, np.diff(coast_south['v']) != 0]
    coast_south['t'] = coast_south['t'][south_updated]
    coast_south['v'] = coast_south['v'][south_updated]

# --- Removing the first elements from each dataset
# The velocity seems inconsistent with the trend of the rest of the data.
# This could be a consequence of it being correlated with the first GPS
# position coordinate, where it would not have been possible to effectively
# determine velocity as a function of two position points.
if False: #Optional data-cleaning method prior to analysis
    coast_north['t'] = coast_north['t'][1:]
    coast_north['v'] = coast_north['v'][1:]
    coast_south['t'] = coast_south['t'][1:]
    coast_south['v'] = coast_south['v'][1:]


# --- Setting constants
# Y = [v]
dYdt = lambda t, Y, m, b, f_t: (f_t(t) - b*Y[0]) / m
mass = 1760 + 150 #kg
f_gravity = lambda m, g, θ: m * g * np.sin(θ)
gravity = 9.81 #m/s^2

f_t_g_north = lambda t: f_gravity(mass, gravity, coast_north['θ'])
f_t_g_south = lambda t: f_gravity(mass, gravity, coast_south['θ'])


# --- Comparing prediction to reality and quantifying error (cumulatively for north and south)
def predict(b, dataset, f_t):
    times = dataset['t']
    vel = dataset['v']
    return scipy.integrate.solve_ivp(fun=dYdt, t_span=(times[0], times[-1]), y0=[vel[0]], t_eval=times, args=(mass, b, f_t)).y[0]

def ivp_res_error(b):
    b = float(np.asarray(b).item())
    north_errors = predict(b, coast_north, f_t_g_north) - coast_north['v']
    south_errors = predict(b, coast_south, f_t_g_south) - coast_south['v']
    errors = np.concatenate((north_errors, south_errors))
    sse = np.sum(np.abs(errors)**2)
    return sse

opt_res = scipy.optimize.minimize(ivp_res_error, [20])
print(opt_res)
b_opt = float(opt_res.x.item())
print(b_opt)

# --- Plotting results with optimized value
plt.figure(1)
plt.plot(coast_north['t'], coast_north['v'], color='blue', label='North Data')
plt.plot(coast_south['t'], coast_south['v'], color='cyan', label='South Data')
plt.plot(coast_north['t'], predict(b_opt, coast_north, f_t_g_north), label=f'North Prediction (b={b_opt:.2f})', linestyle='--', color='red')
plt.plot(coast_south['t'], predict(b_opt, coast_south, f_t_g_south), label=f'South Prediction (b={b_opt:.2f})', linestyle='--', color='magenta')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.title('Speed vs. Time')

plt.show()