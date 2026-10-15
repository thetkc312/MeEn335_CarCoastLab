"""
Written by T.K.C. 10/14/2026, adapted from code written by R. R.
"""

import matplotlib.pyplot as plt
import matplotlib.colors
import matplotlib.ticker
import numpy as np
import scipy

RAW_DATA_FILENAME = 'data/LexusData.mat'

# Load the .mat file
data = scipy.io.loadmat(RAW_DATA_FILENAME)
angle = np.deg2rad(data['theta_deg'].item())
#unit_note = data['units_note']
# time in seconds, v1 in m/s

# --- Formatting useful dictionaries of data
coast_south = {}
coast_south['t'] = data['t1'].flatten()
coast_south['v'] = data['v1'].flatten()
coast_south['θ'] = -angle

coast_north = {}
coast_north['t'] = data['t2'].flatten()
coast_north['v'] = data['v2'].flatten()
coast_north['θ'] = angle

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
if True: #Optional data-cleaning method prior to analysis
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
if True: #Optional data-cleaning method prior to analysis
    coast_north['t'] = coast_north['t'][1:]
    coast_north['v'] = coast_north['v'][1:]
    coast_south['t'] = coast_south['t'][1:]
    coast_south['v'] = coast_south['v'][1:]


# --- Setting constants
# Y = [v]
# SOUTH
m = 1760 + 150 #kg
Ap = 2.15 #m^2
pf = 1.06 #kg/m^3
vw = 3.12928 #m/s
gravity = 9.81 #m/s^2
# Cd = ?
# k = ?

dYdt = lambda t, Y, Cd, vw, f_k, f_g: -0.5 * Cd * Ap * pf * (Y[0] + vw)**2 / m - f_k / m + f_g / m

f_gravity = lambda θ: -m * gravity * np.sin(θ)
f_g_north = f_gravity(coast_north['θ'])
f_g_south = f_gravity(coast_south['θ'])
f_kfriction = lambda k, θ: k * m * gravity * np.cos(θ)
vw_north = vw
vw_south = -vw

# --- Making a prediction based on given parameters
def predict(Cd, k, dataset, vw, f_g):
    times = dataset['t']
    vel = dataset['v']
    angle = dataset['θ']
    result = scipy.integrate.solve_ivp(
        fun=dYdt,
        t_span=(times[0], times[-1]),
        y0=[vel[0]],
        t_eval=times,
        args=(Cd, vw, f_kfriction(k, angle), f_g),
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result


# --- Defining the error function to be minimized
def ivp_res_error(arr_Cd_k):
    Cd = arr_Cd_k[0]
    k = arr_Cd_k[1]
    try:
        north_errors = predict(Cd, k, coast_north, vw_north, f_g_north).y[0] - coast_north['v']
        south_errors = predict(Cd, k, coast_south, vw_south, f_g_south).y[0] - coast_south['v']
    except RuntimeError:
        return np.inf
    errors = np.concatenate((north_errors, south_errors))
    mse = np.mean(np.abs(errors)**2)
    return mse


# --- Using scipy.optimize.shgo to find the optimal values of Cd and k that minimize the error
# This method is better than using scipy.optimize.minimize because it can handle multiple local minima, which I suspected might be possible.
opt_res = scipy.optimize.shgo(ivp_res_error, [(0.0, 1.0), (0.0, 1.0)])
print(opt_res)

# I was able to verify that the minimize function was able to find the same minimum as shgo, so its results are consistent
#opt_res_min = scipy.optimize.minimize(ivp_res_error, [0.27, 0.017])
#print(opt_res_min)


# --- Plotting results with optimized value
# Only a single local minimum was found, so we can directly use the single result from shgo to plot the results.
plt.figure(1)
plt.plot(coast_north['t'], coast_north['v'], color='blue', label='North Data')
plt.plot(coast_south['t'], coast_south['v'], color='cyan', label='South Data')
plt.plot(coast_north['t'], predict(opt_res.x[0], opt_res.x[1], coast_north, vw_north, f_g_north).y[0], label=f'North Prediction (Cd={opt_res.x[0]:.2f}, k={opt_res.x[1]:.2e})', linestyle='--', color='magenta')
plt.plot(coast_south['t'], predict(opt_res.x[0], opt_res.x[1], coast_south, vw_south, f_g_south).y[0], label=f'South Prediction (Cd={opt_res.x[0]:.2f}, k={opt_res.x[1]:.2e})', linestyle='--', color='red')
plt.grid(True)
plt.xlabel('Time (s)')
plt.ylabel('Speed (m/s)')
plt.legend()
plt.title('Speed vs. Time')

plt.show()


# --- Plotting MSE as a function of Cd and k to visualize the error landscape
# This was implemented with the support of ChatGPT-5.6 Luna
Cd_values = np.linspace(0.5*opt_res.x[0], 1.5*opt_res.x[0], 50)
k_values = np.logspace(np.log10(0.00001 * opt_res.x[1]), np.log10(100 * opt_res.x[1]), 50)
Cd_grid, k_grid = np.meshgrid(Cd_values, k_values)
mse_grid = np.zeros_like(Cd_grid)

for i in range(len(Cd_values)):
    for j in range(len(k_values)):
        mse_grid[j, i] = ivp_res_error([Cd_grid[j, i], k_grid[j, i]])

fig, ax = plt.subplots()
ax.contourf(Cd_grid, k_grid, mse_grid, locator=matplotlib.ticker.LogLocator())
ax.set_xlabel('Cd')
ax.set_ylabel('k')
ax.set_yscale('log')
ax.set_title('Error Landscape (Logarithmic Scale)')
plt.show()

fig, ax = plt.subplots()
ax.contourf(Cd_grid, k_grid, mse_grid, levels=50)
ax.set_xlabel('Cd')
ax.set_ylabel('k')
ax.set_yscale('log')
ax.set_title('Error Landscape (Linear Scale)')
plt.show()