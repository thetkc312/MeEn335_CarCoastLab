"""
Written by T.K.C. 10/13/2026
"""

import numpy as np
import scipy

RAW_DATA_FILENAME = 'data/LexusData.mat'

# Load the .mat file
data = scipy.io.loadmat(RAW_DATA_FILENAME)
angle = data['theta_deg'].flatten()[0]
#unit_note = data['units_note']
# time in seconds, v1 in m/s

# --- Formatting useful dictionaries of data
coast_north = {}
coast_north['t'] = data['t1'].flatten()
coast_north['v'] = data['v1'].flatten()
coast_north['θ'] = angle.flatten()

coast_south = {}
coast_south['t'] = data['t2'].flatten()
coast_south['v'] = data['v2'].flatten()
coast_south['θ'] = -angle.flatten()

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


#------------------------------------#
# Part 1

# --- Setting constants
# Y = [v]
dYdt = lambda t, Y, m, b, f_t: (f_t(t) - b*Y[0]) / m
mass = 1760 + 150 #kg
f_gravity = lambda m, g, θ: m * g * np.sin(θ)
gravity = 9.81 #m/s^2

# TEMPORARY
b_test = 1


# --- Using initial conditions from real data to generate predicted data
north_times = coast_north['t']
f_t_g_north = lambda t: f_gravity(mass, gravity, coast_north['θ'])
north_pred = scipy.integrate.solve_ivp(fun=dYdt, t_span=(north_times[0], north_times[-1]), y0=[coast_north['v'][0]], t_eval=north_times, args=(mass, b_test, f_t_g_north))
print(north_pred)

south_times = coast_south['t']
f_t_g_south = lambda t: f_gravity(mass, gravity, coast_south['θ'])
south_pred = scipy.integrate.solve_ivp(fun=dYdt, t_span=(south_times[0], south_times[-1]), y0=[coast_south['v'][0]], t_eval=south_times, args=(mass, b_test, f_t_g_south))
print(south_pred)


# --- Comparing prediction to reality and quantifying error (cumulatively for north and south)

