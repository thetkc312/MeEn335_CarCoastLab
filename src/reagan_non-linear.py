import numpy as np

# POWER
Cd = 0.6017
mu = 0.0001372

theta = 0
power_55 = (-1.1397*Cd*(55/2.237)**2 - 18737.1*(np.cos(theta)*mu - np.sin(theta)))*(55/2.237)
power_100 = (-1.1397*Cd*(100/2.237)**2 - 18737.1*(np.cos(theta)*mu - np.sin(theta)))*(100/2.237)

print(power_55 / 745.7)
print(power_100 / 745.7) # converts W to hp
