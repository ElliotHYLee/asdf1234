from astropy.time import Time, TimeDelta
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from astropy import units as u
import numpy as np
import matplotlib.pyplot  as plt

# Load measurement data from file
npzfile = np.load('GPS_meas.npz', allow_pickle=True)
noisy_measurement = npzfile['measurements']
t_measurement = npzfile['t_measurements']
# Convert measurement times to Time objects
t_measurement_time = Time(t_measurement)

N = len(t_measurement) # Number of measurements.

# Calculate time intervals (dt) between consecutive measurements
dt_list = (t_measurement_time[1:] - t_measurement_time[:-1]) # TimeDelta object. Shape: (720,)


# gps measurement data
meas_pos = noisy_measurement[:, 0:3] # measured position. Shape: (N, 3)
meas_vel = noisy_measurement[:, 3:6] # measured velocity. Shape: (N, 3)

# Initial state vector
position = [3235.64171524, 2693.72565982, -5335.42793567] 
velocity = [-4.87430005, 5.89879341, 0.01977648] 
t_astropy = Time(t_measurement[0]) # Time object
x = np.hstack((position, velocity)) # initial state vector. Shape: (6,)
print(x.shape)
print(x)


states = [x]
for i in range(1, N):
    orbit = Orbit.from_vectors(Earth, x[:3]<< u.km, x[3:]<< u.km / u.s)
    dt = dt_list[i-1].sec
    orbit = orbit.propagate(TimeDelta(dt * u.s))
    x = np.hstack((orbit.r.to(u.km).value, orbit.v.to(u.km / u.s).value))
    print(x)
    states.append(x)


# Plotting the results (3,2) plots, first 3 colrums pos x, y, z, last 3 colrums vel x, y, z
states = np.array(states)   
pos = states[:, :3]
vel = states[:, 3:]

fig, axs = plt.subplots(3, 2, figsize=(12, 12))
for i in range(3):
    axs[i, 0].plot(pos[:, i], 'b.-', alpha=1, label='Propagated')
    axs[i, 0].plot(meas_pos[:, i], 'g.-',alpha=0.5, label='Mesurement')
    axs[i, 0].set_ylabel(f'Position {["x", "y", "z"][i]} (km)')
    axs[i, 0].legend()
    axs[i, 1].plot(vel[:, i], 'b', alpha=1, label='Propagated')
    axs[i, 1].plot(meas_vel[:, i], 'g--',alpha=0.5, label='Mesurement')
    axs[i, 1].set_ylabel(f'Velocity {["x", "y", "z"][i]} (km/s)')
    axs[i, 1].legend()

plt.tight_layout()
plt.show()
