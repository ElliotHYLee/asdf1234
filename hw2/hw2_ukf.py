from astropy.time import Time, TimeDelta
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from astropy import units as u
import numpy as np
import matplotlib.pyplot  as plt
from my_notes import UnscentedKalmanFilter
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True, precision=6)

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
# add more gaussian noise to the measurement for fun
# meas_pos += np.random.normal(0, 10**3, meas_pos.shape)
# meas_vel += np.random.normal(0, 10**0, meas_vel.shape)

# Initial state vector
position = [3235.64171524, 2693.72565982, -5335.42793567] 
velocity = [-4.87430005, 5.89879341, 0.01977648] 
x = np.hstack((position, velocity)) # initial state vector. Shape: (6,)

# fx
def fx(x, dt):
    orbit = Orbit.from_vectors(Earth, x[:3]<< u.km, x[3:]<< u.km / u.s)
    orbit = orbit.propagate(TimeDelta(dt.sec*u.s)) # seconds!
    new_pos, new_vel = orbit.r.to(u.km).value, orbit.v.to(u.km / u.s).value
    return np.hstack((new_pos, new_vel))

def hx(x):
    return x

Q = np.eye(6)*10**-5 # Process noise covariance. Shape: (6, 6)
R = np.eye(6)*10**-1 # Measurement noise covariance. Shape: (6, 6)
ukf = UnscentedKalmanFilter(dim_x=6, dim_z=6, fx=fx, hx=hx, Q=Q, R=R, alpha=10**-3, beta=2, kappa=0)
ukf.x = x

N = 100
states = [x]

for i in range(1, N):
    print('Iteration:', i)
    ukf.predict(dt_list[i-1])
    ukf.update(noisy_measurement[i])
    x = ukf.x
    P = ukf.P
    states.append(x)

# Plotting the results (3,2) plots, first 3 colrums pos x, y, z, last 3 colrums vel x, y, z
states = np.array(states)   
pos = states[:, :3]
vel = states[:, 3:]

fig, axs = plt.subplots(3, 2, figsize=(12, 12))
subdata_idx = N
for i in range(3):
    axs[i, 0].plot(pos[:subdata_idx, i], 'b.-', alpha=1, label='UKF')
    axs[i, 0].plot(meas_pos[:subdata_idx, i], 'g--',alpha=0.9, label='Mesurement')
    axs[i, 0].set_ylabel(f'Position {["x", "y", "z"][i]} (km)')
    axs[i, 0].legend()
    axs[i, 1].plot(vel[:subdata_idx, i], 'b.-', alpha=1, label='UKF')
    axs[i, 1].plot(meas_vel[:subdata_idx, i], 'g--', alpha=0.9, label='Mesurement')
    axs[i, 1].set_ylabel(f'Velocity {["x", "y", "z"][i]} (km/s)')
    axs[i, 1].legend()

plt.tight_layout()
plt.show()
