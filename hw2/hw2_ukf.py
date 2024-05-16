from astropy.time import Time, TimeDelta
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from astropy import units as u
import numpy as np
import matplotlib.pyplot  as plt
from utils import UnscentedKalmanFilter
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

# Initial state vector
position = [3235.64171524, 2693.72565982, -5335.42793567] 
velocity = [-4.87430005, 5.89879341, 0.01977648] 
t_astropy = Time(t_measurement[0]) # Time object
x = np.hstack((position, velocity)) # initial state vector. Shape: (6,)
# print(x.shape)
# print(x)

# fx
def fx(x, dt):
    # print("fx")
    # print("x", x)
    try:
        orbit = Orbit.from_vectors(Earth, x[:3]<< u.km, x[3:]<< u.km / u.s)
        orbit = orbit.propagate(TimeDelta(dt.sec*u.s)) # seconds!
    except Exception as e:
        print(e)
        print('x:', x)
        print('dt:', dt)
        return x

    new_pos, new_vel = orbit.r.to(u.km).value, orbit.v.to(u.km / u.s).value
    return np.hstack((new_pos, new_vel)) 

Q = np.eye(6)*10**-1 # Process noise covariance. Shape: (6, 6)
R = np.eye(6)*10**-1 # Measurement noise covariance. Shape: (6, 6)
ukf = UnscentedKalmanFilter(dim_x=6, dim_z=6)
ukf.set_fx(fx)

N=200
states = [x]
for i in range(1, N):
    print('Iteration:', i)
    sigma_points = ukf.sigma_points(x, Q)   
    print("sigma potins")
    # print(sigma_points)

    x_pred, P_pred = ukf.predict(sigma_points, dt_list[i-1])
    print(x_pred.shape)
    print(x_pred)


    x = x_pred
    Q = P_pred
    states.append(x)




# Plotting the results (3,2) plots, first 3 colrums pos x, y, z, last 3 colrums vel x, y, z
states = np.array(states)   
pos = states[:, :3]
vel = states[:, 3:]

fig, axs = plt.subplots(3, 2, figsize=(12, 12))
for i in range(3):
    axs[i, 0].plot(pos[:, i], 'b.-', alpha=0.5, label='Propagated')
    axs[i, 0].plot(meas_pos[:, i], 'g.-', label='Mesurement')
    axs[i, 0].set_ylabel(f'Position {["x", "y", "z"][i]} (km)')
    axs[i, 0].legend()
    axs[i, 1].plot(vel[:, i], 'b', alpha=0.5, label='Propagated')
    axs[i, 1].plot(meas_vel[:, i], 'g--', label='Mesurement')
    axs[i, 1].set_ylabel(f'Velocity {["x", "y", "z"][i]} (km/s)')
    axs[i, 1].legend()

plt.tight_layout()
plt.show()
