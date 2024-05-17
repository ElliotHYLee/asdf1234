from astropy.time import Time, TimeDelta
from poliastro.twobody import Orbit
from poliastro.bodies import Earth
from astropy import units as u
import numpy as np
import matplotlib.pyplot  as plt
from utils import SquareRootUKF
np.set_printoptions(linewidth=np.inf)
np.set_printoptions(suppress=True, precision=6)



npzfile = np.load('GPS_meas.npz', allow_pickle=True)

noisy_measurement = npzfile['measurements']
t_measurement = npzfile['t_measurements']
# Convert measurement times to Time objects
t_measurement_time = Time(t_measurement)

N = len(t_measurement) # Number of measurements.

# Calculate time intervals (dt) between consecutive measurements
dt_list = (t_measurement_time[1:] - t_measurement_time[:-1]) # TimeDelta object. Shape: (720,)


# Define the state transition function
def fx(x, dt):
    orbit = Orbit.from_vectors(Earth, x[:3] * u.km, x[3:] * u.km / u.s)
    orbit = orbit.propagate(TimeDelta(dt * u.s)) # seconds!
    new_pos, new_vel = orbit.r.to(u.km).value, orbit.v.to(u.km / u.s).value
    return np.hstack((new_pos, new_vel))

# Define the measurement function
def hx(x):
    return x

# Initial state vector
position = [3235.64171524, 2693.72565982, -5335.42793567] 
velocity = [-4.87430005, 5.89879341, 0.01977648] 
x = np.hstack((position, velocity)) # initial state vector. Shape: (6,)

# Process & measurement noise
P = np.eye(6)
Q = np.eye(6)*10**-6 # Process noise covariance. Shape: (6, 6)
R = np.eye(6)*10**4 # Measurement noise covariance. Shape: (6, 6)


sr_ukf = SquareRootUKF(x, P, Q, R)

# N = 200
states = []
for i in range (1, N):
    print(f"Iteration {i}")
    sr_ukf.predict(fx, dt_list[i-1].sec)
    # print("state after pred")
    # print(sr_ukf.x)
    # print(sr_ukf.sqrt_P)
    sr_ukf.correct(hx, noisy_measurement[i])
    # print("state after corr")
    # print(sr_ukf.x)
    # print(sr_ukf.sqrt_P)
    states.append([sr_ukf.x])

states = np.concatenate(states)
print(states.shape)

fig, axs = plt.subplots(3, 2, figsize=(12, 12))

axs[0, 0].plot(states[:, 0], '.-', label='X', color='red')
axs[0, 0].plot(noisy_measurement[:, 0], '.-', label='Measured X', color='green')
axs[0, 0].set_title('Position X')
axs[0, 0].legend()

axs[1, 0].plot(states[:, 1], '.-', label='Y', color='red')
axs[1, 0].plot(noisy_measurement[:, 1], '.-', label='Measured Y', color='green')
axs[1, 0].set_title('Position Y')
axs[1, 0].legend()

axs[2, 0].plot(states[:, 2], '.-', label='Z', color='red')
axs[2, 0].plot(noisy_measurement[:, 2], '.-', label='Measured Z', color='green')
axs[2, 0].set_title('Position Z')
axs[2, 0].legend()

axs[0, 1].plot(states[:, 3], '.-', label='VX', color='red')
axs[0, 1].plot(noisy_measurement[:, 3], '.-', label='Measured VX', color='green')
axs[0, 1].set_title('Velocity X')
axs[0, 1].legend()

axs[1, 1].plot(states[:, 4], '.-', label='VY', color='red')
axs[1, 1].plot(noisy_measurement[:, 4], '.-', label='Measured VY', color='green')
axs[1, 1].set_title('Velocity Y')
axs[1, 1].legend()

axs[2, 1].plot(states[:, 5], '.-', label='VZ', color='red')
axs[2, 1].plot(noisy_measurement[:, 5], '.-', label='Measured VZ', color='green')
axs[2, 1].set_title('Velocity Z')
axs[2, 1].legend()

plt.tight_layout()
plt.show()





















