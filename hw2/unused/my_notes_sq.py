import numpy as np
import matplotlib.pyplot as plt
from utils import SRUnscentedKalmanFilter

# Define state transition function for constant velocity model
def fx(x, dt):
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    return F @ x

# Define measurement function
def hx(x):
    return x  # Return all state components (position and velocity)

if __name__ == '__main__':
    # Simulation parameters
    total_time = 10
    dt = 0.1
    timesteps = int(total_time / dt)
    Q = np.eye(6) * 10**-1  # Process noise
    R = np.eye(6) * 10**-0  # Measurement noise

    # Initialize SR-UKF with example parameters
    sr_ukf = SRUnscentedKalmanFilter(dim_x=6, dim_z=6, fx=fx, hx=hx, Q=Q, R=R, alpha=1e-3, beta=2, kappa=0)

    # Generate true trajectory and measurements
    true_states = []
    measurements = []

    # create the data
    for t in np.arange(0, total_time, dt):
        true_state = np.array([np.sin(t), np.sin(t), np.sin(t), np.cos(t), np.cos(t), np.cos(t)])
        measurement = true_state + np.random.normal(0, 0.1, 6)
        true_states.append(true_state)
        measurements.append(measurement)

    true_states = np.array(true_states)
    measurements = np.array(measurements)

    # SR-UKF estimation
    sr_ukf_estimates = []

    for z in measurements:
        sr_ukf.predict(dt)
        sr_ukf.update(z)
        sr_ukf_estimates.append(sr_ukf.x.copy())

    sr_ukf_estimates = np.array(sr_ukf_estimates)

    # Plot results with measurements included
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    time = np.arange(0, total_time, dt)

    axs[0, 0].plot(time, true_states[:, 0], '.-', label='True X', color='b', linestyle='dotted')
    axs[0, 0].plot(time, measurements[:, 0],'.-', label='Measured X', color='green', linestyle='dashed')
    axs[0, 0].plot(time, sr_ukf_estimates[:, 0], '.-',alpha=0.5,label='Estimated X', color='magenta')
    axs[0, 0].set_title('Position X')
    axs[0, 0].legend()

    axs[1, 0].plot(time, true_states[:, 1], '.-',label='True Y', color='b', linestyle='dotted')
    axs[1, 0].plot(time, measurements[:, 1], '.-',label='Measured Y', color='green', linestyle='dashed')
    axs[1, 0].plot(time, sr_ukf_estimates[:, 1],'.-', alpha=0.5,label='Estimated Y', color='magenta')
    axs[1, 0].set_title('Position Y')
    axs[1, 0].legend()

    axs[2, 0].plot(time, true_states[:, 2],'.-', label='True Z', color='b', linestyle='dotted')
    axs[2, 0].plot(time, measurements[:, 2], '.-',label='Measured Z', color='green', linestyle='dashed')
    axs[2, 0].plot(time, sr_ukf_estimates[:, 2],'.-', alpha=0.5,label='Estimated Z', color='magenta')
    axs[2, 0].set_title('Position Z')
    axs[2, 0].legend()

    axs[0, 1].plot(time, true_states[:, 3], '.-',label='True VX', color='b', linestyle='dotted')
    axs[0, 1].plot(time, measurements[:, 3], '.-',label='Measured VX', color='green', linestyle='dashed')
    axs[0, 1].plot(time, sr_ukf_estimates[:, 3], '.-',alpha=0.5,label='Estimated VX', color='magenta')
    axs[0, 1].set_title('Velocity X')
    axs[0, 1].legend()

    axs[1, 1].plot(time, true_states[:, 4],'.-', label='True VY', color='b', linestyle='dotted')
    axs[1, 1].plot(time, measurements[:, 4], '.-',label='Measured VY', color='green', linestyle='dashed')
    axs[1, 1].plot(time, sr_ukf_estimates[:, 4],'.-', alpha=0.5,label='Estimated VY', color='magenta')
    axs[1, 1].set_title('Velocity Y')
    axs[1, 1].legend()

    axs[2, 1].plot(time, true_states[:, 5], '.-',label='True VZ', color='b', linestyle='dotted')
    axs[2, 1].plot(time, measurements[:, 5], '.-',label='Measured VZ', color='green', linestyle='dashed')
    axs[2, 1].plot(time, sr_ukf_estimates[:, 5],'.-', alpha=0.5, label='Estimated VZ', color='magenta')
    axs[2, 1].set_title('Velocity Z')
    axs[2, 1].legend()

    plt.tight_layout()
    plt.show()
