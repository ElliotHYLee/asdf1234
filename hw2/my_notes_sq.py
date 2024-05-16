import numpy as np
import matplotlib.pyplot as plt

# Define the Square-Root Unscented Kalman Filter
class SquareRootUnscentedKalmanFilter:
    def __init__(self, dim_x, dim_z, fx, hx, Q, R, alpha=1e-3, beta=2, kappa=0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.fx = fx
        self.hx = hx
        self.Q = Q  # process noise
        self.R = R  # measurement noise
        
        self.x = np.zeros(dim_x)
        self.S = np.linalg.cholesky(np.eye(dim_x))  # Cholesky factor of state covariance
        
        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambd = self.alpha**2 * (dim_x + self.kappa) - dim_x
        
        self.gamma = np.sqrt(dim_x + self.lambd)
        
        # Weights for mean and covariance
        self.Wm = np.full(2 * dim_x + 1, 1 / (2 * (dim_x + self.lambd)))
        self.Wc = np.full(2 * dim_x + 1, 1 / (2 * (dim_x + self.lambd)))
        self.Wm[0] = self.lambd / (dim_x + self.lambd)
        self.Wc[0] = self.lambd / (dim_x + self.lambd) + (1 - self.alpha**2 + self.beta)
        
    def sigma_points(self):
        Psqrt = self.S
        sigma_points = np.zeros((2 * self.dim_x + 1, self.dim_x))
        sigma_points[0] = self.x
        for i in range(self.dim_x):
            sigma_points[i + 1] = self.x + self.gamma * Psqrt[:, i]
            sigma_points[self.dim_x + i + 1] = self.x - self.gamma * Psqrt[:, i]
        return sigma_points

    def predict(self, dt):
        sigma_points = self.sigma_points()
        x_pred = np.zeros(self.dim_x)
        
        for i in range(2 * self.dim_x + 1):
            sigma_points[i] = self.fx(sigma_points[i], dt)
            x_pred += self.Wm[i] * sigma_points[i]
        
        # Compute predicted state covariance square-root
        X = (sigma_points - x_pred).T
        S_pred = np.linalg.qr(np.hstack((np.sqrt(self.Wc[1]) * X[:, 1:], self.Q)))[0]
        
        self.x = x_pred
        self.S = S_pred
    
    def update(self, z):
        sigma_points = self.sigma_points()
        Z = np.zeros((2 * self.dim_x + 1, self.dim_z))
        z_pred = np.zeros(self.dim_z)
        
        for i in range(2 * self.dim_x + 1):
            Z[i] = self.hx(sigma_points[i])
            z_pred += self.Wm[i] * Z[i]
        
        # Compute measurement prediction covariance square-root
        Y = (Z - z_pred).T
        S_z = np.linalg.qr(np.hstack((np.sqrt(self.Wc[1]) * Y[:, 1:], self.R)))[0]
        
        # Cross-covariance
        X = (sigma_points - self.x).T
        P_xz = np.dot(np.sqrt(self.Wc[1]) * X[:, 1:], Y[:, 1:].T)
        
        K = np.linalg.solve(S_z.T, np.linalg.solve(S_z, P_xz.T)).T
        
        self.x += np.dot(K, (z - z_pred))
        self.S = np.linalg.cholesky(np.dot(self.S, self.S.T) - np.dot(K, np.dot(S_z, S_z.T)).dot(K.T))
        
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
    Q = np.eye(6) * 10**-2  # Process noise
    R = np.eye(6) * 10**-5  # Measurement noise

    # Initialize UKF with example parameters
    ukf = SquareRootUnscentedKalmanFilter(dim_x=6, dim_z=6, fx=fx, hx=hx, Q=Q, R=R, alpha=1e-3, beta=2, kappa=0)

    # Generate true trajectory and measurements
    true_states = []
    measurements = []

    for t in np.arange(0, total_time, dt):
        true_state = np.array([np.sin(t), np.sin(t), np.sin(t), np.cos(t), np.cos(t), np.cos(t)])
        measurement = true_state + np.random.normal(0, 0.1, 6)
        true_states.append(true_state)
        measurements.append(measurement)

    true_states = np.array(true_states)
    measurements = np.array(measurements)

    # UKF estimation
    ukf_estimates = []

    for z in measurements:
        ukf.predict(dt)
        ukf.update(z)
        ukf_estimates.append(ukf.x.copy())

    ukf_estimates = np.array(ukf_estimates)

    # Plot results with measurements included
    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    time = np.arange(0, total_time, dt)

    axs[0, 0].plot(time, true_states[:, 0], '.-', label='True X', color='red', linestyle='dotted')
    axs[0, 0].plot(time, measurements[:, 0], '.-', label='Measured X', color='green', linestyle='dashed')
    axs[0, 0].plot(time, ukf_estimates[:, 0], '.-', alpha=0.5, label='Estimated X', color='magenta')
    axs[0, 0].set_title('Position X')
    axs[0, 0].legend()

    axs[1, 0].plot(time, true_states[:, 1], '.-', label='True Y', color='red', linestyle='dotted')
    axs[1, 0].plot(time, measurements[:, 1], '.-', label='Measured Y', color='green', linestyle='dashed')
    axs[1, 0].plot(time, ukf_estimates[:, 1], '.-', alpha=0.5, label='Estimated Y', color='magenta')
    axs[1, 0].set_title('Position Y')
    axs[1, 0].legend()

    axs[2, 0].plot(time, true_states[:, 2], '.-', label='True Z', color='red', linestyle='dotted')
    axs[2, 0].plot(time, measurements[:, 2], '.-', label='Measured Z', color='green', linestyle='dashed')
    axs[2, 0].plot(time, ukf_estimates[:, 2], '.-', alpha=0.5, label='Estimated Z', color='magenta')
    axs[2, 0].set_title('Position Z')
    axs[2, 0].legend()

    axs[0, 1].plot(time, true_states[:, 3], '.-', label='True VX', color='red', linestyle='dotted')
    axs[0, 1].plot(time, measurements[:, 3], '.-', label='Measured VX', color='green', linestyle='dashed')
    axs[0, 1].plot(time, ukf_estimates[:, 3], '.-', alpha=0.5, label='Estimated VX', color='magenta')
    axs[0, 1].set_title('Velocity X')
    axs[0, 1].legend()

    axs[1, 1].plot(time, true_states[:, 4], '.-', label='True VY', color='red', linestyle='dotted')
    axs[1, 1].plot(time, measurements[:, 4], '.-', label='Measured VY', color='green', linestyle='dashed')
    axs[1, 1].plot(time, ukf_estimates[:, 4], '.-', alpha=0.5, label='Estimated VY', color='magenta')
    axs[1, 1].set_title('Velocity Y')
    axs[1, 1].legend()

    axs[2, 1].plot(time, true_states[:, 5], '.-',label='True VZ', color='red', linestyle='dotted')
    axs[2, 1].plot(time, measurements[:, 5], '.-',label='Measured VZ', color='green', linestyle='dashed')
    axs[2, 1].plot(time, ukf_estimates[:, 5],'.-', alpha=0.5, label='Estimated VZ', color='magenta')
    axs[2, 1].set_title('Velocity Z')
    axs[2, 1].legend()

    plt.tight_layout()
    plt.show()