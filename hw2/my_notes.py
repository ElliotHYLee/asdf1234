import numpy as np
import matplotlib.pyplot as plt

# Define state transition function for constant velocity model
def fx(x, dt):
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    return F @ x

# Define measurement function
def hx(x):
    return x  # Return all state components (position and velocity)

# Define the Unscented Kalman Filter
class UnscentedKalmanFilter:
    def __init__(self, dim_states, dim_meas, Q, R, alpha=10**-3, beta=2, kappa=0):
        self.dim_states = dim_states
        self.dim_meas = dim_meas
        self.Q = Q # Process noise covariance
        self.R = R # Measurement noise covariance
        
        self.x = np.zeros(dim_states)
        self.P = np.eye(dim_states)
        self.fx = None
        self.hx = None
        
        # UKF parameters
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        self.lambd = self.alpha**2 * (dim_states + self.kappa) - dim_states
        
        self.gamma = np.sqrt(dim_states + self.lambd)
        
        # Weights for mean and covariance
        self.Wm = np.full(2 * dim_states + 1, 1 / (2 * (dim_states + self.lambd)))
        self.Wc = np.full(2 * dim_states + 1, 1 / (2 * (dim_states + self.lambd)))
        self.Wm[0] = self.lambd / (dim_states + self.lambd)
        self.Wc[0] = self.lambd / (dim_states + self.lambd) + (1 - self.alpha**2 + self.beta)
    
    def set_fx(self, fx):
        self.fx = fx
        
    def sigma_points(self):
        Psqrt = np.linalg.cholesky(self.P)
        sigma_points = np.zeros((2 * self.dim_states + 1, self.dim_states))
        sigma_points[0] = self.x
        for i in range(self.dim_states):
            sigma_points[i + 1] = self.x + self.gamma * Psqrt[:, i]
            sigma_points[self.dim_states + i + 1] = self.x - self.gamma * Psqrt[:, i]
        return sigma_points

    def predict(self):
        sigma_points = self.sigma_points()
        x_pred = np.zeros(self.dim_states)
        P_pred = np.zeros((self.dim_states, self.dim_states))
        
        for i in range(2 * self.dim_states + 1):
            sigma_points[i] = self.fx(sigma_points[i], self.dt)
            x_pred += self.Wm[i] * sigma_points[i]
        
        for i in range(2 * self.dim_states + 1):
            y = sigma_points[i] - x_pred
            P_pred += self.Wc[i] * np.outer(y, y)
        P_pred += self.Q
        
        self.x = x_pred
        self.P = P_pred
    
    def update(self, z):
        sigma_points = self.sigma_points()
import numpy as np
import matplotlib.pyplot as plt

# Define state transition function for constant velocity model
def fx(x, dt):
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    return F @ x

# Define measurement function
def hx(x):
    return x[:3]

# Define the Unscented Kalman Filter
class UnscentedKalmanFilter:
    def __init__(self, dim_x, dim_z, dt, fx, hx, Q, R):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.fx = fx
        self.hx = hx
        self.Q = Q
        self.R = R
        
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        
        # UKF parameters
        self.alpha = 1e-3
        self.beta = 2
        self.kappa = 0
        self.lambd = self.alpha**2 * (dim_x + self.kappa) - dim_x
        
        self.gamma = np.sqrt(dim_x + self.lambd)
        
        # Weights for mean and covariance
        self.Wm = np.full(2 * dim_x + 1, 1 / (2 * (dim_x + self.lambd)))
        self.Wc = np.full(2 * dim_x + 1, 1 / (2 * (dim_x + self.lambd)))
        self.Wm[0] = self.lambd / (dim_x + self.lambd)
        self.Wc[0] = self.lambd / (dim_x + self.lambd) + (1 - self.alpha**2 + self.beta)
        
    def sigma_points(self):
        Psqrt = np.linalg.cholesky(self.P)
        sigma_points = np.zeros((2 * self.dim_x + 1, self.dim_x))
        sigma_points[0] = self.x
        for i in range(self.dim_x):
            sigma_points[i + 1] = self.x + self.gamma * Psqrt[:, i]
            sigma_points[self.dim_x + i + 1] = self.x - self.gamma * Psqrt[:, i]
        return sigma_points

    def predict(self):
        sigma_points = self.sigma_points()
        x_pred = np.zeros(self.dim_x)
        P_pred = np.zeros((self.dim_x, self.dim_x))
        
        for i in range(2 * self.dim_x + 1):
            sigma_points[i] = self.fx(sigma_points[i], self.dt)
            x_pred += self.Wm[i] * sigma_points[i]
        
        for i in range(2 * self.dim_x + 1):
            y = sigma_points[i] - x_pred
            P_pred += self.Wc[i] * np.outer(y, y)
        P_pred += self.Q
        
        self.x = x_pred
        self.P = P_pred
    
    def update(self, z):
        sigma_points = self.sigma_points()
        Z = np.zeros((2 * self.dim_x + 1, self.dim_z))
        z_pred = np.zeros(self.dim_z)
        Pz = np.zeros((self.dim_z, self.dim_z))
        Pxz = np.zeros((self.dim_x, self.dim_z))
        
        for i in range(2 * self.dim_x + 1):
            Z[i] = self.hx(sigma_points[i])
            z_pred += self.Wm[i] * Z[i]
import numpy as np
import matplotlib.pyplot as plt

# Define state transition function for constant velocity model
def fx(x, dt):
    F = np.eye(6)
    F[0, 3] = F[1, 4] = F[2, 5] = dt
    return F @ x

# Define measurement function
def hx(x):
    return x  # Return all state components (position and velocity)

# Define the Unscented Kalman Filter
class UnscentedKalmanFilter:
    def __init__(self, dim_x, dim_z, dt, fx, hx, Q, R, alpha, beta, kappa):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.dt = dt
        self.fx = fx
        self.hx = hx
        self.Q = Q
        self.R = R
        
        self.x = np.zeros(dim_x)
        self.P = np.eye(dim_x)
        
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
        Psqrt = np.linalg.cholesky(self.P)
        sigma_points = np.zeros((2 * self.dim_x + 1, self.dim_x))
        sigma_points[0] = self.x
        for i in range(self.dim_x):
            sigma_points[i + 1] = self.x + self.gamma * Psqrt[:, i]
            sigma_points[self.dim_x + i + 1] = self.x - self.gamma * Psqrt[:, i]
        return sigma_points

    def predict(self):
        sigma_points = self.sigma_points()
        x_pred = np.zeros(self.dim_x)
        P_pred = np.zeros((self.dim_x, self.dim_x))
        
        for i in range(2 * self.dim_x + 1):
            sigma_points[i] = self.fx(sigma_points[i], self.dt)
            x_pred += self.Wm[i] * sigma_points[i]
        
        for i in range(2 * self.dim_x + 1):
            y = sigma_points[i] - x_pred
            P_pred += self.Wc[i] * np.outer(y, y)
        P_pred += self.Q
        
        self.x = x_pred
        self.P = P_pred
    
    def update(self, z):
        sigma_points = self.sigma_points()
        Z = np.zeros((2 * self.dim_x + 1, self.dim_z))
        z_pred = np.zeros(self.dim_z)
        Pz = np.zeros((self.dim_z, self.dim_z))
        Pxz = np.zeros((self.dim_x, self.dim_z))
        
        for i in range(2 * self.dim_x + 1):
            Z[i] = self.hx(sigma_points[i])
            z_pred += self.Wm[i] * Z[i]
        
        for i in range(2 * self.dim_x + 1):
            y = Z[i] - z_pred
            Pz += self.Wc[i] * np.outer(y, y)
        
        Pz += self.R
        
        for i in range(2 * self.dim_x + 1):
            x_diff = sigma_points[i] - self.x
            z_diff = Z[i] - z_pred
            Pxz += self.Wc[i] * np.outer(x_diff, z_diff)
        
        K = Pxz @ np.linalg.inv(Pz)
        self.x += K @ (z - z_pred)
        self.P -= K @ Pz @ K.T

# Simulation parameters
dt = 0.1
total_time = 10
timesteps = int(total_time / dt)
Q = np.eye(6) * 10**-2  # Process noise
R = np.eye(6) * 10**-1  # Measurement noise

# Initialize UKF with example parameters
ukf = UnscentedKalmanFilter(dim_x=6, dim_z=6, dt=dt, fx=fx, hx=hx, Q=Q, R=R, alpha=1e-3, beta=2, kappa=0)

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
    ukf.predict()
    ukf.update(z)
    ukf_estimates.append(ukf.x.copy())

ukf_estimates = np.array(ukf_estimates)



# Plot results with measurements included
fig, axs = plt.subplots(3, 2, figsize=(12, 12))
time = np.arange(0, total_time, dt)

axs[0, 0].plot(time, true_states[:, 0], label='True X', color='red', linestyle='dotted')
axs[0, 0].plot(time, measurements[:, 0], label='Measured X', color='green', linestyle='dashed')
axs[0, 0].plot(time, ukf_estimates[:, 0], alpha=0.5,label='Estimated X', color='magenta')
axs[0, 0].set_title('Position X')
axs[0, 0].legend()

axs[1, 0].plot(time, true_states[:, 1], label='True Y', color='red', linestyle='dotted')
axs[1, 0].plot(time, measurements[:, 1], label='Measured Y', color='green', linestyle='dashed')
axs[1, 0].plot(time, ukf_estimates[:, 1], alpha=0.5,label='Estimated Y', color='magenta')
axs[1, 0].set_title('Position Y')
axs[1, 0].legend()

axs[2, 0].plot(time, true_states[:, 2], label='True Z', color='red', linestyle='dotted')
axs[2, 0].plot(time, measurements[:, 2], label='Measured Z', color='green', linestyle='dashed')
axs[2, 0].plot(time, ukf_estimates[:, 2], alpha=0.5,label='Estimated Z', color='magenta')
axs[2, 0].set_title('Position Z')
axs[2, 0].legend()

axs[0, 1].plot(time, true_states[:, 3], label='True VX', color='red', linestyle='dotted')
axs[0, 1].plot(time, measurements[:, 3], label='Measured VX', color='green', linestyle='dashed')
axs[0, 1].plot(time, ukf_estimates[:, 3], alpha=0.5,label='Estimated VX', color='magenta')
axs[0, 1].set_title('Velocity X')
axs[0, 1].legend()

axs[1, 1].plot(time, true_states[:, 4], label='True VY', color='red', linestyle='dotted')
axs[1, 1].plot(time, measurements[:, 4], label='Measured VY', color='green', linestyle='dashed')
axs[1, 1].plot(time, ukf_estimates[:, 4], alpha=0.5,label='Estimated VY', color='magenta')
axs[1, 1].set_title('Velocity Y')
axs[1, 1].legend()

axs[2, 1].plot(time, true_states[:, 5], label='True VZ', color='red', linestyle='dotted')
axs[2, 1].plot(time, measurements[:, 5], label='Measured VZ', color='green', linestyle='dashed')
axs[2, 1].plot(time, ukf_estimates[:, 5], alpha=0.5, label='Estimated VZ', color='magenta')
axs[2, 1].set_title('Velocity Z')
axs[2, 1].legend()

plt.tight_layout()
plt.show()
