import numpy as np

# Define the Unscented Kalman Filter
class UnscentedKalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.fx = None
        self.hx = None
        
        # UKF parameters
        self.alpha = 1e-3  # default for non-additive noise
        self.beta = 2      # optimal for Gaussian distribution
        self.kappa = 0     # optimal for Gaussian distribution
        self.lambd = self.alpha**2 * (dim_x + self.kappa) - dim_x 
        self.gamma = np.sqrt(dim_x + self.lambd) # scaling factor
        
        # Weights for mean and covariance
        self.Wm = np.full(2 * dim_x + 1, 1 / (2 * (dim_x + self.lambd))) # wieghts for mean
        self.Wc = np.full(2 * dim_x + 1, 1 / (2 * (dim_x + self.lambd))) # weights for covariance
        self.Wm[0] = self.lambd / (dim_x + self.lambd)
        self.Wc[0] = self.lambd / (dim_x + self.lambd) + (1 - self.alpha**2 + self.beta)

    def set_fx(self, fx):
        self.fx = fx
        
    def sigma_points(self, x, P):
        Psqrt = (np.linalg.cholesky(P)).T
        sigma_points = np.zeros((2 * self.dim_x + 1, self.dim_x))
        sigma_points[0] = x
        for i in range(self.dim_x):
            sigma_points[i + 1] = x + self.gamma * Psqrt[:, i]
            sigma_points[self.dim_x + i + 1] = x - self.gamma * Psqrt[:, i]
        return sigma_points

    def predict(self, sigma_points, dt, Q):
        x_pred = np.zeros(self.dim_x)
        P_pred = np.zeros((self.dim_x, self.dim_x))
        
        for i in range(2 * self.dim_x + 1):
            sigma_points[i] = self.fx(sigma_points[i], dt)
            x_pred += self.Wm[i] * sigma_points[i]
        
        for i in range(2 * self.dim_x + 1):
            y = sigma_points[i] - x_pred
            P_pred += self.Wc[i] * np.outer(y, y)
        P_pred += Q
        
        return x_pred, P_pred
    
    def update(self, x, z, R):
        sigma_points = self.sigma_points()
        Z = np.zeros((2 * self.dim_x + 1, self.dim_z))
        z_pred = np.zeros(self.dim_z)

        Pxz = np.zeros((self.dim_x, self.dim_z))
        
        for i in range(2 * self.dim_x + 1):
            Z[i] = self.hx(sigma_points[i])
            z_pred += self.Wm[i] * Z[i]
        
        for i in range(2 * self.dim_x + 1):
            y = Z[i] - z_pred
            R += self.Wc[i] * np.outer(y, y)
        
        R += self.R
        
        for i in range(2 * self.dim_x + 1):
            x_diff = sigma_points[i] - x
            z_diff = Z[i] - z_pred
            Pxz += self.Wc[i] * np.outer(x_diff, z_diff)
        
        K = Pxz @ np.linalg.inv(R)
        
        x_new += K @ (z - z_pred)
        R_new -= K @ R @ K.T
        return x_new, R_new
