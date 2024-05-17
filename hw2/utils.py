# reference: https://github.com/Al-khwarizmi-780/OpenKF/blob/main/python/examples/Square_Root_Unscented_Kalman_Filter.ipynb

import numpy as np

def cholupdate(L, W, beta):

    r = np.shape(W)[1]
    m = np.shape(L)[0]
       
    for i in range(r):
        L_out = np.copy(L)
        b = 1.0
        
        for j in range(m):
            Ljj_pow2 = L[j, j]**2
            wji_pow2 = W[j, i]**2
            
            L_out[j, j] = np.sqrt(Ljj_pow2 + (beta / b) * wji_pow2)
            upsilon = (Ljj_pow2 * b) + (beta * wji_pow2)
            
            for k in range(j+1, m):
                W[k, i] -= (W[j, i] / L[j,j]) * L[k,j]
                L_out[k, j] = ((L_out[j, j] / L[j, j]) * L[k,j]) + (L_out[j, j] * beta * W[j, i] * W[k, i] / upsilon)
            
            b += beta * (wji_pow2 / Ljj_pow2)
        
        L = np.copy(L_out)
    
    return L_out

# Define the Square-Root Unscented Kalman Filter
class SRUnscentedKalmanFilter:
    def __init__(self, dim_x, dim_z, fx, hx, Q, R, alpha=1e-3, beta=2, kappa=0):
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.fx = fx
        self.hx = hx
        self.Q = Q  # process noise
        self.R = R  # measurement noise
        
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
        print("self.p")
        print(self.P)
        Psqrt = np.linalg.cholesky(self.P)
        print(Psqrt)
        sigma_points = np.zeros((2 * self.dim_x + 1, self.dim_x))
        sigma_points[0] = self.x
        for i in range(self.dim_x):
            sigma_points[i + 1] = self.x + self.gamma * Psqrt[:, i]
            sigma_points[self.dim_x + i + 1] = self.x - self.gamma * Psqrt[:, i]
        return sigma_points

    def predict(self, dt):
        sigma_points = self.sigma_points()
        x_pred = np.zeros(self.dim_x)
        P_pred_sqrt = np.zeros((self.dim_x, self.dim_x))
        
        for i in range(2 * self.dim_x + 1):
            sigma_points[i] = self.fx(sigma_points[i], dt)
            x_pred += self.Wm[i] * sigma_points[i]
        
        for i in range(2 * self.dim_x + 1):
            y = sigma_points[i] - x_pred
            P_pred_sqrt += self.Wc[i] * np.outer(y, y)
        P_pred_sqrt += self.Q
        
        self.x = x_pred
        self.P = P_pred_sqrt@P_pred_sqrt.T + self.Q
        
        
    
    def update(self, z):
        print("update")
        sigma_points = self.sigma_points()
        Z = np.zeros((2 * self.dim_x + 1, self.dim_z))
        z_pred = np.zeros(self.dim_z)
        Pz_sqrt = np.zeros((self.dim_z, self.dim_z))
        Pxz_sqrt = np.zeros((self.dim_x, self.dim_z))
        
        for i in range(2 * self.dim_x + 1):
            Z[i] = self.hx(sigma_points[i])
            z_pred += self.Wm[i] * Z[i]
        
        for i in range(2 * self.dim_x + 1):
            y = Z[i] - z_pred
            Pz_sqrt += self.Wc[i] * np.outer(y, y)
        
        Pz_sqrt += self.R
        
        for i in range(2 * self.dim_x + 1):
            x_diff = sigma_points[i] - self.x
            z_diff = Z[i] - z_pred
            Pxz_sqrt += self.Wc[i] * np.outer(x_diff, z_diff)
        
        Pz = np.linalg.cholesky(Pz_sqrt) 
        K = Pxz_sqrt @ np.linalg.inv(Pz @ Pz.T)
        self.x += K @ (z - z_pred)
        self.P -= K @ Pz @ Pz.T @ K.T
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
