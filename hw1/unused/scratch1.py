import numpy as np
import matplotlib.pyplot as plt

# Generate sample data ranging from 10^-15 to 10^-7
np.random.seed(42)
data = np.random.uniform(low=1e-15, high=1e-7, size=1000)

# Apply log10 transformation
log_data = np.log10(data)

# Plot the original data
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(data, bins=30, edgecolor='k', alpha=0.7)
plt.title('Original Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.yscale('log')
plt.xscale('log')

# Plot the log-transformed data
plt.subplot(1, 2, 2)
plt.hist(log_data, bins=30, edgecolor='k', alpha=0.7)
plt.title('Log-transformed Data')
plt.xlabel('log10(Value)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
