import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
import matplotlib.pyplot as plt

# ======================================
# 1. Load data and append new points
# ======================================
file_path_inputs = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_4\initial_inputs.npy"
file_path_output = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_4\initial_outputs.npy"

new_points = np.array([
    [0.513410, 0.526219, 0.380761, 0.426987],
    [0.488066, 0.362496, 0.374974, 0.431134],
    [0.371549, 0.410236, 0.317621, 0.438427],
    [0.431379, 0.366525, 0.341442, 0.434503],
    [0.404002, 0.402292, 0.324716, 0.465648],
    [0.426660, 0.413728, 0.298524, 0.364634],
    [0.426321, 0.418651, 0.401658, 0.405084],
    [0.385722, 0.411590, 0.374702, 0.452439],
    [0.387041, 0.392673, 0.377283, 0.405951],
    [0.344925, 0.365049, 0.382426, 0.434117],
    [0.400768, 0.365733, 0.433017, 0.453863]
])
new_outputs = np.array([
    -2.6064682011864764,
    -0.9697243285331036,
    -0.3306618740636291,
    0.3487649906122461,
    -1.0903953261941974,
    -0.6767760631922068,
    0.6004803493115003,
    -0.2481007781152793,0.11895855445723358,0.4133516885866766,-0.08875689574136514
])

inputs = np.load(file_path_inputs)
outputs = np.load(file_path_output)

# Append new points
inputs = np.vstack([inputs, new_points])
outputs = np.append(outputs, new_outputs)

# ======================================
# 2. Prepare DataFrame
# ======================================
df = pd.DataFrame(inputs, columns=['x1','x2','x3','x4'])
df['y'] = outputs
X = df[['x1','x2','x3','x4']].values
y = df['y'].values

print("Data shape:", X.shape, y.shape)
print(df.head())

# ======================================
# 3. Feature scaling
# ======================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ======================================
# 4. Fit Gaussian Process with ARD
# ======================================
kernel = C(1.0,(1e-3,1e3)) * RBF(length_scale=[1.0]*4, length_scale_bounds=(1e-2,1e2)) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-6,1e-1))
gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, n_restarts_optimizer=10, normalize_y=True, random_state=42)
gp.fit(X_scaled, y)

print("\nFitted kernel:", gp.kernel_)

# ======================================
# 5. Generate bounded candidate points
# ======================================
buffer = 0.05
x_min = np.maximum(0, df[['x1','x2','x3','x4']].min().values - buffer)
x_max = np.minimum(1, df[['x1','x2','x3','x4']].max().values + buffer)

n_candidates = 20000
X_candidates = np.random.uniform(low=x_min, high=x_max, size=(n_candidates,4))
X_candidates_scaled = scaler.transform(X_candidates)

# ======================================
# 6. GP prediction
# ======================================
y_mean, y_std = gp.predict(X_candidates_scaled, return_std=True)
y_max = np.max(y)
xi = 0.005

# ======================================
# 7. Expected Improvement (EI)
# ======================================
with np.errstate(divide='warn'):
    improvement = y_mean - y_max - xi
    Z = improvement / (y_std + 1e-9)
    EI = improvement * norm.cdf(Z) + y_std * norm.pdf(Z)

# ======================================
# 8. Upper Confidence Bound (UCB)
# ======================================
beta = 1.0
UCB = y_mean + beta * y_std

# ======================================
# 9. Hybrid acquisition: weighted EI + UCB
# ======================================
w_ei = 0.5
w_ucb = 0.5
hybrid_acq = w_ei * EI + w_ucb * UCB

next_index = np.argmax(hybrid_acq)
next_x = X_candidates[next_index]

print("Next best input suggested by GP (x1,x2,x3,x4):", next_x)

# ======================================
# 10. Optional visualization for first two dimensions
# ======================================
plt.figure(figsize=(8,6))
plt.scatter(X[:,0], X[:,1], c='red', s=50, label='Existing points')
plt.scatter(next_x[0], next_x[1], c='white', s=100, marker='*', label='Next suggested point')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Next suggested point projected on x1-x2 plane (4D GP hybrid)')
plt.legend()
plt.show()

# ======================================
# 11. Predict output for next-best point
# ======================================
next_y_mean, next_y_std = gp.predict(next_x.reshape(1,-1), return_std=True)
current_best_y = np.max(y)
expected_improvement = next_y_mean[0] - current_best_y

print(f"\nCurrent best observed output: {current_best_y:.6f}")
print(f"Predicted output mean at next point: {next_y_mean[0]:.6f}")
print(f"Predicted uncertainty (std): {next_y_std[0]:.6f}")
print(f"Expected improvement: {expected_improvement:.6f}")
