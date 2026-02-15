import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

# ======================================
# 1. Load and append data
# ======================================
file_path_inputs = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_2\initial_inputs.npy"
file_path_output = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_2\initial_outputs.npy"

inputs = np.load(file_path_inputs)
outputs = np.load(file_path_output)

new_inputs = np.array([
    [0.704082, 0.290816],
    [0.695764, 0.540976],
    [0.686869, 0.424242],
    [0.551020, 0.408163],
    [0.551020, 0.244898],
    [0.795918, 0.897959],
    [0.755102, 0.183673],
    [0.040816, 0.857143],
    [0.719511, 0.538107],
    [0.700577, 0.537331],
    [0.700577, 0.744922]
])

new_outputs = np.array([
    0.6671334865943398,
    0.5956572423053663,
    0.4539386926551567,
    0.24130487484788948,
    0.15027923014982658,
    0.043843674079144516,
    0.2434335499626224,
    0.18656377257747334,
    0.6214428433399743,
    0.6132592515850679,
    0.4836193304656602
])

X = np.vstack([inputs, new_inputs])
y = np.append(outputs, new_outputs)

# ======================================
# 2. Scale inputs (NEW)
# ======================================
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# ======================================
# 3. Fit Gaussian Process (FIXED)
# ======================================
kernel = Matern(
    length_scale=[0.3, 0.3],
    length_scale_bounds=(0.05, 1.0),
    nu=1.5
)

gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    normalize_y=True,        # CHANGED
    n_restarts_optimizer=10,
    random_state=42
)

gp.fit(X_scaled, y)

# ======================================
# 4. Conditional candidate grid
# ======================================
grid_size = 50
x1_new = 0.70057687

# Nearest neighbours in x1
nbrs_x1 = NearestNeighbors(n_neighbors=5)
nbrs_x1.fit(X[:, [0]])

_, idx = nbrs_x1.kneighbors([[x1_new]])
x2_neighbours = X[idx[0], 1]

x2_min, x2_max = x2_neighbours.min(), x2_neighbours.max()

x2_grid = np.linspace(x2_min, x2_max, grid_size)
x1_grid = np.full(grid_size, x1_new)

X_grid = np.column_stack([x1_grid, x2_grid])
X_grid_scaled = scaler_X.transform(X_grid)

# ======================================
# 5. GP prediction
# ======================================
y_mean, y_std = gp.predict(X_grid_scaled, return_std=True)

# ======================================
# 6. Acquisition (FIXED)
# ======================================
beta = 0.4
ucb = y_mean + beta * y_std

# Distance penalty (weakened)
nbrs = NearestNeighbors(n_neighbors=1).fit(X)
dist_to_data, _ = nbrs.kneighbors(X_grid)
dist_to_data = dist_to_data.ravel()

acquisition = ucb - 0.1 * dist_to_data

# ======================================
# 7. Select next point
# ======================================
next_index = np.argmax(acquisition)
next_x = X_grid[next_index]

print("\n===== NEW BEST SUGGESTED POINT =====")
print(f"Suggested inputs (x1, x2): {next_x}")

# ======================================
# 8. Predict at suggested point
# ======================================
next_y_mean, next_y_std = gp.predict(
    scaler_X.transform(next_x.reshape(1, -1)),
    return_std=True
)

print(f"Predicted output mean: {next_y_mean[0]:.4f}")
print(f"Predicted uncertainty (std): {next_y_std[0]:.4f}")

# ======================================
# 9. Visualisation
# ======================================
plt.figure(figsize=(8, 6))
plt.plot(x2_grid, acquisition, lw=2)
plt.scatter(next_x[1], acquisition[next_index], c="red", s=80, label="Next")
plt.xlabel("x2 (data-consistent range)")
plt.ylabel("Penalised UCB")
plt.title(f"Conditional BO slice at x1 = {x1_new:.4f}")
plt.legend()
plt.grid(True)
plt.show()
