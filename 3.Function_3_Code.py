import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

# ======================================
# 1. Load initial data and append new points
# ======================================
file_path_inputs = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_3\initial_inputs.npy"
file_path_output = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_3\initial_outputs.npy"

new_points = np.array([
    [0.157895, 0.210526, 0.500000],
    [0.473684, 0.894737, 0.500000],
    [0.421053, 0.473684, 0.500000],
    [0.399339, 0.690781, 0.545587],
    [0.526316, 0.160526, 0.500000],
    [0.734694, 0.520677, 0.500000],
    [0.489796, 0.578195, 0.500000],
    [0.510204, 0.424812, 0.500000],
    [0.367347, 0.667788, 0.508044],
    [0.306122, 0.613860, 0.508044],
    [0.326531, 0.577907, 0.508044]
    
])
new_outputs = np.array([
    -0.06326395090068407,
    -0.03505347865351867,
    -0.011326279662511995,
    -0.024566239888812874,
    -0.03353882734124177,
    -0.03097004336598361,
    -0.015841467484459634,
    -0.029388284366143783,
    -0.008374473062857676,-0.005455581102065078,-0.022800320165013578
])

inputs = np.load(file_path_inputs)
outputs = np.load(file_path_output)

inputs = np.vstack([inputs, new_points])
outputs = np.append(outputs, new_outputs)

# ======================================
# 2. Prepare DataFrame
# ======================================
df = pd.DataFrame(inputs, columns=['x1','x2','x3'])
df['y'] = outputs
X = df[['x1','x2','x3']].values
y = df['y'].values

print(df.head())
print("X shape:", X.shape, "y shape:", y.shape)

# ======================================
# 3. Standardize inputs and outputs
# ======================================
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

scaler_y = StandardScaler()
y_scaled = scaler_y.fit_transform(y.reshape(-1,1)).ravel()

# ======================================
# 4. Fit Gaussian Process with Matérn kernel + WhiteKernel
# ======================================
kernel = Matern(length_scale=[0.5,0.5,0.5], length_scale_bounds=(0.05,1.5), nu=1.5) + WhiteKernel(noise_level=1e-4)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, random_state=42)
gp.fit(X_scaled, y_scaled)

print("\nFitted kernel:", gp.kernel_)
print("Learned length scales:", gp.kernel_.k1.length_scale)
print("Learned noise level:", gp.kernel_.k2.noise_level)

# ======================================
# 5. Create dense grid for prediction with padding
# ======================================
grid_size = 50
pad = 0.05

x1_min, x1_max = X[:,0].min(), X[:,0].max()
x2_min, x2_max = X[:,1].min(), X[:,1].max()
x3_min, x3_max = X[:,2].min(), X[:,2].max()

x1_grid = np.linspace(max(0, x1_min-pad), min(1, x1_max+pad), grid_size)
x2_grid = np.linspace(max(0, x2_min-pad), min(1, x2_max+pad), grid_size)
x3_slices = np.linspace(max(0, x3_min-pad), min(1, x3_max+pad), 3)

best_ucb = -np.inf
next_x_best = None
k_ucb = 0.8  # UCB coefficient

# ======================================
# 6. Evaluate UCB across slices
# ======================================
for x3_fixed in x3_slices:
    x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
    X_grid = np.column_stack([
        x1_mesh.ravel(),
        x2_mesh.ravel(),
        np.full_like(x1_mesh.ravel(), x3_fixed)
    ])
    X_grid_scaled = scaler_X.transform(X_grid)

    y_mean, y_std = gp.predict(X_grid_scaled, return_std=True)

    # Upper Confidence Bound acquisition
    ucb = y_mean + k_ucb * y_std

    # Optional: penalize points far from existing data (avoid corners)
    distances = np.min(np.linalg.norm(X_grid_scaled[:,None,:] - X_scaled[None,:,:], axis=2), axis=1)
    ucb -= 0.5 * distances  # tune coefficient to adjust exploration

    idx = np.argmax(ucb)
    if ucb[idx] > best_ucb:
        best_ucb = ucb[idx]
        next_x_best = X_grid[idx]

# ======================================
# 7. Visualize UCB on middle slice
# ======================================
x3_fixed = x3_slices[1]
x1_mesh, x2_mesh = np.meshgrid(x1_grid, x2_grid)
X_grid = np.column_stack([
    x1_mesh.ravel(),
    x2_mesh.ravel(),
    np.full_like(x1_mesh.ravel(), x3_fixed)
])
X_grid_scaled = scaler_X.transform(X_grid)
y_mean, y_std = gp.predict(X_grid_scaled, return_std=True)
ucb = y_mean + k_ucb * y_std
distances = np.min(np.linalg.norm(X_grid_scaled[:,None,:] - X_scaled[None,:,:], axis=2), axis=1)
ucb -= 0.5 * distances

plt.figure(figsize=(8,6))
plt.contourf(x1_mesh, x2_mesh, ucb.reshape(grid_size, grid_size), levels=50, cmap='viridis')
plt.colorbar(label='UCB')
plt.scatter(X[:,0], X[:,1], c='red', s=50, label='Existing points')
plt.scatter(next_x_best[0], next_x_best[1], c='white', s=100, marker='*', label='Next suggested')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(f'Bayesian Optimization (UCB) slice at x3 = {x3_fixed:.2f}')
plt.legend()
plt.show()

# ======================================
# 8. Predict GP output for next-best point
# ======================================
next_x_scaled = scaler_X.transform(next_x_best.reshape(1,-1))

next_y_mean, next_y_std = gp.predict(next_x_best.reshape(1,-1), return_std=True)

# Extract scalars
next_y_mean = next_y_mean[0]
next_y_std = next_y_std[0]

current_best_y = y.max()
expected_improvement = next_y_mean - current_best_y
percent_improvement = (expected_improvement / abs(current_best_y)) * 100

print("\n===== NEXT BEST SUGGESTED POINT =====")
print(f"Suggested inputs (x1, x2, x3): {next_x_best}")
print(f"Predicted output mean: {next_y_mean:.6f}")
print(f"Predicted output uncertainty (std): {next_y_std:.6f}")
print(f"Current best observed output: {current_best_y:.6f}")
print(f"Expected improvement: {expected_improvement:.6f} ({percent_improvement:.2f}%)")
