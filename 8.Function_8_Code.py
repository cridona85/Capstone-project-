import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from scipy.stats import norm
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C , WhiteKernel
from scipy.stats import norm
from sklearn.gaussian_process.kernels import Matern

from sklearn.preprocessing import StandardScaler
np.random.seed(42)


# ======================================
# 1. Define file paths
# ======================================
file_path_inputs = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_8\initial_inputs.npy"
file_path_output = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_8\initial_outputs.npy"

new_input1 = np.array([[0.299010, 0.223021, 0.250826, 0.155864, 0.851361, 0.653003, 0.203269, 0.325062]]) 
new_output1 = np.array([9.8393174627591])  

new_input2 = np.array([[0.107582, 0.273912, 0.203028, 0.053818, 0.897551, 0.664482, 0.278924, 0.771234]]) 
new_output2 = np.array([9.9120780683799])
new_input3 = np.array([[0.197874, 0.352337, 0.121556, 0.142444, 0.876550, 0.731629, 0.340124, 0.683725]]) 
new_output3 = np.array([9.8430776989295])
new_input4 = np.array([[0.154508, 0.092547, 0.103317, 0.199439, 0.894305, 0.238203, 0.218561, 0.677882]])
new_output4 = np.array([9.9118967787191])
new_input5 = np.array([[0.083010, 0.232804, 0.183853, 0.219042, 0.875771, 0.852477, 0.219291, 0.170504]])
new_output5 = np.array([9.8327973182799])
new_input6 = np.array([[0.208969, 0.072241, 0.295718, 0.165068, 0.863496, 0.336988, 0.184770, 0.869133]])
new_output6 = np.array([9.8512946946801])
new_input7 = np.array([[0.114479, 0.204533, 0.260370, 0.061330, 0.891791, 0.115866, 0.274717, 0.542669]])
new_output7 = np.array([9.7744898210984])
new_input8 = np.array([[0.204009, 0.080239, 0.076047, 0.058548, 0.702810, 0.421387, 0.252626, 0.485410]])
new_output8 = np.array([9.938646381405])
new_input9 = np.array([[0.114097, 0.588059, 0.080073, 0.085769, 0.705599, 0.479615, 0.210454, 0.503052]])
new_output9 = np.array([9.7900733380251])
new_input10 = np.array([[0.051182, 0.408022, 0.074569, 0.735239, 0.759601, 0.604153, 0.178485, 0.574868]])
new_output10 = np.array([9.5642829396621])
new_input11 = np.array([[0.501114, 0.140757, 0.133242, 0.119674, 0.819325, 0.216377, 0.276552, 0.216920]])
new_output11 = np.array([9.5701543070015])
# ======================================
# 2. Load .npy data files
# ======================================
inputs = np.load(file_path_inputs)
outputs = np.load(file_path_output)
inputs = np.vstack([inputs, new_input1, new_input2, new_input3, new_input4, new_input5, new_input6, new_input7, new_input8, new_input9, new_input10, new_input11])
outputs = np.append(outputs, np.array([new_output1, new_output2, new_output3,new_output4, new_output5, new_output6, new_output7, new_output8, new_output9, new_output10, new_output11]))



# ======================================
# 3. Split into x1, x2,x3 y and X
# ======================================
x1 = inputs[:, 0]
x2 = inputs[:, 1]
x3 = inputs[:, 2]
x4 = inputs[:, 3]
x5 = inputs[:, 4]
x6 = inputs[:, 5]
x7 = inputs[:, 6]
x8 = inputs[:, 7]
y = outputs

# ======================================
# 4. Put into a DataFrame
# ======================================
df = pd.DataFrame({
    "x1": x1,
    "x2": x2,
    "x3": x3,
    "x4": x4,
    "x5": x5,
    "x6": x6,
    "x7": x7,
    "x8": x8,
    "y": y
})
X = df[['x1', 'x2','x3','x4','x5','x6','x7','x8']].values
y = outputs.flatten()

# Display
print(df.head(100))
print(X)
print("X shape:", X.shape)
print("y shape:", y.shape)
# Annotate the best point
best_idx = np.argmax(df["y"])
best_point = df.iloc[best_idx]
previous_best_y = np.max(outputs[:-3])  # best y before adding new points
new_best_y = best_point["y"]
print(f"Current best (y): {df['y'].max():.4e}")
print(f"Inputs producing best result: [{best_point['x1']:.6f}, {best_point['x2']:.6f}, {best_point['x3']:.6f},{best_point['x4']:.6f},{best_point['x5']:.6f}],{best_point['x6']:.6f}],{best_point['x7']:.7f}],{best_point['x8']:.8f}]")
percent_improvement = ((new_best_y - previous_best_y) / abs(previous_best_y)) * 100
print(f"Percentage improvement from old best to new best: {percent_improvement:.2f}%")


# Kernel with ARD
kernel = Matern(length_scale=[1.0]*8, length_scale_bounds=(1e-2, 10.0), nu=2.5) + WhiteKernel(noise_level=1e-6)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, normalize_y=True)

gp.fit(X, y)

# Define plausible grid using 5th–95th percentiles
grid_size = 50000
bounds = []
for i in range(8):
    lower = np.percentile(X[:,i], 5)
    upper = np.percentile(X[:,i], 95)
    bounds.append((lower, upper))

X_grid = np.random.uniform([b[0] for b in bounds], [b[1] for b in bounds], size=(grid_size, 8))

# Predict mean and std
y_mean, y_std = gp.predict(X_grid, return_std=True)

# Expected Improvement
y_max = np.max(y)
xi = 0.1
with np.errstate(divide='warn'):
    improvement = y_mean - y_max - xi
    Z = improvement / (y_std + 1e-9)
    EI = improvement * norm.cdf(Z) + y_std * norm.pdf(Z)

# Next suggested point
next_index = np.argmax(EI)
next_x = X_grid[next_index]
print("Next suggested input (x1..x8):", next_x)


# Predicted output at next point
next_y_mean, next_y_std = gp.predict(next_x.reshape(1, -1), return_std=True)
print("\n===== NEW BEST SUGGESTED POINT =====")
print(f"Suggested inputs (x1..x8): {next_x}")
print(f"Predicted output mean: {next_y_mean[0]:.6f}")
print(f"Predicted output uncertainty (std): {next_y_std[0]:.6f}")

# Compare with current best
current_best_y = np.max(y)
print(f"\nCurrent best observed output: {current_best_y:.6f}")
print(f"Expected improvement from next point: {next_y_mean[0] - current_best_y:.6f}")


