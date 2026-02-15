import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from scipy.stats import norm

# ======================================
# 1. Load data and append new points
# ======================================
file_path_inputs = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_7\initial_inputs.npy"
file_path_output = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_7\initial_outputs.npy"

inputs = np.load(file_path_inputs)
outputs = np.load(file_path_output)

# Add new points
new_input1 = np.array([[0.123397, 0.379005, 0.618888, 0.770178, 0.457913, 0.769742]]) 
new_output1 = np.array([0.2882165652456174])  

inputs = np.load(file_path_inputs)
outputs = np.load(file_path_output)
new_input2 = np.array([[0.069978, 0.082696, 0.364360, 0.210290, 0.385626, 0.616410]]) 
new_output2 = np.array([2.5648122364249124])
new_input3 = np.array([[0.107340, 0.033913, 0.856063, 0.077174, 0.383594, 0.654532]]) 
new_output3 = np.array([1.190364466077317])
new_input4 = np.array([[0.070032, 0.047183, 0.362687, 0.250284, 0.386104, 0.758118]])
new_output4 = np.array([2.425170021820331])
new_input5 = np.array([[0.072132, 0.665516, 0.371643, 0.093777, 0.378924, 0.865853]]) 
new_output5 = np.array([0.5886666234676696])
new_input6 = np.array([[0.076147, 0.092708, 0.389406, 0.107775, 0.355210, 0.626703]])
new_output6 = np.array([2.266370169121572])
new_input7 = np.array([[0.061514, 0.068289, 0.164746, 0.179373, 0.499400, 0.593121]])
new_output7 = np.array([1.1827415013074278])
new_input8 = np.array([[0.418996, 0.022393, 0.100640, 0.268504, 0.396592, 0.628900]])
new_output8 = np.array([1.9780425008732128])
new_input9 = np.array([[0.108104, 0.024214, 0.030831, 0.528220, 0.355452, 0.933762]])
new_output9 = np.array([0.6891803024870201])
new_input10 = np.array([[0.526445, 0.105476, 0.385084, 0.137523, 0.426273, 0.691444]])
new_output10 = np.array([1.506450291739205])
new_input11 = np.array([[0.159396, 0.055903, 0.188210, 0.230559, 0.357143, 0.644387]])
new_output11 = np.array([2.57422657440579])
inputs = np.vstack([inputs, new_input1, new_input2, new_input3, new_input4,new_input5, new_input6, new_input7, new_input8, new_input9, new_input10, new_input11])
outputs = np.append(outputs, np.array([new_output1, new_output2, new_output3,new_output4,new_output5,new_output6,new_output7,new_output8,new_output9,new_output10,new_output11]))



# ======================================
# 2. Prepare X, y
# ======================================
X = inputs
y = outputs.flatten()

# ======================================
# 3. Fit Gaussian Process
# ======================================
kernel = RBF(length_scale=[1.0]*6, length_scale_bounds=(1e-2, 50.0)) + WhiteKernel(noise_level=1e-6, noise_level_bounds=(1e-8, 1e-1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, normalize_y=True, random_state=42)
gp.fit(X, y)
print("Learned kernel:", gp.kernel_)

# ======================================
# 4. Generate random 6D candidate points
# ======================================
n_candidates = 50000
np.random.seed(42)
X_candidates = np.random.rand(n_candidates, 6)

# Scale candidates to observed ranges
for i in range(6):
    X_candidates[:, i] = X_candidates[:, i] * (X[:, i].max() - X[:, i].min()) + X[:, i].min()

# ======================================
# 5. Predict mean and std
# ======================================
y_mean, y_std = gp.predict(X_candidates, return_std=True)

# ======================================
# 6. Probability of Improvement (PI)
# ======================================
current_best = np.max(y)
xi = 0.3  # increased exploration factor

Z = (y_mean - current_best - xi) / (y_std + 1e-9)
PI = norm.cdf(Z)

# Optionally combine PI and UCB for hybrid acquisition
kappa = 2.0
UCB = y_mean + kappa * y_std
UCB_norm = (UCB - UCB.min()) / (UCB.max() - UCB.min() + 1e-12)
PI_norm = (PI - PI.min()) / (PI.max() - PI.min() + 1e-12)
Hybrid_PI_UCB = 0.5 * PI_norm + 0.5 * UCB_norm

# ======================================
# 7. Select next best input according to Hybrid PI+UCB
# ======================================
next_index = np.argmax(Hybrid_PI_UCB)
next_x = X_candidates[next_index]
next_y_mean, next_y_std = gp.predict(next_x.reshape(1, -1), return_std=True)

print("\n===== NEW BEST SUGGESTED POINT (Hybrid PI+UCB) =====")
print(f"Suggested inputs (x1..x6): {next_x}")
print(f"Predicted output mean: {next_y_mean[0]:.6f}")
print(f"Predicted output uncertainty (std): {next_y_std[0]:.6f}")
print(f"Current best observed output: {current_best:.6f}")
print(f"Expected improvement from next point: {next_y_mean[0] - current_best:.6f}")
