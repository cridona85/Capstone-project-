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
# ======================================
# 1. Define file paths
# ======================================
file_path_inputs = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_6\initial_inputs.npy"
file_path_output = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_6\initial_outputs.npy"
new_input1 = np.array([[0.296169, 0.541996, 0.293551, 0.809418, 0.317393]]) 
new_output1 = np.array([-0.8968372092673046])
new_input2 = np.array([[0.844816, 0.260545, 0.616459, 0.923239, 0.096011]]) 
new_output2 = np.array([-0.7633533240897648])
new_input3 = np.array([[0.453689, 0.226969, 0.976939, 0.734051, 0.010401]]) 
new_output3 = np.array([-0.6712042023399859])
new_input4 = np.array([[0.840499, 0.126172, 0.957794, 0.845630, 0.008041]])
new_output4 = np.array([-1.1022363882949306])
new_input5 = np.array([[0.152563, 0.311027, 0.729861, 0.646393, 0.068354]])
new_output5= np.array([-0.8279472145381053])
new_input6 = np.array([[0.421579, 0.226908, 0.978674, 0.450073, 0.038053]])
new_output6 = np.array([-1.1022363882949306])
new_input7 = np.array([[0.771865, 0.384289, 0.135537, 0.960838, 0.368258]])
new_output7 = np.array([-1.2930382991372498])
new_input8 = np.array([[0.459664, 0.157470, 0.857951, 0.668057, 0.005990]])
new_output8 = np.array([-0.6623532142461501])
new_input9 = np.array([[0.204553, 0.651523, 0.022591, 0.760552, 0.021296]])
new_output9 = np.array([-1.168779590204196])
new_input10 = np.array([[0.419983, 0.284134, 0.304300, 0.770171, 0.142473]])
new_output10 = np.array([-0.464413944863255])
new_input11 = np.array([[0.406735, 0.181303, 0.590789, 0.882436, 0.020603]])
new_output11 = np.array([-0.45722342632763796])
# ======================================
# 2. Load .npy data files
# ======================================
inputs = np.load(file_path_inputs)
outputs = np.load(file_path_output)
inputs = np.vstack([inputs, new_input1, new_input2, new_input3, new_input4,new_input5, new_input6, new_input7, new_input8, new_input9, new_input10, new_input11])
outputs = np.append(outputs, np.array([new_output1, new_output2, new_output3,new_output4,new_output5,new_output6,new_output7,new_output8,new_output9,new_output10,new_output11]))

# ======================================
# 3. Split into x1, x2,x3 y and X
# ======================================
x1 = inputs[:, 0]
x2 = inputs[:, 1]
x3 = inputs[:, 2]
x4 = inputs[:, 3]
x5 = inputs[:, 4]
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
    "y": y
})
X = df[['x1', 'x2','x3','x4','x5']].values
y = outputs.flatten()

# Display
print(df.head(40))
print(X)
print("X shape:", X.shape)
print("y shape:", y.shape)

# Annotate the best point
best_idx = np.argmax(df["y"])
best_point = df.iloc[best_idx]
previous_best_y = np.max(outputs[:-3])  # best y before adding new points
new_best_y = best_point["y"]
print(f"Current best (y): {df['y'].max():.4e}")
print(f"Inputs producing best result: [{best_point['x1']:.6f}, {best_point['x2']:.6f}, {best_point['x3']:.6f}]")
percent_improvement = ((new_best_y - previous_best_y) / abs(previous_best_y)) * 100
print(f"Percentage improvement from old best to new best: {percent_improvement:.2f}%")



# ===============================
# 2. Fit Gaussian Process
# ===============================
kernel = RBF(length_scale=[1.0]*5, length_scale_bounds=(1e-2, 100.0)) + WhiteKernel(noise_level=1e-8, noise_level_bounds=(1e-10, 1e-1))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=20, normalize_y=True, random_state=42)
gp.fit(X, y)

# ===============================
# 3. Create a grid of candidate points
# ===============================
# For 5D, we usually sample randomly instead of full grid due to curse of dimensionality
n_candidates = 50000
np.random.seed(42)

X_candidates = np.column_stack([
    np.random.uniform(df['x1'].min(), df['x1'].max(), n_candidates),
    np.random.uniform(df['x2'].min(), df['x2'].max(), n_candidates),
    np.random.uniform(df['x3'].min(), df['x3'].max(), n_candidates),
    np.random.uniform(df['x4'].min(), df['x4'].max(), n_candidates),
    np.random.uniform(df['x5'].min(), df['x5'].max(), n_candidates),
])

# ===============================
# 4. Predict mean and std
# ===============================
y_mean, y_std = gp.predict(X_candidates, return_std=True)

# ===============================
# 5. Upper Confidence Bound (UCB)
# ===============================
beta = 0.2  # Exploration-exploitation tradeoff (tuneable)
UCB = y_mean + beta * y_std

# ===============================
# 6. Select next point based on UCB
# ===============================
next_index = np.argmax(UCB)
next_x = X_candidates[next_index]



print("Next suggested input (x1,x2,x3,x4,x5):", next_x)

next_y_mean, next_y_std = gp.predict(next_x.reshape(1, -1), return_std=True)

print("\n===== NEW BEST SUGGESTED POINT =====")
print(f"Suggested inputs (x1, x2, x3, x4,x5): {next_x}")
print(f"Predicted output mean: {next_y_mean[0]:.6f}")
print(f"Predicted output uncertainty (std): {next_y_std[0]:.6f}")

# Optionally, compare with current best observed output
current_best_y = np.max(y)
print(f"\nCurrent best observed output: {current_best_y:.6f}")
print(f"Expected improvement from next point: {next_y_mean[0] - current_best_y:.6f}")
