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
file_path_inputs = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_5\initial_inputs.npy"
file_path_output = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_5\initial_outputs.npy"
new_input1 = np.array([[0.280863, 0.792001, 0.872623, 0.950670]]) 
new_output1 = np.array([1242.3463972515578])
new_input2 = np.array([[0.534752, 0.862540, 0.879484, 0.957644]]) 
new_output2 = np.array([1820.214875347875])
new_input3 = np.array([[0.161034, 0.040304, 0.196555, 0.915891]]) 
new_output3 = np.array([37.90051207500032])
new_input4 = np.array([[0.823569, 0.762715, 0.836275, 0.936509]])
new_output4 = np.array([1791.974448024398])
new_input5 = np.array([[0.825776, 0.852736, 0.804440, 0.943017]])
new_output5 = np.array([2132.5145605543207])
new_input6 = np.array([[0.825776, 0.852736, 0.804440, 0.943017]])
new_output6 = np.array([2132.5145605543207])
new_input7 = np.array([[0.140627, 0.317598, 0.867043, 0.929857]])
new_output7 = np.array([510.6917319499449])
new_input8 = np.array([[0.834051, 0.476783, 0.819746, 0.956017]])
new_output8 = np.array([1297.8045668080533])
new_input9 = np.array([[0.812613, 0.100574, 0.843632, 0.949938]])
new_output9 = np.array([1168.8396036960723])
new_input10 = np.array([[0.736491, 0.825706, 0.644877, 0.949834]])
new_output10 = np.array([1162.9980299994934])
new_input11 = np.array([[0.157568, 0.110991, 0.103233, 0.080375]])
new_output11 = np.array([161.6955280233999])
# ======================================
# 2. Load .npy data files
# ======================================
inputs = np.load(file_path_inputs)
outputs = np.load(file_path_output)
inputs = np.vstack([inputs, new_input1, new_input2, new_input3, new_input4, new_input5,new_input6,new_input7,new_input8,new_input9,new_input10,new_input11])
outputs = np.append(outputs, np.array([new_output1, new_output2, new_output3,new_output4, new_output5,new_output6,new_output7,new_output8,new_output9,new_output10,new_output11]))

# ======================================
# 3. Split into x1, x2,x3 y and X
# ======================================
x1 = inputs[:, 0]
x2 = inputs[:, 1]
x3 = inputs[:, 2]
x4 = inputs[:, 3]
y = outputs

# ======================================
# 4. Put into a DataFrame
# ======================================
df = pd.DataFrame({
    "x1": x1,
    "x2": x2,
    "x3": x3,
    "x4": x4,
    "y": y
})
X = df[['x1', 'x2','x3','x4']].values
y = outputs.flatten()

# Display
print(df.head(40))
print(X)
print("X shape:", X.shape)
print("y shape:", y.shape)

# ======================================
# 5. Annotate the best point
# ======================================
best_idx = np.argmax(df["y"])
best_point = df.iloc[best_idx]
previous_best_y = np.max(outputs[:-3])  # best y before adding new points
new_best_y = best_point["y"]
print(f"Current best (y): {df['y'].max():.4e}")
print(f"Inputs producing best result: [{best_point['x1']:.6f}, {best_point['x2']:.6f}, {best_point['x3']:.6f}]")
percent_improvement = ((new_best_y - previous_best_y) / abs(previous_best_y)) * 100
print(f"Percentage improvement from old best to new best: {percent_improvement:.2f}%")

# ---------- Transform y (log) and scale X ----------
y_log = np.log1p(y)               # compress range; works for y>=0
X_scaler = StandardScaler()
X_scaled = X_scaler.fit_transform(X)

# Optionally standardize the log target (recommended):
y_mean = y_log.mean()
y_std = y_log.std()
y_scaled = (y_log - y_mean) / (y_std + 1e-12)   # keep as 1D array
# We'll do GP training on y_scaled

# ---------- Kernel: ARD Matern + constant + White (noise bounds not fixed) ----------
d = X.shape[1]
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=np.ones(d),
                                     length_scale_bounds=(1e-2, 1e2),
                                     nu=2.5) + WhiteKernel(noise_level=0.01, noise_level_bounds=(1e-6, 1e1))

gp = GaussianProcessRegressor(kernel=kernel,
                              alpha=0.0,                      # we use WhiteKernel for noise
                              optimizer='fmin_l_bfgs_b',
                              n_restarts_optimizer=25,        # more restarts for robust hyperparam search
                              normalize_y=False,              # already scaled y
                              random_state=42)

gp.fit(X_scaled, y_scaled)
print("Learned kernel:", gp.kernel_)
print("Log Marginal Likelihood:", gp.log_marginal_likelihood(gp.kernel_.theta))

# ---------- Candidate generation: random / quasi-random sample (cover space better) ----------
n_candidates = 100000
rng = np.random.RandomState(0)
X_grid = np.column_stack([
    rng.uniform(df['x1'].min(), df['x1'].max(), n_candidates),
    rng.uniform(df['x2'].min(), df['x2'].max(), n_candidates),
    rng.uniform(df['x3'].min(), df['x3'].max(), n_candidates),
    rng.uniform(df['x4'].min(), df['x4'].max(), n_candidates),
])
X_grid_scaled = X_scaler.transform(X_grid)

# ---------- Predict in transformed space (mean and std) ----------
y_mean_scaled, y_std_scaled = gp.predict(X_grid_scaled, return_std=True)

# ---------- Expected Improvement (computed in scaled log space) ----------
current_best_scaled = np.max((np.log1p(y) - y_mean) / (y_std + 1e-12))  # careful: we standardized; simpler use max(y_scaled)
current_best_scaled = np.max((y_log - y_mean) / (y_std + 1e-12))   # equal to np.max(y_scaled)
xi = 0.01
improvement = y_mean_scaled - current_best_scaled - xi
Z = improvement / (y_std_scaled + 1e-12)
EI = improvement * norm.cdf(Z) + y_std_scaled * norm.pdf(Z)
EI = np.maximum(EI, 0.0)

# ---------- Choose best candidate ----------
next_index = np.argmax(EI)
next_x = X_grid[next_index]
next_x_scaled = X_grid_scaled[next_index]
print("Next suggested input (original scale):", next_x)

# ---------- Predict back to original y scale for reporting ----------
next_y_mean_scaled, next_y_std_scaled = gp.predict(next_x_scaled.reshape(1,-1), return_std=True)
# undo standardization and log transform:
next_y_log = next_y_mean_scaled[0] * (y_std + 1e-12) + y_mean
next_y_pred = np.expm1(next_y_log)       # predicted y (original scale)
print("Predicted y (original scale) at next point (mean):", next_y_pred)
print("Predicted y uncertainty (approx, original scale):", 
      np.expm1(next_y_log + next_y_std_scaled[0]*y_std) - next_y_pred)

# ---------- Diagnostics ----------
import numpy.linalg as la
lengthscales = gp.kernel_.k2.length_scale if hasattr(gp.kernel_.k2, 'length_scale') else None
print("Kernel lengthscales (ARD):", lengthscales)

# ---------- Visualize EI distribution ----------
plt.figure(figsize=(8,4))
plt.hist(EI, bins=60)
plt.title("EI distribution (transformed log space)")
plt.xlabel("Expected Improvement")
plt.ylabel("Frequency")
plt.show()

# ======================================
# 9. Compute Expected Improvement (EI) at the suggested next point
# ======================================
# EI quantifies how much improvement we can expect over the current best observation,
# balancing exploration (high uncertainty) and exploitation (high predicted mean).
# Predict mean and std in scaled log space
next_y_mean_scaled, next_y_std_scaled = gp.predict(next_x_scaled.reshape(1, -1), return_std=True)

# Undo standardization (back to log space)
next_y_log_mean = next_y_mean_scaled[0] * (y_std + 1e-12) + y_mean
next_y_log_std = next_y_std_scaled[0] * (y_std + 1e-12)

# Undo log transform to return to original scale (mean and variance propagation)
next_y_mean_original = np.expm1(next_y_log_mean)
# approximate variance in original space using delta method
next_y_var_original = (np.expm1(next_y_log_mean + next_y_log_std**2) - np.expm1(next_y_log_mean))**2
next_y_std_original = np.sqrt(next_y_var_original)

# Compute current best in original scale
current_best_original = np.max(y)

# Expected Improvement in ORIGINAL space
xi = 0.01
improvement = next_y_mean_original - current_best_original - xi
Z = improvement / (next_y_std_original + 1e-12)
EI_original = improvement * norm.cdf(Z) + next_y_std_original * norm.pdf(Z)
EI_original = np.maximum(EI_original, 0.0)

print("\n===== EXPECTED IMPROVEMENT ANALYSIS (ORIGINAL SCALE) =====")
print(f"Current best observed y (original): {current_best_original:.6f}")
print(f"Predicted mean at suggested point (original): {next_y_mean_original:.6f}")
print(f"Predicted uncertainty (std, original): {next_y_std_original:.6f}")
print(f"Expected Improvement (EI, original): {EI_original:.6e}")

# GP prediction at next suggested input
next_y_mean_scaled, next_y_std_scaled = gp.predict(next_x_scaled.reshape(1, -1), return_std=True)

# Convert back to original scale
next_y_log_mean = next_y_mean_scaled[0] * y_std + y_mean
next_y_mean_original = np.expm1(next_y_log_mean)

print("Predicted mean y at next point (original scale):", next_y_mean_original)


from scipy.stats import norm

# Current best
current_best = np.max(y)

# GP predicted mean and std in original scale
mu = next_y_mean_original
sigma = next_y_std_original

# Expected new best using Gaussian properties
# Formula: E[max(X, c)] for X ~ N(mu, sigma^2), c = current best
Z = (current_best - mu) / sigma
expected_new_best = mu * norm.cdf(Z) + current_best * (1 - norm.cdf(Z)) + sigma * norm.pdf(Z)

print("Expected new best y after sampling next point:", expected_new_best)

samples = np.random.normal(next_y_mean_original, next_y_std_original, 10000)
expected_new_best_mc = np.mean(np.maximum(samples, current_best))
print("Expected new best y (Monte Carlo estimate):", expected_new_best_mc)
