import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors

from scipy.stats import qmc

# ======================================
# 1. Load data and append new points
# ======================================
file_path_inputs = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_1\initial_inputs.npy"
file_path_output = r"C:\Users\Crist\OneDrive\Documents\programs\Databases\Capstone\Initial_data_points_starter\initial_data\function_1\initial_outputs.npy"

inputs = np.load(file_path_inputs)
outputs = np.load(file_path_output)

new_inputs = np.array([
    [0.367347, 0.102041],
    [0.734694, 0.734694],
    [0.795918, 0.795918],
    [0.727273, 0.767677],
    [0.119274, 0.299747],
    [0.728727, 0.765461],
    [0.322783, 0.196739],
    [0.322788, 0.624415],
    [0.685160, 0.862044],
    [0.078390, 0.907921],
    [0.913837, 0.043286]
])

new_outputs = np.array([
    -6.40608222720135e-74,
    8.829025681475009e-17,
    1.9538087967596605e-39,
    -6.189898180805763e-21,
    2.470707859320033e-75,
    -8.932857527470858e-21,
    8.732181782889409e-43,
    7.412267935971386e-37,
    9.782562731810732e-41,
    1.982932291862667e-247,
    -8.041976978462411e-269
])

inputs = np.vstack([inputs, new_inputs])
outputs = np.append(outputs, new_outputs)

# ======================================
# 2. Prepare dataset
# ======================================
df = pd.DataFrame({
    "x1": inputs[:, 0],
    "x2": inputs[:, 1],
    "y": outputs
})

X = df[["x1", "x2"]].values
y = df["y"].values

# ======================================
# 3. Scale inputs
# ======================================
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# ======================================
# 4. Outlier filtering (joint space)
# ======================================
y_std = (y - y.mean()) / y.std()
joint = np.hstack([X_scaled, y_std.reshape(-1, 1)])

svm = OneClassSVM(kernel="rbf", nu=0.10, gamma="scale")
svm.fit(joint)
inliers = svm.predict(joint) == 1

X_in = X[inliers]
y_in = y[inliers]

if X_in.shape[0] < 5:
    raise RuntimeError("Too few inliers for GP training.")

# ======================================
# 5. Bounded candidate region
# ======================================
pad = 0.05
x1_min, x1_max = max(0, X_in[:, 0].min() - pad), min(1, X_in[:, 0].max() + pad)
x2_min, x2_max = max(0, X_in[:, 1].min() - pad), min(1, X_in[:, 1].max() + pad)

# ======================================
# 6. Candidate generation
# ======================================
sampler = qmc.LatinHypercube(d=2, seed=42)
X_cand = sampler.random(3000)

mask = (
    (X_cand[:, 0] >= x1_min) & (X_cand[:, 0] <= x1_max) &
    (X_cand[:, 1] >= x2_min) & (X_cand[:, 1] <= x2_max)
)
X_cand = X_cand[mask]

# ======================================
# 7. Distance penalty
# ======================================
nbrs = NearestNeighbors(n_neighbors=1).fit(X_in)
dist_penalty, _ = nbrs.kneighbors(X_cand)
dist_penalty = dist_penalty.ravel()

# ======================================
# 8. Gaussian Process (FIXED)
# ======================================
kernel = Matern(length_scale=0.5, nu=1.5)
gp = GaussianProcessRegressor(
    kernel=kernel,
    alpha=1e-6,
    normalize_y=True,          # CHANGED
    n_restarts_optimizer=10,
    random_state=42
)

gp.fit(scaler_X.transform(X_in), y_in)

# ======================================
# 9. Acquisition: Penalised UCB
# ======================================
X_cand_scaled = scaler_X.transform(X_cand)
y_mean, y_std = gp.predict(X_cand_scaled, return_std=True)

beta = 0.3                                   # CHANGED
ucb = y_mean + beta * y_std
acquisition = ucb - 0.05 * dist_penalty     # CHANGED

idx_best = np.argmax(acquisition)
x_next = X_cand[idx_best]

# ======================================
# 10. Visualisation
# ======================================
plt.figure(figsize=(7, 6))
plt.scatter(X_cand[:, 0], X_cand[:, 1], c=acquisition, cmap="viridis", s=8)
plt.scatter(X_in[:, 0], X_in[:, 1], c="white", edgecolor="k", s=70)
plt.scatter(x_next[0], x_next[1], c="red", marker="*", s=220)
plt.colorbar(label="Penalised UCB")
plt.xlim(0, 1)
plt.ylim(0, 1)
plt.title("Revised Distance-aware UCB")
plt.show()

# ======================================
# 11. Prediction at suggested point
# ======================================
y_pred, y_std_pred = gp.predict(
    scaler_X.transform(x_next.reshape(1, -1)),
    return_std=True
)

# ======================================
# 12. Diagnostics
# ======================================
print("\n=== FINAL DIAGNOSTICS ===")
print(f"Total evaluations: {len(df)}")
print(f"Inliers used: {X_in.shape[0]}")
print(f"Suggested next input: {x_next}")
print(f"Predicted y: {y_pred[0]:.6e}")
print(f"Current best observed y: {df.y.max():.6e}")
