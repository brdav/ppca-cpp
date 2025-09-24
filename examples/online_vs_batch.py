"""Example: Compare batch vs. online (mini-batch) PPCA fits
Run: python examples/online_vs_batch.py
"""

import numpy as np
from ppca import PPCA

rng = np.random.RandomState(3)
# moderately large synthetic dataset
X = rng.normal(size=(2000, 20))

# Batch fit
model_batch = PPCA(n_components=5, random_state=0)
model_batch.fit(X)
params_batch = model_batch.get_params()

# Online/mini-batch fit (batch_size=600)
model_online = PPCA(n_components=5, batch_size=600, random_state=0)
model_online.fit(X)
params_online = model_online.get_params()

print("Batch noise_variance:", params_batch.get("noise_variance"))
print("Online noise_variance:", params_online.get("noise_variance"))

# Compare components (subspace angle)
try:
    from scipy.linalg import subspace_angles

    n_components = params_batch["components"].shape[0]
    angle = np.rad2deg(
        subspace_angles(params_batch["components"].T, params_online["components"].T)[0]
    )
    print(f"Angle between subspaces (deg): {angle:.6f}")
except Exception:
    print("scipy not available — skip subspace angle computation")

print("Done")
