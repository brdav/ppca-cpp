"""Quick start example for PPCA
Run: python examples/quick_start.py
"""

import numpy as np
from ppca import PPCA

rng = np.random.RandomState(0)
# Synthetic data: 100 samples, 6 features, latent dim 2
Z_true = rng.normal(size=(100, 2))
W = rng.normal(scale=1.0, size=(6, 2))
X = Z_true @ W.T + rng.normal(scale=0.1, size=(100, 6))

model = PPCA(n_components=2)
model.fit(X)

print("Fitted PPCA")
print("components_ shape:", model.components_.shape)
print("mean_ shape:", model.mean_.shape)
print("noise_variance_:", getattr(model, "noise_variance_", None))

# Latent posterior mean
mZ, covZ = model.posterior_latent(X)
print("posterior latent mean shape:", mZ.shape)
print("Done")
