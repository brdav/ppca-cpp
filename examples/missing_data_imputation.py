"""Example: Imputing missing data with PPCA
Run: python examples/missing_data_imputation.py
"""

import numpy as np
from ppca import PPCA

rng = np.random.RandomState(1)
X = rng.normal(size=(80, 5))
# Introduce 10% missing completely at random
mask = rng.uniform(size=X.shape) < 0.1
X_missing = X.copy()
X_missing[mask] = np.nan

model = PPCA(n_components=2)
model.fit(X_missing)

print("Fitted PPCA on data with missing values")
# Mean imputation from posterior predictive
X_imputed_mean, X_imputed_cov = model.impute_missing(X_missing)
print("X_imputed_mean shape:", X_imputed_mean.shape)
# Show a few imputed entries
n_show = 8
missing_idx = np.argwhere(np.isnan(X_missing))[:n_show]
for i, j in missing_idx:
    print(f"sample {i} feature {j}: imputed={X_imputed_mean[i,j]:.4f}")

# Multiple imputation: sample 5 complete datasets
X_samples = model.sample_missing(X_missing, n_draws=5)
print("X_samples shape:", X_samples.shape)
print("Done")
