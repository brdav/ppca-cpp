"""Example: Sampling from the posterior over latents and likelihood
Run: python examples/posterior_sampling.py
"""

import numpy as np
from ppca import PPCA

rng = np.random.RandomState(2)
X = rng.normal(size=(60, 4))
model = PPCA(n_components=2)
model.fit(X)

# Draw 50 posterior latent samples
Z_samples = model.sample_posterior_latent(X, n_draws=50)
print("Z_samples shape (n_draws, n_samples, k):", Z_samples.shape)

# Convert a latent draw to a likelihood sample (observable space)
Z0 = Z_samples[0]
X_mean, X_cov = model.likelihood(Z0)
print("likelihood mean shape:", X_mean.shape)

# Draw samples from likelihood (optional)
X_lik_samples = model.sample_likelihood(Z0, n_draws=10)
print("X_lik_samples shape:", X_lik_samples.shape)
print("Done")
