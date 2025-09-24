# PPCA (Probabilistic PCA) — Reference

This document describes the `PPCA` class provided by this library: its purpose, key attributes, and public methods with brief usage examples.

## Overview

Probabilistic Principal Component Analysis (PPCA) models observed data x as a linear combination of latent variables z with isotropic Gaussian noise:

x = W z + mu + noise,  noise ~ N(0, sigma^2 I)

The `PPCA` implementation in this package exposes a scikit-learn-like interface and additional methods for dealing with missing data, posterior inference, and imputation.

## Constructor

PPCA(...)

Keyword arguments:

- `n_components` (int): Number of latent dimensions (k).
- `max_iter` (int): Maximum number of EM iterations.
- `min_iter` (int): Minimum number of EM iterations.
- `rtol` (float): Relative tolerance for EM convergence.
- `rotate_to_orthogonal` (bool): If true, rotate components to an orthonormal basis after fitting.
- `batch_size` (int | None): If provided, fit will run in online/mini-batch mode (reduced memory footprint).
- `random_state` (int | None): Random seed for reproducibility.

## Attributes

After fitting, the model exposes several attributes (naming follows scikit-learn conventions where possible):

- `components_` (array, shape (k, n_features)) — Principal axes (W). Need unit normalization for direct comparison to sklearn PCA components.
- `mean_` (array, shape (n_features,)) — Data mean used for centering.
- `noise_variance_` (float or array) — Estimated isotropic noise variance (sigma^2).
- `explained_variance_` (array, shape (k,)) — Variance explained by each latent dimension.
- `explained_variance_ratio_` (array, shape (k,)) — Fraction of total variance explained per component.
- `n_samples_` (int) — Number of samples used to fit the model.
- `n_features_in_` (int) — Number of features.
- `n_components_` (int) - Number of components.

## Key methods

The implementation provides methods for training, scoring, posterior inference, and handling missing values.

- `fit(X)`
  - Fit the PPCA model to data `X`. `X` may contain `np.nan` to indicate missing values. Returns `self`.
  - Side effects: populates attributes listed above.

- `get_params()` / `set_params(params)`
  - `get_params()` returns a serializable parameter dict (components, mean, noise_variance, ...).
  - `set_params(params)` sets model parameters from a dict.

- `score(X)`
  - Mean log-likelihood of the data under the model (comparable to sklearn's `score`).

- `score_samples(X)`
  - Per-sample log-likelihoods.

- `get_covariance()` / `get_precision()`
  - Model covariance and precision matrices.

- `posterior_latent(X_obs)`
  - Compute the posterior mean and covariance of the latent variables given observed data `X_obs` (which may contain `np.nan`).
  - Returns `(mZ, covZ)` where `mZ` has shape `(n_samples, k)` and `covZ` has shape `(n_samples, k, k)`.

- `sample_posterior_latent(X_obs, n_draws=...)`
  - Draw samples from the posterior over latents. Returns array shape `(n_draws, n_samples, k)`.

- `impute_missing(X_obs)`
  - Compute the posterior predictive mean and covariance for missing entries `p(X_missing | X_obs)`.
  - Returns `(X_imputed_mean, X_imputed_cov)`; `X_imputed_mean` has shape `(n_samples, n_features)` with observed entries unchanged and missing entries replaced by their posterior means.

- `sample_missing(X_obs, n_draws=...)`
  - Multiple-imputation style sampling of missing entries. Returns `(n_draws, n_samples, n_features)`.

- `likelihood(Z)`
  - Given latent codes `Z` (shape `(n_samples, k)`), compute expected observation mean and covariance under the likelihood. Returns `(mX, covX)`.

- `sample_likelihood(Z, n_draws=...)`
  - Draw samples from the likelihood. Returns `(n_draws, n_samples, n_features)`.

- `lmmse_reconstruction(Z)`
  - Compute the orthogonal linear minimum mean square error reconstruction for the noisy model. Returns reconstructed `X_hat`.

## Notes and compatibility

- The `PPCA` implementation is intentionally similar to scikit-learn's `PCA` but follows maximum likelihood estimates (no Bessel correction). To directly compare values (e.g., explained variance) with scikit-learn's PCA you need to scale parameters by `sqrt(n / (n - 1))`:

```python
# after .fit()
params = model.get_params()
params["components"] *= np.sqrt(model.n_samples_ / (model.n_samples_ - 1))
params["noise_variance"] *= model.n_samples_ / (model.n_samples_ - 1)
model.set_params(params)
```

- Most public methods accept arrays with missing entries encoded as `np.nan` and will handle them appropriately.

## Minimal usage example

```python
import numpy as np
from ppca import PPCA

# X: (n_samples, n_features), may contain np.nan
X = np.random.randn(100, 5)
model = PPCA(n_components=2)
model.fit(X)

# latent posterior mean and covariance
mZ, covZ = model.posterior_latent(X)

# impute missing values (mean imputation)
X_imputed, X_imputed_cov = model.impute_missing(X)

# sample multiple imputations
X_samples = model.sample_missing(X, n_draws=10)

# export parameters
params = model.get_params()

# set parameters back
model.set_params(params)
```

## Where to find more

The `examples/` folder contains short, focused examples demonstrating common PPCA workflows.
