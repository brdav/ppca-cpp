# Copyright 2025 brdav

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
from ppca import PPCA


def make_synthetic(n_samples=120, n_features=6, n_components=2, sig2=0.05, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_components, n_features))
    # Make components reasonably scaled and well-conditioned
    W = (W / np.linalg.norm(W, axis=1, keepdims=True)) * np.linspace(
        2.0, 1.2, n_components
    )[:, None]
    mu = rng.normal(size=(n_features,))
    Z = rng.normal(size=(n_samples, n_components))
    X_clean = Z @ W + mu
    X = X_clean + rng.normal(scale=np.sqrt(sig2), size=(n_samples, n_features))
    return X, W, mu, sig2


def test_impute_missing_observed_equals_input():
    X, _, _, _ = make_synthetic(seed=42)
    rng = np.random.default_rng(123)
    # Introduce about 25% missing uniformly at random
    mask = rng.random(X.shape) < 0.25
    X_missing = X.copy()
    X_missing[mask] = np.nan

    model = PPCA(n_components=2, rtol=1e-8, random_state=7)
    model.fit(X_missing)

    mX, covX = model.impute_missing(X_missing)

    # For observed entries, conditional mean equals the observation
    observed = ~np.isnan(X_missing)
    assert np.allclose(mX[observed], X_missing[observed], atol=1e-10, rtol=0)

    # For observed-observed blocks, conditional covariance is ~0
    # Check a few random rows to keep test cheap
    rows = np.arange(X.shape[0])
    rng.shuffle(rows)
    for i in rows[:10]:
        obs_idx = np.flatnonzero(observed[i])
        if obs_idx.size > 0:
            S = covX[i][np.ix_(obs_idx, obs_idx)]
            assert np.allclose(S, 0.0, atol=1e-10, rtol=0)


def test_impute_all_missing_row_equals_prior():
    X, _, _, _ = make_synthetic(seed=3)
    # Fit on complete data
    model = PPCA(n_components=2, rtol=1e-8, random_state=5)
    model.fit(X)

    # Query with one fully-missing row and some partial rows
    Xq = X.copy()
    Xq[0, :] = np.nan  # fully missing
    Xq[1, :2] = np.nan

    mX, covX = model.impute_missing(Xq)

    # For fully-missing row, conditional equals prior over X: mean=mu, cov=Sigma
    Sigma = model.get_covariance()
    assert np.allclose(mX[0], model.mean_, rtol=1e-6, atol=1e-8)
    assert np.allclose(covX[0], Sigma, rtol=1e-5, atol=1e-7)


def test_posterior_sampling_matches_moments_single_sample():
    # Use a single complete sample to validate posterior sampling moments
    X, _, _, _ = make_synthetic(seed=11)

    model = PPCA(n_components=2, rtol=1e-9, random_state=11)
    model.fit(X)

    mZ, covZ = model.posterior_latent(X)
    # Focus on one sample with stable stats
    i = 3
    draws = 1500
    Zs = model.sample_posterior_latent(X, n_draws=draws)
    Zs = Zs[:, i, :]  # (n_draws, n_components)

    emp_mean = Zs.mean(axis=0)
    emp_cov = np.cov(Zs, rowvar=False, bias=False)

    assert np.allclose(emp_mean, mZ[i], rtol=0.15, atol=0.05)
    assert np.allclose(emp_cov, covZ[i], rtol=0.25, atol=0.05)


def test_missing_sampling_consistency():
    X, _, _, _ = make_synthetic(seed=21)
    # Make at least one row with multiple missings
    i = 4
    X[i, [0, 2, 5]] = np.nan

    model = PPCA(n_components=2, rtol=1e-8, random_state=22)
    model.fit(X)

    mX, covX = model.impute_missing(X)
    # Sample conditional over X given observed, for the chosen row
    draws = 800
    Xs = model.sample_missing(X, n_draws=draws)  # (n_draws, n_samples, n_features)

    obs_mask = ~np.isnan(X[i])
    miss_mask = np.isnan(X[i])

    # Observed dims should be constant equal to the observation
    assert np.allclose(Xs[:, i, obs_mask], X[i, obs_mask][None, :], atol=1e-12, rtol=0)

    # For missing dims, sample mean and covariance should match analytic
    emp_mean = Xs[:, i, miss_mask].mean(axis=0)
    emp_cov = np.cov(Xs[:, i, miss_mask], rowvar=False, bias=False)

    assert np.allclose(emp_mean, mX[i, miss_mask], rtol=0.2, atol=0.05)
    # Compare the missing-missing block of covariance
    S = covX[i][np.ix_(miss_mask, miss_mask)]
    assert np.allclose(emp_cov, S, rtol=0.3, atol=0.05)
