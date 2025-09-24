# Copyright 2025 brdav

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
from ppca import PPCA


def make_synthetic(n_samples=200, n_features=8, n_components=3, sig2=0.02, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(size=(n_components, n_features))
    W = (W / np.linalg.norm(W, axis=1, keepdims=True)) * np.linspace(
        2.0, 1.0, n_components
    )[:, None]
    mu = rng.normal(size=(n_features,))
    Z = rng.normal(size=(n_samples, n_components))
    X = Z @ W + mu + rng.normal(scale=np.sqrt(sig2), size=(n_samples, n_features))
    return X


def test_batched_vs_full_equivalence():
    X = make_synthetic(seed=123)

    # Fit full-batch
    full = PPCA(n_components=3, rtol=1e-9, rotate_to_orthogonal=True, random_state=0)
    full.fit(X)
    params_full = full.get_params()

    # Fit mini-batch with a few different batch sizes and seeds
    for bs in (16, 32, 64):
        bat = PPCA(
            n_components=3,
            rtol=1e-9,
            rotate_to_orthogonal=True,
            batch_size=bs,
            random_state=0,
        )
        bat.fit(X)

        params_bat = bat.get_params()

        # Check model parameters
        assert np.allclose(
            params_bat["components"], params_full["components"], rtol=1e-5, atol=1e-7
        )
        assert np.allclose(
            params_bat["mean"], params_full["mean"], rtol=1e-5, atol=1e-7
        )
        assert np.allclose(
            params_bat["noise_variance"],
            params_full["noise_variance"],
            rtol=1e-5,
            atol=1e-7,
        )

        # Spectral summaries should agree
        assert np.allclose(
            full.explained_variance_, bat.explained_variance_, rtol=1e-5, atol=1e-7
        )
        assert np.allclose(
            full.explained_variance_ratio_,
            bat.explained_variance_ratio_,
            rtol=1e-5,
            atol=1e-7,
        )

        # Mean should match closely
        assert np.allclose(full.mean_, bat.mean_, rtol=1e-6, atol=1e-8)

        # Scoring should be equivalent
        assert np.allclose(full.score(X), bat.score(X), rtol=1e-6, atol=1e-8)
        assert np.allclose(
            full.score_samples(X), bat.score_samples(X), rtol=1e-6, atol=1e-8
        )


def test_batched_vs_full_equivalence_missing():
    X = make_synthetic(seed=123)
    X[::10, ::3] = np.nan  # introduce some missing values

    # Fit full-batch
    full = PPCA(n_components=3, rtol=1e-9, rotate_to_orthogonal=True, random_state=1)
    full.fit(X)
    params_full = full.get_params()

    # Fit mini-batch with a few different batch sizes and seeds
    for bs in (16, 32, 64):
        bat = PPCA(
            n_components=3,
            rtol=1e-9,
            rotate_to_orthogonal=True,
            batch_size=bs,
            random_state=1,
        )
        bat.fit(X)

        params_bat = bat.get_params()

        # Check model parameters
        assert np.allclose(
            params_bat["components"], params_full["components"], rtol=1e-5, atol=1e-7
        )
        assert np.allclose(
            params_bat["mean"], params_full["mean"], rtol=1e-5, atol=1e-7
        )
        assert np.allclose(
            params_bat["noise_variance"],
            params_full["noise_variance"],
            rtol=1e-5,
            atol=1e-7,
        )

        # Spectral summaries should agree
        assert np.allclose(
            full.explained_variance_, bat.explained_variance_, rtol=1e-5, atol=1e-7
        )
        assert np.allclose(
            full.explained_variance_ratio_,
            bat.explained_variance_ratio_,
            rtol=1e-5,
            atol=1e-7,
        )

        # Mean should match closely
        assert np.allclose(full.mean_, bat.mean_, rtol=1e-6, atol=1e-8)

        # Scoring should be equivalent
        assert np.allclose(full.score(X), bat.score(X), rtol=1e-6, atol=1e-8)
        assert np.allclose(
            full.score_samples(X), bat.score_samples(X), rtol=1e-6, atol=1e-8
        )
