import numpy as np

from lib_ppca import PPCA


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
        assert np.allclose(params_bat["W"], params_full["W"], rtol=1e-5, atol=1e-7)
        assert np.allclose(params_bat["mu"], params_full["mu"], rtol=1e-5, atol=1e-7)
        assert np.allclose(
            params_bat["sig2"], params_full["sig2"], rtol=1e-5, atol=1e-7
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
