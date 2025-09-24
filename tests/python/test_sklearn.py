# Copyright 2025 brdav

# Use of this source code is governed by an MIT-style
# license that can be found in the LICENSE file or at
# https://opensource.org/licenses/MIT.

import numpy as np
from ppca import PPCA
from sklearn.decomposition import PCA as PCA_sklearn


def test_sklearn():
    from sklearn.datasets import load_iris

    X = load_iris().data

    pca_sklearn = PCA_sklearn(n_components=2)
    Z_sklearn = pca_sklearn.fit_transform(X)
    X_hat_sklearn = pca_sklearn.inverse_transform(Z_sklearn)

    ppca = PPCA(n_components=2, rtol=1e-16, rotate_to_orthogonal=True)
    ppca.fit(X)

    def transform(X, mean, components):
        return (X - mean[None, :]) @ components.T

    def inverse_transform(X, mean, components):
        return (X @ components) + mean[None, :]

    # For comparison with sklearn, we need to normalize components
    norm_components = ppca.components_ / np.linalg.norm(
        ppca.components_, axis=1, keepdims=True
    )
    Z = transform(X, ppca.mean_, norm_components)
    X_hat = inverse_transform(Z, ppca.mean_, norm_components)

    # Check simple attributes
    assert ppca.n_components_ == pca_sklearn.n_components_
    assert ppca.n_samples_ == pca_sklearn.n_samples_
    assert ppca.n_features_in_ == pca_sklearn.n_features_in_

    # Check transformations and reconstructions
    assert np.allclose(Z, Z_sklearn, rtol=1e-6, atol=1e-8)
    assert np.allclose(X_hat, X_hat_sklearn, rtol=1e-6, atol=1e-8)

    # Apply Bessel-correction to components and noise variance for parity with sklearn
    params = ppca.get_params()
    params["components"] *= np.sqrt(ppca.n_samples_ / (ppca.n_samples_ - 1))
    params["noise_variance"] *= ppca.n_samples_ / (ppca.n_samples_ - 1)
    ppca.set_params(params)

    # Check components and attributes
    assert np.allclose(norm_components, pca_sklearn.components_, rtol=1e-6, atol=1e-8)
    assert np.allclose(ppca.mean_, pca_sklearn.mean_, rtol=1e-6, atol=1e-8)
    assert np.allclose(
        ppca.noise_variance_,
        pca_sklearn.noise_variance_,
        rtol=1e-6,
        atol=1e-8,
    )
    assert np.allclose(
        ppca.explained_variance_,
        pca_sklearn.explained_variance_,
        rtol=1e-6,
        atol=1e-8,
    )
    assert np.allclose(
        ppca.explained_variance_ratio_,
        pca_sklearn.explained_variance_ratio_,
        rtol=1e-6,
        atol=1e-8,
    )

    # Check remaining methods, allow a bit looser tolerance
    assert np.allclose(
        ppca.score_samples(X), pca_sklearn.score_samples(X), rtol=1e-5, atol=1e-6
    )
    assert np.allclose(ppca.score(X), pca_sklearn.score(X), rtol=1e-5, atol=1e-6)
    assert np.allclose(
        ppca.get_covariance(), pca_sklearn.get_covariance(), rtol=1e-5, atol=1e-6
    )
    assert np.allclose(
        ppca.get_precision(), pca_sklearn.get_precision(), rtol=1e-5, atol=1e-6
    )
