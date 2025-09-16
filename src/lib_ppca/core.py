from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ._version import __version__  # noqa: F401

try:
    from . import _bindings  # type: ignore
except Exception as e:  # pragma: no cover
    _bindings = None
    _import_error = str(e)
else:
    _import_error = ""


class PPCA:
    """High-level PPCA wrapper around the compiled C++ core.

    Parameters map directly to the C++ implementation. Inputs and outputs in
    this Python wrapper use the conventional scikit-learn orientation where
    samples are rows and features are columns.

    Args:
        n_components: Number of latent components (q > 0).
        max_iter: Maximum EM iterations.
        min_iter: Minimum EM iterations before early stopping is considered.
        rtol: Relative tolerance for convergence.
        rotate_to_orthogonal: If True, rotate loadings to an orthonormal basis
            after fitting and expose spectral summaries (components_, etc.).
        batch_size: Optional mini-batch size for EM; if None, use full batch.
        random_state: Optional RNG seed forwarded to the C++ backend.

    Raises:
        RuntimeError: If the compiled extension failed to import.
        ValueError: If ``batch_size`` is provided and not positive.
    """

    def __init__(
        self,
        n_components: int,
        max_iter: int = 10000,
        min_iter: int = 20,
        rtol: float = 1e-6,
        rotate_to_orthogonal: bool = True,
        batch_size: int | None = None,
        random_state: int | None = None,
    ):
        if _bindings is None:
            raise RuntimeError(f"PPCA extension module not built: {_import_error}")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive when provided")
        self._model = _bindings._CPP_PPCA(
            n_components,
            max_iter,
            min_iter,
            rtol,
            rotate_to_orthogonal,
            0 if batch_size is None else batch_size,
            random_state,
        )

    def fit(self, X: ArrayLike) -> "PPCA":
        """Fit the PPCA model.

        If ``batch_size`` was provided at construction time, a mini-batch EM
        loop is used internally (one EM iteration per batch per outer epoch)
        until convergence. Otherwise a full-batch EM / closed-form solver is
        used.

        Args:
            X: Input data of shape (n_samples, n_features). Missing entries
                should be encoded as NaN.

        Returns:
            PPCA: This instance for chaining.
        """
        X = np.asarray(X, dtype=float)
        self._model.fit(X.T)
        return self

    def score(self, X: ArrayLike) -> float:
        """Mean log-likelihood of the data under the model.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            float: Average per-sample log-likelihood.
        """
        X = np.asarray(X, dtype=float)
        return self._model.score(X.T)

    def score_samples(self, X: ArrayLike) -> NDArray[np.floating]:
        """Per-sample log-likelihoods.

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            ndarray: Shape (n_samples,) of log-likelihoods.
        """
        X = np.asarray(X, dtype=float)
        return np.squeeze(self._model.score_samples(X.T), axis=1)

    def get_covariance(self) -> NDArray[np.floating]:
        """Return the model covariance matrix.

        Returns:
            ndarray: Shape (n_features, n_features).
        """
        return self._model.get_covariance()

    def get_precision(self) -> NDArray[np.floating]:
        """Return the model precision (inverse covariance) matrix.

        Returns:
            ndarray: Shape (n_features, n_features).
        """
        return self._model.get_precision()

    def posterior_latent(
        self, X: ArrayLike
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Posterior over latent variables p(Z | X).

        Args:
            X: Data of shape (n_samples, n_features).

        Returns:
            tuple: (mZ, covZ)
                - mZ: Posterior means, shape (n_samples, n_components)
                - covZ: Posterior covariances per sample, shape
                  (n_samples, n_components, n_components)
        """
        X = np.asarray(X, dtype=float)
        mZ, covZ = self._model.posterior_latent(X.T)
        return mZ.T, covZ.transpose(2, 1, 0)

    def likelihood(
        self, Z: ArrayLike
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Likelihood / generative mapping p(X | Z).

        Args:
            Z: Latent variables of shape (n_samples, n_components).

        Returns:
            tuple: (mX, covX)
                - mX: Means, shape (n_samples, n_features)
                - covX: Covariances per sample, shape
                  (n_samples, n_features, n_features)
        """
        Z = np.asarray(Z, dtype=float)
        mX, covX = self._model.likelihood(Z.T)
        return mX.T, covX.transpose(2, 1, 0)

    def impute_missing(
        self, X: ArrayLike
    ) -> Tuple[NDArray[np.floating], NDArray[np.floating]]:
        """Conditional predictive distribution for missing entries.

        Args:
            X: Data with NaNs for missing values, shape (n_samples, n_features).

        Returns:
            tuple: (mX, covX) with the same shapes as in ``likelihood``.
        """
        X = np.asarray(X, dtype=float)
        mX, covX = self._model.impute_missing(X.T)
        return mX.T, covX.transpose(2, 1, 0)

    def sample_posterior_latent(
        self, X: ArrayLike, n_draws: int = 1
    ) -> NDArray[np.floating]:
        """Draw samples from p(Z | X).

        Args:
            X: Data of shape (n_samples, n_features).
            n_draws: Number of samples per observation.

        Returns:
            ndarray: Shape (n_draws, n_samples, n_components).
        """
        X = np.asarray(X, dtype=float)
        Z_tilde = self._model.sample_posterior_latent(X.T, n_draws)
        return Z_tilde.transpose(2, 1, 0)

    def sample_likelihood(self, Z: ArrayLike, n_draws: int = 1) -> NDArray[np.floating]:
        """Draw samples from p(X | Z).

        Args:
            Z: Latent variables of shape (n_samples, n_components).
            n_draws: Number of samples per latent instance.

        Returns:
            ndarray: Shape (n_draws, n_samples, n_features).
        """
        Z = np.asarray(Z, dtype=float)
        X_tilde = self._model.sample_likelihood(Z.T, n_draws)
        return X_tilde.transpose(2, 1, 0)

    def sample_missing(self, X: ArrayLike, n_draws: int = 1) -> NDArray[np.floating]:
        """Draw samples for missing entries p(X_miss | X_obs).

        Args:
            X: Data with NaNs for missing values, shape (n_samples, n_features).
            n_draws: Number of samples per observation.

        Returns:
            ndarray: Shape (n_draws, n_samples, n_features).
        """
        X = np.asarray(X, dtype=float)
        X_tilde = self._model.sample_missing(X.T, n_draws)
        return X_tilde.transpose(2, 1, 0)

    def lmmse_reconstruction(self, Z: ArrayLike) -> NDArray[np.floating]:
        """Linear MMSE reconstruction E[X | Z].

        Args:
            Z: Latent variables of shape (n_samples, n_components).

        Returns:
            ndarray: Reconstructed means, shape (n_samples, n_features).
        """
        Z = np.asarray(Z, dtype=float)
        X_hat = self._model.lmmse_reconstruction(Z.T)
        return X_hat.T

    def get_params(self) -> Dict[str, NDArray[np.floating]]:
        """Return current model parameters.

        Returns:
            dict: With keys
                - "W": Loadings in Python orientation (n_components, n_features)
                - "mu": Mean vector (n_features,)
                - "sig2": Noise variance (scalar stored as ndarray)
        """
        params = self._model.get_params()
        return {
            "W": np.asarray(params["W"]).T,
            "mu": np.squeeze(np.asarray(params["mu"]), axis=1),
            "sig2": np.asarray(params["sig2"]),
        }

    def set_params(self, params: Mapping[str, Any]) -> None:
        """Set model parameters from Python-orientation arrays.

        Expects keys "W", "mu", and "sig2". Shapes are in Python
        orientation: W is (n_components, n_features) and will be transposed for
        the backend; mu is (n_features,); sig2 is a scalar (float or ndarray).

        Args:
            params: Mapping with keys "W", "mu", "sig2".

        Raises:
            KeyError: If required keys are missing.
        """
        W = np.asarray(params["W"], dtype=float).T
        mu = np.asarray(params["mu"], dtype=float)
        sig2 = float(np.asarray(params["sig2"], dtype=float))
        self._model.set_params(W, mu, sig2)

    @property
    def components_(self) -> NDArray[np.floating]:
        """Orthogonal components (if available), shape (n_components, n_features)."""
        return np.asarray(self._model.components).T

    @property
    def mean_(self) -> NDArray[np.floating]:
        """Feature-wise mean, shape (n_features,)."""
        return np.squeeze(np.asarray(self._model.mean), axis=1)

    @property
    def noise_variance_(self) -> float | NDArray[np.floating]:
        """Isotropic noise variance (scalar)."""
        return np.asarray(self._model.noise_variance)

    @property
    def explained_variance_(self) -> NDArray[np.floating]:
        """Variance explained by each component, shape (n_components,)."""
        return np.squeeze(np.asarray(self._model.explained_variance), axis=1)

    @property
    def explained_variance_ratio_(self) -> NDArray[np.floating]:
        """Fraction of total variance explained by each component, shape (n_components,)."""
        return np.squeeze(np.asarray(self._model.explained_variance_ratio), axis=1)

    @property
    def n_samples_(self) -> int:
        """Number of samples seen during fit."""
        return self._model.n_samples

    @property
    def n_features_in_(self) -> int:
        """Number of features (columns)."""
        return self._model.n_features_in

    @property
    def n_components_(self) -> int:
        """Number of latent components."""
        return self._model.n_components
