// MIT License
// C++ implementation of Probabilistic PCA (PPCA) using Armadillo.
#pragma once
#include <armadillo>
#include <functional>
#include <optional>
#include <stdexcept>
#include <tuple>

namespace ppca {

/**
 * @brief Probabilistic PCA (PPCA) model implemented with Armadillo.
 *
 * The internal representation uses (n_features x n_samples) orientation. The
 * Python wrapper handles transposes to present the usual (n_samples x
 * n_features) interface.
 */
class PPCA {
 public:
  /**
   * @brief Construct a PPCA model with given hyperparameters.
   * @param n_components Number of latent components (q > 0).
   * @param max_iter Maximum number of EM iterations.
   * @param min_iter Minimum number of EM iterations before convergence checks.
   * @param rtol Relative tolerance for convergence (change in NLL).
   * @param rotate_to_orthogonal If true, rotate to orthonormal components
   *        after fit and expose spectral summaries (components_, etc.).
   * @param batch_size Mini-batch size for EM (0 = full batch).
   * @param random_state Optional RNG seed.
   */
  PPCA(std::size_t n_components, std::size_t max_iter = 10000,
       std::size_t min_iter = 20, double rtol = 1e-8,
       bool rotate_to_orthogonal = true, std::size_t batch_size = 0,
       std::optional<unsigned int> random_state = std::nullopt);

  /**
   * @brief Fit the model to data with possible missing values (NaN).
   * @param X Data matrix (n_features x n_samples).
   * @return Reference to this instance (for chaining).
   */
  PPCA &fit(const arma::mat &X);
  /**
   * @brief Per-sample log-likelihoods under the current model.
   * @param X Data (n_features x n_samples).
   * @return Vector (n_samples) of log-likelihoods.
   */
  arma::vec score_samples(const arma::mat &X) const;
  /**
   * @brief Mean log-likelihood over samples.
   */
  double score(const arma::mat &X) const {
    return arma::mean(score_samples(X));
  }
  /**
   * @brief Covariance matrix C = W W^T + sig2 I.
   * @return (n_features x n_features) SPD matrix.
   */
  arma::mat get_covariance() const;
  /**
   * @brief Precision matrix C^{-1} computed via Woodbury identity.
   * @return (n_features x n_features) symmetric matrix.
   */
  arma::mat get_precision() const;

  /**
   * @brief Posterior over latent variables p(Z | X).
   * @param X (n_features x n_samples), NaNs denote missing.
   * @return Pair (mZ, covZ): mZ is (n_components x n_samples), covZ cube is
   *         (n_components x n_components x n_samples).
   */
  std::pair<arma::mat, arma::cube> posterior_latent(const arma::mat &X) const;

  /**
   * @brief Likelihood p(X | Z) for given latent variables.
   * @param Z (n_components x n_samples).
   * @return Pair (mX, covX): mX is (n_features x n_samples), covX cube is
   *         (n_features x n_features x n_samples).
   */
  std::pair<arma::mat, arma::cube> likelihood(const arma::mat &Z) const;

  /**
   * @brief Conditional predictive distribution for missing entries p(X |
   * X_obs).
   * @param X (n_features x n_samples) with NaNs for missing values.
   * @return Pair (mX, covX) with the same shapes as in likelihood().
   */
  std::pair<arma::mat, arma::cube> impute_missing(const arma::mat &X) const;

  /**
   * @brief Draw samples from posterior p(Z | X).
   * @param X (n_features x n_samples).
   * @param n_draws Number of samples per column.
   * @return Cube (n_components x n_samples x n_draws).
   */
  arma::cube sample_posterior_latent(const arma::mat &X,
                                     size_t n_draws = 1) const;
  /**
   * @brief Draw samples from likelihood p(X | Z).
   * @param Z (n_components x n_samples).
   * @param n_draws Number of samples per column.
   * @return Cube (n_features x n_samples x n_draws).
   */
  arma::cube sample_likelihood(const arma::mat &Z, size_t n_draws = 1) const;
  /**
   * @brief Draw samples for missing entries p(X | X_obs).
   * @param X (n_features x n_samples) with NaNs.
   * @param n_draws Number of samples per column.
   * @return Cube (n_features x n_samples x n_draws).
   */
  arma::cube sample_missing(const arma::mat &X, size_t n_draws = 1) const;

  /**
   * @brief Linear MMSE reconstruction E[X | Z].
   * @param Z (n_components x n_samples).
   * @return (n_features x n_samples) means.
   */
  arma::mat lmmse_reconstruction(const arma::mat &Z) const;

  /**
   * @brief Core ML parameters.
   */
  struct Params {
    arma::mat W;   //!< loadings (n_features x n_components)
    arma::vec mu;  //!< mean (n_features)
    double sig2;   //!< isotropic noise variance
  };
  /**
   * @brief Get a copy of current parameters.
   */
  Params get_params() const;
  /**
   * @brief Set core ML parameters. Optionally updates orthogonal summaries.
   * @param W Loadings (n_features x n_components).
   * @param mu Mean vector (n_features).
   * @param sig2 Noise variance (>= 0).
   * @throws std::invalid_argument on shape/value mismatch.
   */
  void set_params(const arma::mat &W, const arma::vec &mu, double sig2);

  /** @return Orthonormal components (n_features x n_components). */
  const arma::mat &components() const;
  /** @return Explained variance per component (n_components). */
  const arma::vec &explained_variance() const;
  /** @return Explained variance ratio per component (n_components). */
  const arma::vec &explained_variance_ratio() const;
  /** @return Feature-wise mean (n_features). */
  const arma::vec &mean() const;
  /** @return Isotropic noise variance. */
  double noise_variance() const;
  /** @return Number of samples seen during fit. */
  std::size_t n_samples() const;
  /** @return Number of input features. */
  std::size_t n_features_in() const;
  /** @return Number of latent components. */
  std::size_t n_components() const;

 private:
  struct LatentMoments {
    arma::mat Ez;
    arma::cube Ezz;
  };
  struct UpdateResult {
    arma::mat W_new;
    double noise_var_new;
    double nll;
  };

  Params em(const arma::mat &X, const arma::umat &obs);
  Params em_complete_batch(const arma::mat &Xc, const arma::umat &obs,
                           const Params &init);
  Params em_missing_batch(const arma::mat &Xc, const arma::umat &obs,
                          const Params &init);
  void run_em_loop(
      std::function<UpdateResult(const arma::mat &, double)> compute_update,
      arma::mat &W, double &noise_var, double &prev_nll,
      const char *fail_msg) const;
  std::pair<std::size_t, std::size_t> compute_batching() const;
  Params initialize_parameters(const arma::mat &X, const arma::umat &obs);

  LatentMoments latent_posterior_moments(const arma::mat &Xc,
                                         const arma::mat &W, double noise_var,
                                         const arma::umat &obs) const;
  arma::vec negative_log_likelihood(const arma::mat &Xc, const arma::mat &W,
                                    double noise_var,
                                    const arma::umat &obs) const;

  /** @brief Throw if model is not yet fitted. */
  void ensure_fitted() const;
  /** @brief Throw if orthogonal summaries are unavailable by configuration. */
  void ensure_components_available() const;
  /**
   * @brief Compute orthonormal components and variance summaries from current
   *        W_ and noise_variance_, then rotate W_ to components_ * diag(s).
   *        Requires n_samples_ and n_features_in_ to be set.
   */
  void compute_orthogonal_summary_and_rotate_W_();

  std::size_t max_iter_;
  std::size_t min_iter_;
  double rtol_;
  bool rotate_to_orthogonal_;
  std::size_t batch_size_;

  // Mirroring scikit-learn attributes
  arma::mat components_;
  arma::vec mean_;
  double noise_variance_;
  arma::vec explained_variance_;
  arma::vec explained_variance_ratio_;
  std::size_t n_components_;
  std::size_t n_samples_;
  std::size_t n_features_in_;

  // Extra cached parameters
  arma::mat W_;
};

}  // namespace ppca
