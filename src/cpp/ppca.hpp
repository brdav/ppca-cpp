// Copyright 2025 brdav

// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PPCA_HPP
#define PPCA_HPP

#include <armadillo>
#include <cstddef>
#include <functional>
#include <optional>
#include <stdexcept>
#include <tuple>

namespace ppca {

/**
 * @brief Implements Probabilistic Principal Component Analysis (PPCA).
 *
 * This class provides a C++ implementation of PPCA using the Armadillo library.
 * PPCA is a dimensionality reduction technique that provides a probabilistic
 * approach to PCA, allowing it to handle missing data effectively.
 *
 * @note Internally, data matrices are handled in a (features x samples)
 * orientation for efficiency with Armadillo's column-major storage.
 */
class PPCA {
 public:
  /**
   * @brief A structure to hold the core parameters of a fitted PPCA model.
   */
  struct Params {
    arma::mat components;   //!< Loadings matrix (n_features x n_components).
    arma::vec mean;         //!< Feature-wise mean vector (n_features).
    double noise_variance;  //!< Isotropic noise variance (sigma^2).
  };

  /**
   * @brief Constructs a PPCA model.
   *
   * @param n_components The number of latent components to estimate.
   * @param max_iter The maximum number of EM iterations to run.
   * @param min_iter The minimum number of EM iterations before checking for
   * convergence.
   * @param rtol Relative tolerance for the change in negative log-likelihood
   * to determine convergence.
   * @param rotate_to_orthogonal If true, the principal components are rotated
   * to be orthonormal after fitting. This enables the `explained_variance_`
   * and `explained_variance_ratio_` attributes.
   * @param batch_size The size of mini-batches for the EM algorithm. If 0,
   * full-batch EM is used.
   * @param random_state An optional seed for Armadillo's random number
   * generator to ensure reproducibility.
   */
  explicit PPCA(std::size_t n_components, std::size_t max_iter = 10000,
                std::size_t min_iter = 20, double rtol = 1e-8,
                bool rotate_to_orthogonal = true, std::size_t batch_size = 0,
                std::optional<unsigned int> random_state = std::nullopt);

  // Core API
  // ---------------------------------------------------------------------------

  /**
   * @brief Fits the PPCA model to the provided data.
   *
   * @param x_data The input data matrix (n_features x n_samples). Missing
   * values should be represented as NaNs.
   * @return A reference to the fitted model instance for chaining.
   */
  PPCA& fit(const arma::mat& x_data);

  /**
   * @brief Calculates the log-likelihood of each sample under the model.
   *
   * @param x_data The data matrix (n_features x n_samples).
   * @return A vector containing the log-likelihood for each sample.
   */
  arma::vec score_samples(const arma::mat& x_data) const;

  /**
   * @brief Calculates the average log-likelihood of the data.
   *
   * @param x_data The data matrix (n_features x n_samples).
   * @return The mean log-likelihood across all samples.
   */
  double score(const arma::mat& x_data) const {
    return arma::mean(score_samples(x_data));
  }

  // Model Properties
  // ---------------------------------------------------------------------------

  /**
   * @brief Computes the model's estimated covariance matrix.
   *
   * The covariance C is calculated as C = W * W^T + sigma^2 * I.
   *
   * @return The estimated covariance matrix (n_features x n_features).
   */
  arma::mat get_covariance() const;

  /**
   * @brief Computes the model's estimated precision matrix.
   *
   * The precision C^-1 is calculated efficiently using the Woodbury matrix
   * identity.
   *
   * @return The estimated precision matrix (n_features x n_features).
   */
  arma::mat get_precision() const;

  // Model Distributions
  // ---------------------------------------------------------------------------

  /**
   * @brief Computes the posterior distribution over the latent variables.
   *
   * This corresponds to p(Z | X), where Z are the latent variables and X is
   * the observed data.
   *
   * @param x_data The data matrix (n_features x n_samples), where NaNs
   * denote missing values.
   * @return A pair containing the mean (n_components x n_samples) and
   * covariance (n_components x n_components x n_samples) of the posterior.
   */
  std::pair<arma::mat, arma::cube> posterior_latent(
      const arma::mat& x_data) const;

  /**
   * @brief Computes the likelihood distribution of the data given latent vars.
   *
   * This corresponds to p(X | Z).
   *
   * @param z_latent The latent variables matrix (n_components x n_samples).
   * @return A pair containing the mean (n_features x n_samples) and
   * covariance (n_features x n_features x n_samples) of the likelihood.
   */
  std::pair<arma::mat, arma::cube> likelihood(const arma::mat& z_latent) const;

  /**
   * @brief Computes the predictive distribution for missing data entries.
   *
   * This corresponds to p(X_missing | X_observed). The returned mean matrix
   * contains both the observed values and the imputed means for missing values.
   * The covariance is non-zero only for the missing entries.
   *
   * @param x_data The data matrix (n_features x n_samples) with NaNs for
   * missing entries.
   * @return A pair containing the imputed data mean and the predictive
   * covariance.
   */
  std::pair<arma::mat, arma::cube> impute_missing(
      const arma::mat& x_data) const;

  /**
   * @brief Linear minimum mean-squared error (LMMSE) reconstruction from
   * latent.
   *
   * Returns the reconstructed observed-space data given latent variables Z,
   * applying the analytic LMMSE scaling that accounts for the isotropic noise.
   *
   * @param z_latent The latent variables matrix (n_components x n_samples).
   * @return Reconstructed data matrix (n_features x n_samples).
   */
  arma::mat lmmse_reconstruction(const arma::mat& z_latent) const;

  // Sampling Methods
  // ---------------------------------------------------------------------------

  /**
   * @brief Draws samples from the latent posterior distribution p(Z | X).
   *
   * @param x_data The data matrix (n_features x n_samples).
   * @param num_draws The number of samples to draw for each data point.
   * @return A cube of samples (n_components x n_samples x num_draws).
   */
  arma::cube sample_posterior_latent(const arma::mat& x_data,
                                     std::size_t num_draws = 1) const;

  /**
   * @brief Draws samples from the likelihood distribution p(X | Z).
   *
   * @param z_latent The latent variables matrix (n_components x n_samples).
   * @param num_draws The number of samples to draw.
   * @return A cube of samples (n_features x n_samples x num_draws).
   */
  arma::cube sample_likelihood(const arma::mat& z_latent,
                               std::size_t num_draws = 1) const;

  /**
   * @brief Draws samples to fill in missing entries from p(X_missing | X_obs).
   *
   * @param x_data The data matrix (n_features x n_samples) with NaNs.
   * @param num_draws The number of samples to draw.
   * @return A cube of imputed data samples (n_features x n_samples x
   * num_draws).
   */
  arma::cube sample_missing(const arma::mat& x_data,
                            std::size_t num_draws = 1) const;

  // Parameter Access
  // ---------------------------------------------------------------------------

  /** @brief Returns a copy of the model's core parameters. */
  Params get_params() const;

  /**
   * @brief Sets the model's core parameters.
   *
   * This allows for initializing a model with pre-existing parameters. If
   * `rotate_to_orthogonal` was set to true, this method will also update the
   * orthogonal component summaries.
   *
   * @param params The parameters struct containing all model parameters.
   * @throws std::invalid_argument on shape or value mismatch.
   */
  void set_params(const Params& params);

  // Fitted Attributes (Accessors)
  // ---------------------------------------------------------------------------

  /** @return The feature-wise mean vector (n_features). */
  const arma::vec& mean() const;

  /** @return The isotropic noise variance (sigma^2). */
  double noise_variance() const;

  /**
   * @brief Returns the orthonormal principal components.
   * @note Only available if `rotate_to_orthogonal` is true.
   * @return The components matrix (n_features x n_components).
   */
  const arma::mat& components() const;

  /**
   * @brief Returns the variance explained by each principal component.
   * @note Only available if `rotate_to_orthogonal` is true.
   * @return A vector of explained variances (n_components).
   */
  const arma::vec& explained_variance() const;

  /**
   * @brief Returns the fraction of variance explained by each component.
   * @note Only available if `rotate_to_orthogonal` is true.
   * @return A vector of explained variance ratios (n_components).
   */
  const arma::vec& explained_variance_ratio() const;

  // Dimension Accessors
  // ---------------------------------------------------------------------------

  /** @return The number of samples seen during fitting. */
  std::size_t n_samples() const;

  /** @return The number of features in the input data. */
  std::size_t n_features_in() const;

  /** @return The number of latent components. */
  std::size_t n_components() const;

 private:
  // Internal helper structs
  struct LatentMoments {
    arma::mat ez;
    arma::cube ezz;
  };
  struct UpdateResult {
    arma::mat components_new;
    double noise_variance_new;
    double neg_log_likelihood;
  };

  // EM algorithm implementation
  Params em(const arma::mat& x_data, const arma::umat& observations);
  Params em_complete_batch(const arma::mat& x_centered,
                           const arma::umat& observations, const Params& init);
  Params em_missing_batch(const arma::mat& x_centered,
                          const arma::umat& observations, const Params& init);
  void run_em_loop(const std::function<UpdateResult(const arma::mat&, double)>&
                       compute_update,
                   arma::mat& w, double& noise_var, double& prev_nll,
                   const char* failure_message) const;

  // Helper methods
  std::pair<std::size_t, std::size_t> compute_batching() const;
  Params initialize_parameters(const arma::mat& x_data,
                               const arma::umat& observations);
  LatentMoments latent_posterior_moments(const arma::mat& x_centered,
                                         const arma::mat& w,
                                         double noise_variance,
                                         const arma::umat& observations) const;
  arma::vec negative_log_likelihood(const arma::mat& x_centered,
                                    const arma::mat& w, double noise_variance,
                                    const arma::umat& observations) const;
  void compute_orthogonal_summary_and_rotate_components();

  // Pre-condition checks
  void ensure_fitted() const;
  void ensure_components_available() const;

  // --- Member Variables ---

  // Hyperparameters
  const std::size_t n_components_;
  const std::size_t max_iter_;
  const std::size_t min_iter_;
  const double rtol_;
  const bool rotate_to_orthogonal_;
  const std::size_t batch_size_;

  // Fitted model parameters
  arma::mat components_;
  arma::vec mean_;
  double noise_variance_ = 0.0;
  std::size_t n_samples_ = 0;
  std::size_t n_features_in_ = 0;

  // Optional cached summaries (if rotate_to_orthogonal_ is true)
  std::optional<arma::vec> explained_variance_;
  std::optional<arma::vec> explained_variance_ratio_;
};

}  // namespace ppca

#endif  // PPCA_HPP
