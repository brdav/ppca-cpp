// Copyright 2025 brdav

// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "ppca.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>

#include "utils.hpp"

namespace ppca {

PPCA::PPCA(std::size_t n_components, std::size_t max_iter, std::size_t min_iter,
           double rtol, bool rotate_to_orthogonal, std::size_t batch_size,
           std::optional<unsigned int> random_state)
    : n_components_(n_components),
      max_iter_(max_iter),
      min_iter_(min_iter),
      rtol_(rtol),
      rotate_to_orthogonal_(rotate_to_orthogonal),
      batch_size_(batch_size) {
  if (rtol_ <= 0.0) {
    throw std::invalid_argument("rtol must be greater than zero.");
  }
  if (n_components_ == 0) {
    throw std::invalid_argument("n_components must be greater than zero.");
  }
  if (random_state) {
    arma::arma_rng::set_seed(*random_state);
  }
}

PPCA& PPCA::fit(const arma::mat& x_data) {
  n_features_in_ = x_data.n_rows;
  n_samples_ = x_data.n_cols;

  // Validate that every sample and feature has at least one observed value.
  const arma::umat observations = ppca::utils::mask_finite(x_data);
  const arma::uvec counts_per_sample = arma::sum(observations, 0).t();
  if (arma::any(counts_per_sample == 0u)) {
    throw std::invalid_argument(
        "Every sample must have at least one observed feature.");
  }
  const arma::uvec counts_per_feature = arma::sum(observations, 1);
  if (arma::any(counts_per_feature == 0u)) {
    throw std::invalid_argument(
        "Every feature must be observed in at least one sample.");
  }

  // Run the Expectation-Maximization algorithm.
  const Params learned_params = em(x_data, observations);

  // Cache the learned model parameters.
  components_ = learned_params.components;
  mean_ = learned_params.mean;
  noise_variance_ = learned_params.noise_variance;

  if (rotate_to_orthogonal_) {
    compute_orthogonal_summary_and_rotate_components();
  }

  return *this;
}

arma::vec PPCA::score_samples(const arma::mat& x_data) const {
  ensure_fitted();
  ppca::utils::check_dims(n_features_in_, x_data.n_rows, "score_samples");
  const arma::mat x_centered = x_data.each_col() - mean_;
  const arma::umat observations = ppca::utils::mask_finite(x_data);
  // Return log-likelihood, not negative log-likelihood.
  return negative_log_likelihood(x_centered, components_, noise_variance_,
                                 observations) *
         (-1.0);
}

arma::mat PPCA::get_covariance() const {
  ensure_fitted();
  const arma::mat covariance =
      components_ * components_.t() +
      noise_variance_ * arma::eye(n_features_in_, n_features_in_);
  return covariance;
}

arma::mat PPCA::get_precision() const {
  ensure_fitted();
  const arma::mat w_transpose = components_.t();
  const arma::mat w_transpose_w = w_transpose * components_;
  arma::mat m_matrix =
      w_transpose_w + noise_variance_ * arma::eye(n_components_, n_components_);
  arma::mat m_inv_w_transpose =
      arma::solve(m_matrix, w_transpose, arma::solve_opts::likely_sympd);
  arma::mat precision_matrix =
      (1.0 / noise_variance_) * (arma::eye(n_features_in_, n_features_in_) -
                                 components_ * m_inv_w_transpose);
  return arma::symmatu(precision_matrix);
}

PPCA::Params PPCA::get_params() const {
  ensure_fitted();
  return {components_, mean_, noise_variance_};
}

void PPCA::set_params(const Params& params) {
  ppca::utils::check_dims(n_components_, params.components.n_cols,
                          "set_params");
  if (params.mean.n_rows != params.components.n_rows) {
    throw std::invalid_argument(
        "set_params: mean vector length must match number of features in "
        "components.");
  }
  if (!(params.noise_variance >= 0.0 && std::isfinite(params.noise_variance))) {
    throw std::invalid_argument(
        "set_params: noise_variance must be finite and non-negative.");
  }

  components_ = params.components;
  mean_ = params.mean;
  noise_variance_ = params.noise_variance;
  n_features_in_ = components_.n_rows;

  // Invalidate cached summaries that depend on the old parameters.
  explained_variance_.reset();
  explained_variance_ratio_.reset();

  if (rotate_to_orthogonal_) {
    compute_orthogonal_summary_and_rotate_components();
  }
}

// Private methods
// -----------------------------------------------------------------------------

void PPCA::compute_orthogonal_summary_and_rotate_components() {
  // Decompose components_ via SVD to find orthonormal components and explained
  // variance.
  arma::mat U;
  arma::vec s;
  arma::mat V;
  arma::svd_econ(U, s, V, components_, "left");  // s is sorted descending.

  // Enforce a deterministic sign for components for consistent output.
  for (arma::uword i = 0; i < U.n_cols; ++i) {
    const arma::uword idx = arma::index_max(arma::abs(U.col(i)));
    if (U(idx, i) < 0.0) {
      U.col(i) *= -1.0;
    }
  }

  // Re-compose components_ based on the orthonormal components and singular
  // values.
  components_ = U * arma::diagmat(s);

  explained_variance_ = arma::square(s) + noise_variance_;
  const double sum_s_sq = arma::accu(arma::square(s));
  const double total_variance = sum_s_sq + n_features_in_ * noise_variance_;
  arma::vec& explained_var = *explained_variance_;
  explained_variance_ratio_ = explained_var / total_variance;
}

void PPCA::ensure_fitted() const {
  if (components_.is_empty()) {
    throw std::runtime_error(
        "This PPCA instance is not fitted yet. Call fit() before using this "
        "method.");
  }
}

void PPCA::ensure_components_available() const {
  if (!rotate_to_orthogonal_) {
    throw std::logic_error(
        "Components and variance attributes are only available when "
        "rotate_to_orthogonal=true.");
  }
}

// Public accessors
// -----------------------------------------------------------------------------

const arma::vec& PPCA::mean() const {
  ensure_fitted();
  return mean_;
}

double PPCA::noise_variance() const {
  ensure_fitted();
  return noise_variance_;
}

const arma::mat& PPCA::components() const {
  ensure_fitted();
  return components_;
}

const arma::vec& PPCA::explained_variance() const {
  ensure_fitted();
  ensure_components_available();
  return *explained_variance_;
}

const arma::vec& PPCA::explained_variance_ratio() const {
  ensure_fitted();
  ensure_components_available();
  return *explained_variance_ratio_;
}

std::size_t PPCA::n_samples() const {
  ensure_fitted();
  return n_samples_;
}
std::size_t PPCA::n_features_in() const {
  ensure_fitted();
  return n_features_in_;
}
std::size_t PPCA::n_components() const { return n_components_; }

}  // namespace ppca
