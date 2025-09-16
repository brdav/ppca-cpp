// MIT License
#include "ppca/ppca.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <string>

#include "ppca/detail/sampling.hpp"
#include "ppca/detail/utils.hpp"

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
  if (rtol_ <= 0) throw std::invalid_argument("rtol must be greater than zero");
  if (n_components_ <= 0)
    throw std::invalid_argument("n_components must be greater than zero");
  if (random_state) arma::arma_rng::set_seed(*random_state);
}

PPCA& PPCA::fit(const arma::mat& X) {
  n_features_in_ = X.n_rows;
  n_samples_ = X.n_cols;
  // Validate: at least one observed per sample and feature
  const arma::umat obs = detail::mask_finite(X);
  const arma::uvec counts_per_sample = arma::sum(obs, 0).t();
  const arma::uvec counts_per_feature = arma::sum(obs, 1);
  if (arma::any(counts_per_sample == 0u)) {
    throw std::invalid_argument(
        "Every sample must have at least one observed feature");
  }
  if (arma::any(counts_per_feature == 0u)) {
    throw std::invalid_argument(
        "Every feature must be observed in at least one sample");
  }

  // EM fit
  const Params p = em(X, obs);

  // Cache ML parameters
  W_ = p.W;
  mean_ = p.mu;
  noise_variance_ = p.sig2;

  if (rotate_to_orthogonal_) {
    compute_orthogonal_summary_and_rotate_W_();
  }

  return *this;
}

arma::vec PPCA::score_samples(const arma::mat& X) const {
  ensure_fitted();
  detail::check_dims(n_features_in_, X.n_rows, "score_samples");
  const arma::mat Xc = X.each_col() - mean_;
  const arma::umat obs = detail::mask_finite(X);
  return negative_log_likelihood(Xc, W_, noise_variance_, obs) * (-1.0);
}

arma::mat PPCA::get_covariance() const {
  ensure_fitted();
  const arma::mat cov =
      W_ * W_.t() + noise_variance_ * arma::eye(n_features_in_, n_features_in_);
  return cov;
}

arma::mat PPCA::get_precision() const {
  ensure_fitted();
  const arma::mat WT = W_.t();
  const arma::mat WT_W = WT * W_;
  arma::mat M =
      WT_W + noise_variance_ * arma::eye(n_components_, n_components_);
  arma::mat Minv_WT = arma::solve(M, WT, arma::solve_opts::likely_sympd);
  arma::mat prec = (1.0 / noise_variance_) *
                   (arma::eye(n_features_in_, n_features_in_) - W_ * Minv_WT);
  return arma::symmatu(prec);
}

PPCA::Params PPCA::get_params() const {
  ensure_fitted();
  return {W_, mean_, noise_variance_};
}

void PPCA::set_params(const arma::mat& W, const arma::vec& mu, double sig2) {
  detail::check_dims(n_components_, W.n_cols, "set_params");
  if (mu.n_rows != W.n_rows) {
    throw std::invalid_argument(
        "set_params: mu length must match number of features in W");
  }
  if (!(sig2 >= 0.0 && std::isfinite(sig2))) {
    throw std::invalid_argument("set_params: sig2 must be finite and >= 0");
  }

  W_ = W;
  mean_ = mu;
  noise_variance_ = sig2;
  n_features_in_ = W_.n_rows;

  // Invalidate cached summaries
  components_.reset();
  explained_variance_.reset();
  explained_variance_ratio_.reset();

  if (rotate_to_orthogonal_) {
    compute_orthogonal_summary_and_rotate_W_();
  }
}

void PPCA::compute_orthogonal_summary_and_rotate_W_() {
  // Compute orthonormal components_ and explained variances via thin SVD of W_
  arma::mat U, V;
  arma::vec s;
  arma::svd_econ(U, s, V, W_, "left");  // s is descending
  components_ = U;
  // Deterministic sign: enforce first element of each component non-negative
  for (arma::uword i = 0; i < components_.n_cols; ++i) {
    if (components_(arma::index_max(arma::abs(components_.col(i))), i) < 0) {
      components_.col(i) *= -1.0;
    }
  }
  explained_variance_ = arma::square(s) + noise_variance_;
  const double sum_s2 = arma::accu(arma::square(s));
  const double total_variance = sum_s2 + n_features_in_ * noise_variance_;
  explained_variance_ratio_ = explained_variance_ / total_variance;
  // re-scale W_ to have orthonormal components
  W_ = components_ * arma::diagmat(s);
}

void PPCA::ensure_fitted() const {
  if (W_.is_empty()) {
    throw std::runtime_error("Model not fitted");
  }
}

void PPCA::ensure_components_available() const {
  if (!rotate_to_orthogonal_) {
    throw std::logic_error(
        "Components and variance attributes only available when "
        "rotate_to_orthogonal=true at fit time");
  }
}

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
  ensure_components_available();
  return components_;
}

const arma::vec& PPCA::explained_variance() const {
  ensure_fitted();
  ensure_components_available();
  return explained_variance_;
}

const arma::vec& PPCA::explained_variance_ratio() const {
  ensure_fitted();
  ensure_components_available();
  return explained_variance_ratio_;
}

std::size_t PPCA::n_samples() const {
  ensure_fitted();
  return n_samples_;
}
std::size_t PPCA::n_features_in() const {
  ensure_fitted();
  return n_features_in_;
}
std::size_t PPCA::n_components() const {
  ensure_fitted();
  return n_components_;
}

}  // namespace ppca
