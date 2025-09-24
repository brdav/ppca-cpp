// Copyright 2025 brdav

// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <stdexcept>
#include <utility>

#include "ppca.hpp"

namespace ppca {

PPCA::Params PPCA::em(const arma::mat& x_data, const arma::umat& observations) {
  Params initial_params = initialize_parameters(x_data, observations);
  const arma::mat x_centered = x_data.each_col() - initial_params.mean;
  const bool any_missing = (arma::accu(observations) != observations.n_elem);

  if (!any_missing) {
    return em_complete_batch(x_centered, observations, initial_params);
  } else {
    return em_missing_batch(x_centered, observations, initial_params);
  }
}

void PPCA::run_em_loop(
    const std::function<UpdateResult(const arma::mat&, double)>& compute_update,
    arma::mat& w, double& noise_var, double& prev_nll,
    const char* failure_message) const {
  for (std::size_t iter = 0; iter < max_iter_; ++iter) {
    UpdateResult result = compute_update(w, noise_var);

    bool converged = false;
    if (iter >= min_iter_) {
      const double relative_change =
          std::abs(result.neg_log_likelihood - prev_nll) /
          (std::abs(prev_nll) + 1e-12);
      if (relative_change < rtol_) {
        converged = true;
      }
    }

    w = std::move(result.components_new);
    noise_var = result.noise_variance_new;
    prev_nll = result.neg_log_likelihood;

    if (converged) {
      return;  // Converged successfully.
    }
  }

  // Loop finished without converging within max_iter_.
  throw std::runtime_error(failure_message);
}

std::pair<std::size_t, std::size_t> PPCA::compute_batching() const {
  if (batch_size_ == 0) {
    return {1, static_cast<std::size_t>(n_samples_)};
  }
  const std::size_t n_batches =
      (static_cast<std::size_t>(n_samples_) + batch_size_ - 1u) / batch_size_;
  return {n_batches, batch_size_};
}

PPCA::Params PPCA::initialize_parameters(const arma::mat& x_data,
                                         const arma::umat& observations) {
  // Compute mean of observed values for each feature.
  arma::vec mu(n_features_in_, arma::fill::zeros);
  const arma::vec counts =
      arma::conv_to<arma::vec>::from(arma::sum(observations, 1));
  for (arma::uword i = 0; i < n_features_in_; ++i) {
    double accumulator = 0.0;
    for (arma::uword j = 0; j < n_samples_; ++j) {
      if (observations(i, j)) {
        accumulator += x_data(i, j);
      }
    }
    mu(i) = (counts(i) > 0.0) ? accumulator / counts(i) : 0.0;
  }

  // Compute variance of observed values for each feature.
  arma::vec feature_var(n_features_in_, arma::fill::zeros);
  for (arma::uword i = 0; i < n_features_in_; ++i) {
    double accumulator_sq = 0.0;
    const double count = counts(i);
    for (arma::uword j = 0; j < n_samples_; ++j) {
      if (observations(i, j)) {
        const double diff = x_data(i, j) - mu(i);
        accumulator_sq += diff * diff;
      }
    }
    feature_var(i) = (count > 0.0) ? (accumulator_sq / count) : 0.0;
  }
  const double noise_var = std::clamp(arma::median(feature_var), 1e-8, 1e12);

  // Initialize W with random values, then normalize and scale.
  arma::mat w = arma::randn(n_features_in_, n_components_);
  for (arma::uword j = 0; j < n_components_; ++j) {
    const double norm = arma::norm(w.col(j));
    if (norm > 0.0) {
      w.col(j) /= norm;
    } else {
      w.col(j).fill(1.0 / std::sqrt(n_features_in_));
    }
  }
  const arma::vec signal =
      arma::clamp(feature_var - noise_var, 0.0, arma::datum::inf);
  const double average_signal = arma::mean(signal);
  const double scale = std::sqrt((average_signal / n_components_) + 1e-12);
  w *= scale;

  return {w, mu, noise_var};
}

PPCA::Params PPCA::em_complete_batch(const arma::mat& x_centered,
                                     const arma::umat& observations,
                                     const Params& initial_params) {
  arma::mat w = initial_params.components;
  arma::vec mu = initial_params.mean;
  double noise_var = initial_params.noise_variance;
  double prev_nll = std::numeric_limits<double>::infinity();
  auto batching = compute_batching();
  const std::size_t n_batches = batching.first;
  const std::size_t effective_batch_size = batching.second;

  auto minibatch_complete_update =
      [&](const arma::mat& w_current,
          double noise_var_current) -> UpdateResult {
    arma::mat sum_ezz(n_components_, n_components_, arma::fill::zeros);
    arma::mat sum_ez_x_T(n_components_, n_features_in_, arma::fill::zeros);
    double sum_x_centered_sq = 0.0;

    for (std::size_t b = 0; b < n_batches; ++b) {
      const arma::uword start =
          static_cast<arma::uword>(b * effective_batch_size);
      const arma::uword end = std::min<arma::uword>(
          x_centered.n_cols, start + effective_batch_size);
      if (start >= end) continue;
      const arma::uword current_batch_size = end - start;
      const arma::span col_span(start, end - 1);

      const arma::mat x_centered_batch = x_centered.cols(col_span);
      const arma::umat observations_batch = observations.cols(col_span);
      const LatentMoments latent_moments = latent_posterior_moments(
          x_centered_batch, w_current, noise_var_current, observations_batch);

      const arma::mat& ez_batch = latent_moments.ez;
      const arma::cube& ezz_batch = latent_moments.ezz;

      for (arma::uword i = 0; i < current_batch_size; ++i) {
        sum_ezz += ezz_batch.slice(i);
      }
      sum_ez_x_T += ez_batch * x_centered_batch.t();
      sum_x_centered_sq += arma::accu(arma::square(x_centered_batch));
    }

    // M-step: Update w and noise variance.
    const arma::mat w_new = arma::solve(arma::symmatu(sum_ezz), sum_ez_x_T,
                                        arma::solve_opts::likely_sympd)
                                .t();
    const arma::mat w_transpose_w = w_new.t() * w_new;

    const double term1 = sum_x_centered_sq;
    const double term2 = 2.0 * arma::trace(sum_ez_x_T * w_new);
    const double term3 = arma::accu(sum_ezz % w_transpose_w);
    double noise_var_new =
        (term1 - term2 + term3) / (n_features_in_ * n_samples_);

    const double nll = arma::accu(negative_log_likelihood(
        x_centered, w_new, noise_var_new, observations));

    return {std::move(w_new), noise_var_new, nll};
  };

  run_em_loop(minibatch_complete_update, w, noise_var, prev_nll,
              "Mini-batch EM (complete data) did not converge");
  return {w, mu, noise_var};
}

PPCA::Params PPCA::em_missing_batch(const arma::mat& x_centered,
                                    const arma::umat& observations,
                                    const Params& initial_params) {
  arma::mat w = initial_params.components;
  arma::vec mu = initial_params.mean;
  double noise_var = initial_params.noise_variance;
  double prev_nll = std::numeric_limits<double>::infinity();
  auto batching = compute_batching();
  const std::size_t n_batches = batching.first;
  const std::size_t effective_batch_size = batching.second;

  auto minibatch_missing_update =
      [&](const arma::mat& w_current,
          double noise_var_current) -> UpdateResult {
    // E-step: Accumulate sufficient statistics.
    arma::cube sum_ezz_per_feature(n_components_, n_components_, n_features_in_,
                                   arma::fill::zeros);
    arma::mat rhs_per_feature(n_components_, n_features_in_, arma::fill::zeros);
    arma::vec sum_x_sq_per_feature(n_features_in_, arma::fill::zeros);
    arma::vec observation_count_per_feature(n_features_in_, arma::fill::zeros);

    for (std::size_t b = 0; b < n_batches; ++b) {
      const arma::uword start =
          static_cast<arma::uword>(b * effective_batch_size);
      const arma::uword end = std::min<arma::uword>(
          x_centered.n_cols, start + effective_batch_size);
      if (start >= end) continue;
      const arma::uword current_batch_size = end - start;
      const arma::span col_span(start, end - 1);

      const arma::mat x_centered_batch = x_centered.cols(col_span);
      const arma::umat observations_batch = observations.cols(col_span);
      const LatentMoments latent_moments = latent_posterior_moments(
          x_centered_batch, w_current, noise_var_current, observations_batch);

      const arma::mat& ez_batch = latent_moments.ez;
      const arma::cube& ezz_batch = latent_moments.ezz;

      for (arma::uword local_idx = 0; local_idx < current_batch_size;
           ++local_idx) {
        const arma::uword global_idx = start + local_idx;
        const arma::uvec observed_indices =
            arma::find(observations_batch.col(local_idx));
        const arma::vec& ez_n = ez_batch.col(local_idx);
        const arma::mat& ezz_n = ezz_batch.slice(local_idx);

        for (arma::uword observed_feature_idx : observed_indices) {
          const double x_in = x_centered(observed_feature_idx, global_idx);
          sum_ezz_per_feature.slice(observed_feature_idx) += ezz_n;
          rhs_per_feature.col(observed_feature_idx) += ez_n * x_in;
          sum_x_sq_per_feature(observed_feature_idx) += x_in * x_in;
          observation_count_per_feature(observed_feature_idx) += 1.0;
        }
      }
    }

    // M-step: Update w and noise variance.
    arma::mat w_new(n_features_in_, n_components_, arma::fill::zeros);
    double noise_variance_numerator = 0.0;
    double total_observations = 0.0;

    for (arma::uword i = 0; i < n_features_in_; ++i) {
      const double count_i = observation_count_per_feature(i);
      if (count_i == 0.0) continue;

      arma::mat& S_i = sum_ezz_per_feature.slice(i);
      const arma::vec& r_i = rhs_per_feature.col(i);

      const arma::vec w_i =
          arma::solve(arma::symmatu(S_i), r_i, arma::solve_opts::likely_sympd);
      w_new.row(i) = w_i.t();

      const double term_a = sum_x_sq_per_feature(i);
      const double term_b = 2.0 * arma::dot(w_i, r_i);
      const double term_c = arma::as_scalar(w_i.t() * S_i * w_i);
      noise_variance_numerator += (term_a - term_b + term_c);
      total_observations += count_i;
    }

    double noise_var_new =
        noise_variance_numerator / std::max(1.0, total_observations);
    noise_var_new = std::clamp(noise_var_new, 1e-12, 1e12);
    const double nll = arma::accu(negative_log_likelihood(
        x_centered, w_new, noise_var_new, observations));

    return {std::move(w_new), noise_var_new, nll};
  };

  run_em_loop(minibatch_missing_update, w, noise_var, prev_nll,
              "Mini-batch EM (missing data) did not converge");
  return {w, mu, noise_var};
}

}  // namespace ppca
