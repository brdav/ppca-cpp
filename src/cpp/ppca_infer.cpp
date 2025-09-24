// Copyright 2025 brdav

// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

#include "ppca.hpp"
#include "utils.hpp"

namespace ppca {

std::pair<arma::mat, arma::cube> PPCA::posterior_latent(
    const arma::mat& x_data) const {
  ensure_fitted();
  ppca::utils::check_dims(components_.n_rows, x_data.n_rows,
                          "posterior_latent");

  const arma::mat x_centered = x_data.each_col() - mean_;
  const arma::uword num_samples = x_data.n_cols;
  const arma::umat observations = ppca::utils::mask_finite(x_centered);
  const LatentMoments latent_moments = latent_posterior_moments(
      x_centered, components_, noise_variance_, observations);

  arma::mat mean_z = latent_moments.ez;
  const arma::cube& ezz = latent_moments.ezz;

  arma::cube cov_z(n_components_, n_components_, num_samples, arma::fill::none);
  for (arma::uword i = 0; i < num_samples; ++i) {
    const arma::vec z_i = mean_z.col(i);
    cov_z.slice(i) = ezz.slice(i) - z_i * z_i.t();
  }
  return {mean_z, cov_z};
}

std::pair<arma::mat, arma::cube> PPCA::likelihood(
    const arma::mat& z_latent) const {
  ensure_fitted();
  ppca::utils::check_dims(components_.n_cols, z_latent.n_rows, "likelihood");

  const arma::uword num_features = components_.n_rows;
  const arma::uword num_samples = z_latent.n_cols;

  arma::mat mean_x = components_ * z_latent;
  mean_x.each_col() += mean_;

  arma::cube cov_x(num_features, num_features, num_samples, arma::fill::zeros);
  for (arma::uword i = 0; i < num_samples; ++i) {
    cov_x.slice(i).eye(num_features, num_features);
    cov_x.slice(i).diag() *= noise_variance_;
  }
  return {mean_x, cov_x};
}

std::pair<arma::mat, arma::cube> PPCA::impute_missing(
    const arma::mat& x_data) const {
  ensure_fitted();
  ppca::utils::check_dims(components_.n_rows, x_data.n_rows, "impute_missing");
  auto [mean_z, cov_z] = posterior_latent(x_data);

  const arma::uword num_features = components_.n_rows;
  const arma::uword num_samples = mean_z.n_cols;
  const arma::umat observations = ppca::utils::mask_finite(x_data);

  // Start with the expected reconstruction, then fill in observed values.
  arma::mat mean_x_imputed = components_ * mean_z;
  mean_x_imputed.each_col() += mean_;
  const arma::uvec observed_flat_indices = arma::find(observations);
  mean_x_imputed.elem(observed_flat_indices) =
      x_data.elem(observed_flat_indices);

  // Calculate covariance only for the missing entries.
  arma::cube cov_x_imputed(num_features, num_features, num_samples,
                           arma::fill::none);
  const arma::mat w_transpose = components_.t();
  for (arma::uword i = 0; i < num_samples; ++i) {
    const arma::uvec observed_indices = arma::find(observations.col(i));
    const arma::uvec missing_indices = arma::find(observations.col(i) == 0u);

    if (observed_indices.is_empty()) {  // All features missing
      arma::mat full_cov = (components_ * cov_z.slice(i)) * w_transpose;
      full_cov.diag() += noise_variance_;
      cov_x_imputed.slice(i) = std::move(full_cov);
      continue;
    }

    cov_x_imputed.slice(i).zeros(num_features, num_features);
    if (missing_indices.is_empty()) continue;  // No features missing

    const arma::mat w_missing = components_.rows(missing_indices);
    arma::mat cov_missing = (w_missing * cov_z.slice(i)) * w_missing.t();
    cov_missing.diag() += noise_variance_;
    cov_x_imputed.slice(i).submat(missing_indices, missing_indices) =
        std::move(cov_missing);
  }
  return {mean_x_imputed, cov_x_imputed};
}

arma::mat PPCA::lmmse_reconstruction(const arma::mat& z_latent) const {
  ensure_fitted();
  ppca::utils::check_dims(components_.n_cols, z_latent.n_rows,
                          "lmmse_reconstruction");
  const arma::vec w_col_sum_sq = arma::sum(arma::square(components_), 0).t();
  const arma::vec scale = (w_col_sum_sq + noise_variance_) / w_col_sum_sq;
  arma::mat x_reconstructed = components_ * arma::diagmat(scale) * z_latent;
  x_reconstructed.each_col() += mean_;
  return x_reconstructed;
}

PPCA::LatentMoments PPCA::latent_posterior_moments(
    const arma::mat& x_centered, const arma::mat& w, double noise_variance,
    const arma::umat& observations) const {
  const arma::uword num_samples = x_centered.n_cols;
  const bool all_observed = (arma::accu(observations) == observations.n_elem);
  arma::mat ez(n_components_, num_samples, arma::fill::zeros);
  arma::cube ezz(n_components_, n_components_, num_samples, arma::fill::zeros);

  if (all_observed) {
    const arma::mat w_transpose = w.t();
    arma::mat m_matrix = w_transpose * w;
    m_matrix.diag() += noise_variance;
    const arma::mat m_inv = arma::inv_sympd(m_matrix);
    ez = (m_inv * w_transpose) * x_centered;
    const arma::mat base_covariance = noise_variance * m_inv;
    for (arma::uword i = 0; i < num_samples; ++i) {
      ezz.slice(i) = base_covariance + ez.col(i) * ez.col(i).t();
    }
    return {ez, ezz};
  }

  // Handle missing data sample by sample.
  for (arma::uword i = 0; i < num_samples; ++i) {
    const arma::uvec observed_indices = arma::find(observations.col(i));
    const arma::mat w_obs = w.rows(observed_indices);
    arma::mat m_n_matrix = w_obs.t() * w_obs;
    m_n_matrix.diag() += noise_variance;
    const arma::mat m_n_inv = arma::inv_sympd(m_n_matrix);
    const arma::vec z_i =
        m_n_inv * w_obs.t() * x_centered(observed_indices, arma::uvec{i});
    ez.col(i) = z_i;
    ezz.slice(i) = noise_variance * m_n_inv + z_i * z_i.t();
  }
  return {ez, ezz};
}

arma::vec PPCA::negative_log_likelihood(const arma::mat& x_centered,
                                        const arma::mat& w,
                                        double noise_variance,
                                        const arma::umat& observations) const {
  const bool all_observed = (arma::accu(observations) == observations.n_elem);
  const arma::uword num_samples = x_centered.n_cols;
  arma::vec nll_per_sample(num_samples, arma::fill::zeros);

  if (all_observed) {
    const arma::uword num_features = n_features_in_;
    arma::mat m_matrix = w.t() * w;
    m_matrix.diag() += noise_variance;
    const double log_det_m = arma::log_det_sympd(m_matrix);
    const double dof =
        static_cast<double>(num_features) - static_cast<double>(n_components_);
    const double log_det_c = dof * std::log(noise_variance) + log_det_m;

    for (arma::uword i = 0; i < num_samples; ++i) {
      const arma::vec& x_i = x_centered.col(i);
      const arma::vec w_transpose_x = w.t() * x_i;
      const arma::vec alpha =
          arma::solve(m_matrix, w_transpose_x, arma::solve_opts::likely_sympd);
      const double quad_term =
          (1.0 / noise_variance) *
          (arma::dot(x_i, x_i) - arma::dot(w_transpose_x, alpha));
      nll_per_sample(i) =
          0.5 * (num_features * std::log(2.0 * arma::datum::pi) + log_det_c +
                 quad_term);
    }
  } else {
    for (arma::uword i = 0; i < num_samples; ++i) {
      const arma::uvec observed_indices = arma::find(observations.col(i));
      const arma::uword p_obs = observed_indices.n_elem;
      const arma::mat w_obs = w.rows(observed_indices);
      arma::mat m_n_matrix = w_obs.t() * w_obs;
      m_n_matrix.diag() += noise_variance;
      const double log_det_m = arma::log_det_sympd(m_n_matrix);
      const double dof =
          static_cast<double>(p_obs) - static_cast<double>(n_components_);
      const double log_det_c = dof * std::log(noise_variance) + log_det_m;

      const arma::vec x_obs = x_centered(observed_indices, arma::uvec{i});
      const arma::vec w_transpose_x = w_obs.t() * x_obs;
      const arma::vec alpha = arma::solve(m_n_matrix, w_transpose_x,
                                          arma::solve_opts::likely_sympd);
      const double quad_term =
          (1.0 / noise_variance) *
          (arma::dot(x_obs, x_obs) - arma::dot(w_transpose_x, alpha));
      nll_per_sample(i) = 0.5 * (p_obs * std::log(2.0 * arma::datum::pi) +
                                 log_det_c + quad_term);
    }
  }
  return nll_per_sample;
}

arma::cube PPCA::sample_posterior_latent(const arma::mat& x_data,
                                         std::size_t num_draws) const {
  auto [mean_z, cov_z] = posterior_latent(x_data);
  return ppca::utils::sample_gaussian(mean_z, cov_z, num_draws);
}

arma::cube PPCA::sample_likelihood(const arma::mat& z_latent,
                                   std::size_t num_draws) const {
  auto [mean_x, cov_x] = likelihood(z_latent);
  return ppca::utils::sample_gaussian(mean_x, cov_x, num_draws);
}

arma::cube PPCA::sample_missing(const arma::mat& x_data,
                                std::size_t num_draws) const {
  auto [mean_x, cov_x] = impute_missing(x_data);
  return ppca::utils::sample_gaussian(mean_x, cov_x, num_draws);
}

}  // namespace ppca
