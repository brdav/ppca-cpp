// MIT License
#include "ppca/detail/sampling.hpp"
#include "ppca/detail/utils.hpp"
#include "ppca/ppca.hpp"

namespace ppca {

std::pair<arma::mat, arma::cube> PPCA::posterior_latent(
    const arma::mat &X) const {
  ensure_fitted();
  detail::check_dims(W_.n_rows, X.n_rows, "posterior_latent");
  const arma::mat Xc = X.each_col() - mean_;
  const arma::uword n_samples_local = X.n_cols;
  const arma::umat obs = detail::mask_finite(Xc);
  const LatentMoments lm =
      latent_posterior_moments(Xc, W_, noise_variance_, obs);
  arma::mat mZ = lm.Ez;
  arma::cube Ezz = lm.Ezz;

  arma::cube covZ(n_components_, n_components_, n_samples_local,
                  arma::fill::none);
  for (arma::uword n = 0; n < n_samples_local; ++n) {
    const arma::vec z = mZ.col(n);
    covZ.slice(n) = Ezz.slice(n) - z * z.t();
  }
  return {mZ, covZ};
}

std::pair<arma::mat, arma::cube> PPCA::likelihood(const arma::mat &Z) const {
  ensure_fitted();
  detail::check_dims(W_.n_cols, Z.n_rows, "likelihood");

  const arma::uword p = W_.n_rows;
  const arma::uword m = Z.n_cols;
  arma::mat mX = W_ * Z;
  mX.each_col() += mean_;

  arma::cube covX(p, p, m, arma::fill::zeros);
  for (arma::uword n = 0; n < m; ++n) {
    covX.slice(n).eye(p, p);
    covX.slice(n).diag() *= noise_variance_;
  }
  return {mX, covX};
}

std::pair<arma::mat, arma::cube> PPCA::impute_missing(
    const arma::mat &X) const {
  ensure_fitted();
  detail::check_dims(W_.n_rows, X.n_rows, "impute_missing");
  arma::mat mZ;
  arma::cube covZ;
  std::tie(mZ, covZ) = posterior_latent(X);

  const arma::uword p = W_.n_rows;
  const arma::uword m = mZ.n_cols;
  const arma::umat obs = detail::mask_finite(X);

  arma::mat mX = W_ * mZ;
  mX.each_col() += mean_;
  const arma::uvec obs_idx = arma::find(obs);
  mX.elem(obs_idx) = X.elem(obs_idx);

  arma::cube covX(p, p, m, arma::fill::none);
  const arma::mat WT = W_.t();
  for (arma::uword n = 0; n < m; ++n) {
    const arma::uvec idx_obs = arma::find(obs.col(n));
    const arma::uvec idx_miss = arma::find(obs.col(n) == 0u);

    if (idx_obs.is_empty()) {
      arma::mat full = (W_ * covZ.slice(n)) * WT;
      full.diag() += noise_variance_;
      covX.slice(n) = std::move(full);
      continue;
    }
    covX.slice(n).zeros(p, p);
    if (idx_miss.is_empty()) continue;

    const arma::mat Wm = W_.rows(idx_miss);
    arma::mat cov_miss = (Wm * covZ.slice(n)) * Wm.t();
    cov_miss.diag() += noise_variance_;
    covX.slice(n).submat(idx_miss, idx_miss) = std::move(cov_miss);
  }
  return {mX, covX};
}

arma::mat PPCA::lmmse_reconstruction(const arma::mat &Z) const {
  ensure_fitted();
  detail::check_dims(W_.n_cols, Z.n_rows, "lmmse_reconstruction");
  const arma::vec d = arma::sum(arma::square(W_), 0).t();
  const arma::vec scale = (d + noise_variance_) / d;
  arma::mat X_hat = W_ * arma::diagmat(scale) * Z;
  X_hat.each_col() += mean_;
  return X_hat;
}

PPCA::LatentMoments PPCA::latent_posterior_moments(
    const arma::mat &Xc, const arma::mat &W, double noise_var,
    const arma::umat &obs) const {
  const arma::uword n_samples_local = Xc.n_cols;
  const bool all_obs = (arma::accu(obs) == obs.n_elem);
  arma::mat Ez(n_components_, n_samples_local, arma::fill::zeros);
  arma::cube Ezz(n_components_, n_components_, n_samples_local,
                 arma::fill::zeros);

  if (all_obs) {
    const arma::mat Wt = W.t();
    arma::mat M = Wt * W;
    M.diag() += noise_var;
    const arma::mat M_inv = arma::inv_sympd(M);
    Ez = (M_inv * Wt) * Xc;
    const arma::mat base = noise_var * M_inv;
    for (arma::uword n = 0; n < n_samples_local; ++n) {
      Ezz.slice(n) = base + Ez.col(n) * Ez.col(n).t();
    }
    return {Ez, Ezz};
  }

  arma::mat w;
  arma::mat M_n(n_components_, n_components_);
  for (arma::uword n = 0; n < n_samples_local; ++n) {
    const arma::uvec idx_obs = arma::find(obs.col(n));
    w = W.rows(idx_obs);
    M_n = w.t() * w;
    M_n.diag() += noise_var;
    const arma::mat M_n_inv = arma::inv_sympd(M_n);
    const arma::vec z = M_n_inv * w.t() * Xc(idx_obs, arma::uvec{n});
    Ez.col(n) = z;
    Ezz.slice(n) = noise_var * M_n_inv + z * z.t();
  }
  return {Ez, Ezz};
}

arma::vec PPCA::negative_log_likelihood(const arma::mat &Xc, const arma::mat &W,
                                        double noise_var,
                                        const arma::umat &obs) const {
  const bool all_obs = (arma::accu(obs) == obs.n_elem);
  const arma::uword n_samples_local = Xc.n_cols;
  arma::vec nll(n_samples_local, arma::fill::zeros);
  if (all_obs) {
    const arma::uword p = n_features_in_;
    arma::mat M = W.t() * W;
    M.diag() += noise_var;
    const double logdetM = arma::log_det_sympd(M);
    const double dof =
        static_cast<double>(p) - static_cast<double>(n_components_);
    const double logdetC = dof * std::log(noise_var) + logdetM;

    for (arma::uword n = 0; n < n_samples_local; ++n) {
      const arma::vec x = Xc.col(n);
      const arma::vec wtx = W.t() * x;
      const arma::vec alpha =
          arma::solve(M, wtx, arma::solve_opts::likely_sympd);
      const double quad =
          (1.0 / noise_var) * (arma::dot(x, x) - arma::dot(wtx, alpha));
      nll(n) = 0.5 * (p * std::log(2.0 * arma::datum::pi) + logdetC + quad);
    }
  } else {
    arma::mat M_n;
    for (arma::uword n = 0; n < n_samples_local; ++n) {
      const arma::uvec idx_obs = arma::find(obs.col(n));
      const arma::uword p = idx_obs.n_elem;
      const arma::mat w = W.rows(idx_obs);
      M_n = w.t() * w;
      M_n.diag() += noise_var;
      const double logdetM = arma::log_det_sympd(M_n);
      const double dof =
          static_cast<double>(p) - static_cast<double>(n_components_);
      const double logdetC = dof * std::log(noise_var) + logdetM;

      const arma::vec x = Xc(idx_obs, arma::uvec{n});
      const arma::vec wtx = w.t() * x;
      const arma::vec alpha =
          arma::solve(M_n, wtx, arma::solve_opts::likely_sympd);
      const double quad =
          (1.0 / noise_var) * (arma::dot(x, x) - arma::dot(wtx, alpha));
      nll(n) = 0.5 * (p * std::log(2.0 * arma::datum::pi) + logdetC + quad);
    }
  }
  return nll;
}

arma::cube PPCA::sample_posterior_latent(const arma::mat &X,
                                         std::size_t n_draws) const {
  auto [mZ, covZ] = posterior_latent(X);
  return detail::sample_gaussian(mZ, covZ, n_draws);
}

arma::cube PPCA::sample_likelihood(const arma::mat &Z,
                                   std::size_t n_draws) const {
  auto [mX, covX] = likelihood(Z);
  return detail::sample_gaussian(mX, covX, n_draws);
}

arma::cube PPCA::sample_missing(const arma::mat &X, std::size_t n_draws) const {
  auto [mX, covX] = impute_missing(X);
  return detail::sample_gaussian(mX, covX, n_draws);
}

}  // namespace ppca
