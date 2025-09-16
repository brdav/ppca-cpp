// MIT License
#include "ppca/detail/utils.hpp"
#include "ppca/ppca.hpp"

namespace ppca {

PPCA::Params PPCA::em(const arma::mat &X, const arma::umat &obs) {
  Params init = initialize_parameters(X, obs);
  const arma::mat Xc = X.each_col() - init.mu;
  const bool any_missing = (arma::accu(obs) != obs.n_elem);
  if (!any_missing) {
    return em_complete_batch(Xc, obs, init);
  } else {
    return em_missing_batch(Xc, obs, init);
  }
}

void PPCA::run_em_loop(
    std::function<UpdateResult(const arma::mat &, double)> compute_update,
    arma::mat &W, double &noise_var, double &prev_nll,
    const char *fail_msg) const {
  for (std::size_t iter = 0; iter < max_iter_; ++iter) {
    UpdateResult r = compute_update(W, noise_var);
    bool done = false;
    if (iter >= min_iter_) {
      double rel = std::abs(r.nll - prev_nll) / (std::abs(prev_nll) + 1e-12);
      if (rel < rtol_) done = true;
    }
    W = std::move(r.W_new);
    noise_var = r.noise_var_new;
    prev_nll = r.nll;
    if (done) return;  // converged
    if (iter + 1 == max_iter_) throw std::runtime_error(fail_msg);
  }
}

std::pair<std::size_t, std::size_t> PPCA::compute_batching() const {
  if (batch_size_ == 0) {
    return {1, static_cast<std::size_t>(n_samples_)};
  }
  const std::size_t n_batches =
      (static_cast<std::size_t>(n_samples_) + batch_size_ - 1) / batch_size_;
  return {n_batches, batch_size_};
}

PPCA::Params PPCA::initialize_parameters(const arma::mat &X,
                                         const arma::umat &obs) {
  arma::vec mu(n_features_in_, arma::fill::zeros);
  const arma::vec counts = arma::conv_to<arma::vec>::from(arma::sum(obs, 1));
  for (arma::uword i = 0; i < n_features_in_; ++i) {
    double acc = 0.0;
    for (arma::uword j = 0; j < n_samples_; ++j) {
      if (obs(i, j)) {
        acc += X(i, j);
      }
    }
    mu(i) = counts(i) > 0.0 ? acc / counts(i) : 0.0;
  }

  arma::vec feature_var(n_features_in_, arma::fill::zeros);
  for (arma::uword i = 0; i < n_features_in_; ++i) {
    double acc2 = 0.0;
    double cnt = counts(i);
    for (arma::uword j = 0; j < n_samples_; ++j) {
      if (obs(i, j)) {
        double d = X(i, j) - mu(i);
        acc2 += d * d;
      }
    }
    feature_var(i) = (cnt > 0.0) ? (acc2 / cnt) : 0.0;
  }
  const double noise_var = std::clamp(arma::median(feature_var), 1e-8, 1e12);

  arma::mat W = arma::randn(n_features_in_, n_components_);
  for (arma::uword j = 0; j < n_components_; ++j) {
    double nrm = arma::norm(W.col(j));
    if (nrm > 0.0) {
      W.col(j) /= nrm;
    } else {
      W.col(j).fill(1.0 / std::sqrt(n_features_in_));
    }
  }
  arma::vec signal =
      arma::clamp(feature_var - noise_var, 0.0, arma::datum::inf);
  const double avg_signal = arma::mean(signal);
  const double scale = std::sqrt((avg_signal / n_components_) + 1e-12);
  W *= scale;
  return {W, mu, noise_var};
}

PPCA::Params PPCA::em_complete_batch(const arma::mat &Xc, const arma::umat &obs,
                                     const Params &init) {
  arma::mat W = init.W;
  arma::vec mu = init.mu;
  double noise_var = init.sig2;
  double prev_nll = std::numeric_limits<double>::infinity();
  auto [n_batches, eff_batch_size] = compute_batching();

  auto mb_complete_update = [&](const arma::mat &W_cur,
                                double noise_var_cur) -> UpdateResult {
    arma::mat sumEzz(n_components_, n_components_, arma::fill::zeros);
    arma::mat cross(n_components_, n_features_in_, arma::fill::zeros);
    double sum_xc2 = 0.0;
    for (std::size_t b = 0; b < n_batches; ++b) {
      const arma::uword start = static_cast<arma::uword>(b * eff_batch_size);
      const arma::uword end =
          std::min<arma::uword>(Xc.n_cols, start + eff_batch_size);
      const arma::uword bs = end - start;
      const arma::span cols(start, end - 1);
      const arma::mat Xc_batch = Xc.cols(cols);
      const arma::umat obs_batch = obs.cols(cols);
      const LatentMoments lm =
          latent_posterior_moments(Xc_batch, W_cur, noise_var_cur, obs_batch);
      const arma::mat Ez_b = lm.Ez;
      const arma::cube Ezz_b = lm.Ezz;
      for (arma::uword local_n = 0; local_n < bs; ++local_n) {
        sumEzz += Ezz_b.slice(local_n);
      }
      cross += Ez_b * Xc_batch.t();
      sum_xc2 += arma::accu(arma::square(Xc_batch));
    }
    arma::mat W_new = arma::solve(arma::symmatu(sumEzz), cross,
                                  arma::solve_opts::likely_sympd)
                          .t();
    const arma::mat WT_W = W_new.t() * W_new;
    const double term1 = sum_xc2;
    const double term2 = 2.0 * arma::trace(cross * W_new);
    const double term3 = arma::accu(sumEzz % WT_W);
    double noise_var_new =
        (term1 - term2 + term3) / (n_features_in_ * n_samples_);
    const double nll =
        arma::accu(negative_log_likelihood(Xc, W_new, noise_var_new, obs));
    return {std::move(W_new), noise_var_new, nll};
  };
  run_em_loop(mb_complete_update, W, noise_var, prev_nll,
              "Mini-batch EM (accumulate) did not converge");
  return {W, mu, noise_var};
}

PPCA::Params PPCA::em_missing_batch(const arma::mat &Xc, const arma::umat &obs,
                                    const Params &init) {
  arma::mat W = init.W;
  arma::vec mu = init.mu;
  double noise_var = init.sig2;
  double prev_nll = std::numeric_limits<double>::infinity();
  auto [n_batches, eff_batch_size] = compute_batching();

  auto mb_missing_update = [&](const arma::mat &W_cur,
                               double noise_var_cur) -> UpdateResult {
    arma::cube sumEzz_rows(n_components_, n_components_, n_features_in_,
                           arma::fill::zeros);
    arma::mat rhs(n_components_, n_features_in_, arma::fill::zeros);
    arma::vec sum_x2(n_features_in_, arma::fill::zeros);
    arma::vec obs_count_feat(n_features_in_, arma::fill::zeros);
    for (std::size_t b = 0; b < n_batches; ++b) {
      const arma::uword start = static_cast<arma::uword>(b * eff_batch_size);
      const arma::uword end =
          std::min<arma::uword>(Xc.n_cols, start + eff_batch_size);
      const arma::uword bs = end - start;
      const arma::span cols(start, end - 1);
      const arma::mat Xc_batch = Xc.cols(cols);
      const arma::umat obs_batch = obs.cols(cols);
      const LatentMoments lm =
          latent_posterior_moments(Xc_batch, W_cur, noise_var_cur, obs_batch);
      const arma::mat Ez_b = lm.Ez;
      const arma::cube Ezz_b = lm.Ezz;
      for (arma::uword local_n = 0; local_n < bs; ++local_n) {
        const arma::uword global_n = start + local_n;
        const arma::uvec idx_obs = arma::find(obs_batch.col(local_n));
        const arma::vec Ez_n = Ez_b.col(local_n);
        const arma::mat &Ezz_n = Ezz_b.slice(local_n);
        for (arma::uword t = 0; t < idx_obs.n_elem; ++t) {
          const arma::uword i = idx_obs[t];
          const double x_i = Xc(i, global_n);
          sumEzz_rows.slice(i) += Ezz_n;
          rhs.col(i) += Ez_n * x_i;
          sum_x2(i) += x_i * x_i;
          obs_count_feat(i) += 1.0;
        }
      }
    }
    arma::mat W_new(n_features_in_, n_components_, arma::fill::zeros);
    double noise_var_num = 0.0;
    double obs_total = 0.0;
    for (arma::uword i = 0; i < n_features_in_; ++i) {
      arma::mat &S = sumEzz_rows.slice(i);
      const arma::vec r = rhs.col(i);
      const double ci = obs_count_feat(i);
      const arma::vec w_i =
          arma::solve(arma::symmatu(S), r, arma::solve_opts::likely_sympd);
      W_new.row(i) = w_i.t();
      const double a = sum_x2(i);
      const double b = 2.0 * arma::dot(w_i, r);
      const double c = arma::as_scalar(w_i.t() * S * w_i);
      noise_var_num += (a - b + c);
      obs_total += ci;
    }
    double noise_var_new = noise_var_num / std::max(1.0, obs_total);
    noise_var_new = std::clamp(noise_var_new, 1e-12, 1e12);
    const double nll =
        arma::accu(negative_log_likelihood(Xc, W_new, noise_var_new, obs));
    return {std::move(W_new), noise_var_new, nll};
  };
  run_em_loop(mb_missing_update, W, noise_var, prev_nll,
              "Mini-batch EM (missing, accumulate) did not converge");
  return {W, mu, noise_var};
}

}  // namespace ppca
