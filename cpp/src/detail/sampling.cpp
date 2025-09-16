// MIT License
#include "ppca/detail/sampling.hpp"

#include <armadillo>

namespace ppca::detail {

arma::cube sample_gaussian(const arma::mat &means, const arma::cube &covs,
                           std::size_t n_draws) {
  const std::size_t d = means.n_rows;
  const std::size_t N = means.n_cols;

  arma::cube samples(d, N, n_draws, arma::fill::none);

  for (std::size_t i = 0; i < N; ++i) {
    arma::mat C = covs.slice(i);
    C = arma::symmatu(C);

    if (C.is_zero()) {
      for (std::size_t j = 0; j < n_draws; ++j)
        samples.slice(j).col(i) = means.col(i);
      continue;
    }

    arma::mat L;
    if (!arma::chol(L, C, "lower")) {
      arma::vec eigval;
      arma::mat eigvec;
      arma::eig_sym(eigval, eigvec, C);

      const double max_ev = eigval.max();
      const double tol = 1e-12 * std::max(1.0, max_ev);
      for (arma::uword k = 0; k < eigval.n_elem; ++k)
        if (eigval(k) < 0 && std::abs(eigval(k)) <= tol) eigval(k) = 0.0;

      const arma::uvec keep = arma::find(eigval > tol);
      if (keep.is_empty()) {
        for (std::size_t j = 0; j < n_draws; ++j)
          samples.slice(j).col(i) = means.col(i);
        continue;
      }
      const arma::mat U = eigvec.cols(keep);
      const arma::vec s = arma::sqrt(eigval(keep));
      L = U * arma::diagmat(s);
    }

    const std::size_t k = L.n_cols;
    for (std::size_t j = 0; j < n_draws; ++j) {
      const arma::vec z = arma::randn<arma::vec>(k);
      samples.slice(j).col(i) = means.col(i) + L * z;
    }
  }
  return samples;
}

}  // namespace ppca::detail
