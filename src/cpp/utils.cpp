// Copyright 2025 brdav

// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include "utils.hpp"

#include <stdexcept>
#include <string>

namespace ppca {
namespace utils {

void check_dims(arma::uword expected, arma::uword got, const char* context) {
  if (expected != got) {
    throw std::invalid_argument(
        std::string("Dimension mismatch in ") + context + ": expected " +
        std::to_string(expected) + ", got " + std::to_string(got));
  }
}

arma::umat mask_finite(const arma::mat& A) {
  arma::umat M(A.n_rows, A.n_cols);
  for (arma::uword i = 0; i < A.n_rows; ++i) {
    for (arma::uword j = 0; j < A.n_cols; ++j) {
      M(i, j) = std::isfinite(A(i, j)) ? 1u : 0u;
    }
  }
  return M;
}

arma::cube sample_gaussian(const arma::mat& means, const arma::cube& covs,
                           std::size_t n_draws) {
  const std::size_t d = means.n_rows;
  const std::size_t n = means.n_cols;

  arma::cube samples(d, n, n_draws, arma::fill::none);

  for (std::size_t i = 0; i < n; ++i) {
    arma::mat c = covs.slice(i);
    c = arma::symmatu(c);

    if (c.is_zero()) {
      for (std::size_t j = 0; j < n_draws; ++j) {
        samples.slice(j).col(i) = means.col(i);
      }
      continue;
    }

    arma::mat l;
    if (!arma::chol(l, c, "lower")) {
      arma::vec eigval;
      arma::mat eigvec;
      arma::eig_sym(eigval, eigvec, c);

      const double max_ev = eigval.max();
      const double tol = 1e-12 * std::max(1.0, max_ev);
      for (arma::uword k = 0; k < eigval.n_elem; ++k) {
        if (eigval(k) < 0 && std::abs(eigval(k)) <= tol) {
          eigval(k) = 0.0;
        }
      }

      const arma::uvec keep = arma::find(eigval > tol);
      if (keep.is_empty()) {
        for (std::size_t j = 0; j < n_draws; ++j) {
          samples.slice(j).col(i) = means.col(i);
        }
        continue;
      }
      const arma::mat u = eigvec.cols(keep);
      const arma::vec s = arma::sqrt(eigval(keep));
      l = u * arma::diagmat(s);
    }

    const std::size_t k = l.n_cols;
    for (std::size_t j = 0; j < n_draws; ++j) {
      const arma::vec z = arma::randn<arma::vec>(k);
      samples.slice(j).col(i) = means.col(i) + l * z;
    }
  }
  return samples;
}

}  // namespace utils
}  // namespace ppca
