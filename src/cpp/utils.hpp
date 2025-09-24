// Copyright 2025 brdav

// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#ifndef PPCA_HELPERS_H_
#define PPCA_HELPERS_H_

#include <armadillo>

namespace ppca {
namespace utils {

// Checks for dimension mismatch and throws an exception.
void check_dims(arma::uword expected, arma::uword got, const char* context);

// Creates a binary mask of finite entries (1 if finite, 0 otherwise).
arma::umat mask_finite(const arma::mat& A);

// Samples from a set of Gaussians with per-column means and per-slice
// covariances.
arma::cube sample_gaussian(const arma::mat& means, const arma::cube& covs,
                           std::size_t n_draws);

}  // namespace utils
}  // namespace ppca

#endif  // PPCA_HELPERS_H_
