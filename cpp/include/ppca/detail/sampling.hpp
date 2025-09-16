// MIT License
#pragma once
#include <armadillo>

namespace ppca::detail {

// Sample from a set of Gaussians with per-column means and per-slice
// covariances
arma::cube sample_gaussian(const arma::mat &means, const arma::cube &covs,
                           std::size_t n_draws);

}  // namespace ppca::detail
