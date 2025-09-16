// MIT License
#pragma once
#include <armadillo>
#include <stdexcept>
#include <string>

namespace ppca::detail {

inline void check_dims(arma::uword expected, arma::uword got,
                       const char* context) {
  if (expected != got) {
    throw std::invalid_argument(
        std::string("Dimension mismatch in ") + context + ": expected " +
        std::to_string(expected) + ", got " + std::to_string(got));
  }
}

// Create a binary mask of finite entries (1 if finite, 0 otherwise)
inline arma::umat mask_finite(const arma::mat& A) {
  arma::umat M(A.n_rows, A.n_cols);
  for (arma::uword i = 0; i < A.n_rows; ++i) {
    for (arma::uword j = 0; j < A.n_cols; ++j) {
      M(i, j) = std::isfinite(A(i, j)) ? 1u : 0u;
    }
  }
  return M;
}

}  // namespace ppca::detail
