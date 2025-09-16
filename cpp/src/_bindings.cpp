#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <armadillo>
#include <carma>

#include "ppca/ppca.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_bindings, m) {
  m.doc() =
      "PPCA C++ core bindings (minimal conversions; orientation handled in "
      "Python)";

  py::class_<ppca::PPCA>(m, "_CPP_PPCA")
      .def(py::init<std::size_t, std::size_t, std::size_t, double, bool,
                    std::size_t, std::optional<unsigned int>>(),
           py::arg("n_components"), py::arg("max_iter") = 10000,
           py::arg("min_iter") = 20, py::arg("rtol") = 1e-8,
           py::arg("rotate_to_orthogonal") = true, py::arg("batch_size") = 0,
           py::arg("random_state") = py::none())
      .def(
          "fit",
          [](ppca::PPCA &self, py::array X) {
            auto view = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(X));
            arma::mat A(view);  // copy to ensure owning (avoid stride issues)
            self.fit(A);
            return &self;
          },
          py::return_value_policy::reference_internal)
      .def("score_samples",
           [](const ppca::PPCA &self, py::array X) {
             auto view = carma::arr_to_mat_view<double>(
                 py::cast<py::array_t<double>>(X));
             arma::mat A(view);
             const arma::vec &ll = self.score_samples(A);
             return carma::col_to_arr(ll);
           })
      .def("score",
           [](const ppca::PPCA &self, py::array X) {
             auto view = carma::arr_to_mat_view<double>(
                 py::cast<py::array_t<double>>(X));
             arma::mat A(view);
             return self.score(A);
           })
      .def("get_covariance",
           [](const ppca::PPCA &self) {
             arma::mat cov = self.get_covariance();
             return carma::mat_to_arr(cov);
           })
      .def("get_precision",
           [](const ppca::PPCA &self) {
             arma::mat prec = self.get_precision();
             return carma::mat_to_arr(prec);
           })
      .def(
          "posterior_latent",
          [](const ppca::PPCA &self, py::array X) {
            auto view = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(X));
            arma::mat A(view);
            auto [Ez, Cz] = self.posterior_latent(A);
            return py::make_tuple(carma::mat_to_arr(Ez),
                                  carma::cube_to_arr(Cz));
          },
          py::arg("X"))
      .def(
          "likelihood",
          [](const ppca::PPCA &self, py::array Z) {
            auto view_Z = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(Z));
            arma::mat Z_mat(view_Z);
            auto [Xhat, cov] = self.likelihood(Z_mat);
            return py::make_tuple(carma::mat_to_arr(Xhat),
                                  carma::cube_to_arr(cov));
          },
          py::arg("Z"))
      .def(
          "impute_missing",
          [](const ppca::PPCA &self, py::array X) {
            auto view = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(X));
            arma::mat A(view);
            auto [Xhat, cov] = self.impute_missing(A);
            return py::make_tuple(carma::mat_to_arr(Xhat),
                                  carma::cube_to_arr(cov));
          },
          py::arg("X"))
      .def(
          "sample_posterior_latent",
          [](const ppca::PPCA &self, py::array X, std::size_t n_draws = 1) {
            auto view = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(X));
            arma::mat A(view);
            arma::cube Z_tilde = self.sample_posterior_latent(A, n_draws);
            return carma::cube_to_arr(Z_tilde);
          },
          py::arg("X"), py::arg("n_draws"))
      .def(
          "sample_likelihood",
          [](const ppca::PPCA &self, py::array Z, std::size_t n_draws = 1) {
            auto view_Z = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(Z));
            arma::mat Z_mat(view_Z);
            arma::cube X_tilde = self.sample_likelihood(Z_mat, n_draws);
            return carma::cube_to_arr(X_tilde);
          },
          py::arg("Z"), py::arg("n_draws"))
      .def(
          "sample_missing",
          [](const ppca::PPCA &self, py::array X, std::size_t n_draws = 1) {
            auto view = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(X));
            arma::mat A(view);
            arma::cube X_tilde = self.sample_missing(A, n_draws);
            return carma::cube_to_arr(X_tilde);
          },
          py::arg("X"), py::arg("n_draws"))
      .def(
          "lmmse_reconstruction",
          [](const ppca::PPCA &self, py::array Ez) {
            auto view_Ez = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(Ez));
            arma::mat Ez_mat(view_Ez);
            arma::mat X_hat = self.lmmse_reconstruction(Ez_mat);
            return carma::mat_to_arr(X_hat);
          },
          py::arg("Ez"))
      .def(
          "set_params",
          [](ppca::PPCA &self, py::array W, py::array mu, double sig2) {
            auto W_view = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(W));
            arma::mat W_mat(W_view);
            auto mu_view = carma::arr_to_col_view<double>(
                py::cast<py::array_t<double>>(mu));
            arma::vec mu_vec(mu_view);
            self.set_params(W_mat, mu_vec, sig2);
            return &self;
          },
          py::arg("W"), py::arg("mu"), py::arg("sig2"),
          py::return_value_policy::reference_internal)
      .def("get_params",
           [](const ppca::PPCA &self) {
             auto p = self.get_params();
             py::dict d;
             d["W"] = carma::mat_to_arr(p.W);
             d["mu"] = carma::col_to_arr(p.mu);
             d["sig2"] = p.sig2;
             return d;
           })
      .def_property_readonly("components",
                             [](const ppca::PPCA &self) {
                               const arma::mat &C = self.components();
                               return carma::mat_to_arr(C);
                             })
      .def_property_readonly("mean",
                             [](const ppca::PPCA &self) {
                               const arma::vec &mvec = self.mean();
                               return carma::col_to_arr(mvec);
                             })
      .def_property_readonly(
          "noise_variance",
          [](const ppca::PPCA &self) { return self.noise_variance(); })
      .def_property_readonly("explained_variance",
                             [](const ppca::PPCA &self) {
                               const arma::vec &v = self.explained_variance();
                               return carma::col_to_arr(v);
                             })
      .def_property_readonly("explained_variance_ratio",
                             [](const ppca::PPCA &self) {
                               const arma::vec &v =
                                   self.explained_variance_ratio();
                               return carma::col_to_arr(v);
                             })
      .def_property_readonly(
          "n_samples", [](const ppca::PPCA &self) { return self.n_samples(); })
      .def_property_readonly(
          "n_features_in",
          [](const ppca::PPCA &self) { return self.n_features_in(); })
      .def_property_readonly("n_components", [](const ppca::PPCA &self) {
        return self.n_components();
      });
}
