// Copyright 2025 brdav

// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <armadillo>
#include <carma>

#include "ppca.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_ppca_bindings, m) {
  m.doc() =
      "PPCA C++ core bindings (minimal conversions; orientation handled in "
      "Python)";

  py::class_<ppca::PPCA>(m, "PPCA")
      .def(py::init<std::size_t, std::size_t, std::size_t, double, bool,
                    std::size_t, std::optional<unsigned int>>(),
           py::arg("n_components"), py::arg("max_iter") = 10000,
           py::arg("min_iter") = 20, py::arg("rtol") = 1e-8,
           py::arg("rotate_to_orthogonal") = true, py::arg("batch_size") = 0,
           py::arg("random_state") = py::none())

      .def(
          "fit",
          [](ppca::PPCA& self, py::array X) {
            auto x_view = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(X));
            arma::mat x_mat(
                x_view);  // copy to ensure owning (avoid stride issues)
            self.fit(x_mat);
            return &self;
          },
          py::return_value_policy::reference_internal)

      .def("score_samples",
           [](const ppca::PPCA& self, py::array X) {
             auto x_view = carma::arr_to_mat_view<double>(
                 py::cast<py::array_t<double>>(X));
             arma::mat x_mat(x_view);
             const arma::vec& ll_vec = self.score_samples(x_mat);
             return carma::col_to_arr(ll_vec);
           })

      .def("score",
           [](const ppca::PPCA& self, py::array X) {
             auto x_view = carma::arr_to_mat_view<double>(
                 py::cast<py::array_t<double>>(X));
             arma::mat x_mat(x_view);
             return self.score(x_mat);
           })

      .def("get_covariance",
           [](const ppca::PPCA& self) {
             arma::mat cov = self.get_covariance();
             return carma::mat_to_arr(cov);
           })

      .def("get_precision",
           [](const ppca::PPCA& self) {
             arma::mat prec = self.get_precision();
             return carma::mat_to_arr(prec);
           })

      .def(
          "posterior_latent",
          [](const ppca::PPCA& self, py::array X) {
            auto x_view = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(X));
            arma::mat x_mat(x_view);
            auto [ez, cz] = self.posterior_latent(x_mat);
            return py::make_tuple(carma::mat_to_arr(ez),
                                  carma::cube_to_arr(cz));
          },
          py::arg("X"))

      .def(
          "likelihood",
          [](const ppca::PPCA& self, py::array Z) {
            auto z_view = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(Z));
            arma::mat z_mat(z_view);
            auto [x_hat, cov_mat] = self.likelihood(z_mat);
            return py::make_tuple(carma::mat_to_arr(x_hat),
                                  carma::cube_to_arr(cov_mat));
          },
          py::arg("Z"))

      .def(
          "impute_missing",
          [](const ppca::PPCA& self, py::array X) {
            auto x_view = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(X));
            arma::mat x_mat(x_view);
            auto [x_hat, cov_mat] = self.impute_missing(x_mat);
            return py::make_tuple(carma::mat_to_arr(x_hat),
                                  carma::cube_to_arr(cov_mat));
          },
          py::arg("X"))

      .def(
          "sample_posterior_latent",
          [](const ppca::PPCA& self, py::array X,
             std::size_t n_draws /* = 1 */) {
            auto x_view = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(X));
            arma::mat x_mat(x_view);
            arma::cube z_tilde = self.sample_posterior_latent(x_mat, n_draws);
            return carma::cube_to_arr(z_tilde);
          },
          py::arg("X"), py::arg("n_draws"))

      .def(
          "sample_likelihood",
          [](const ppca::PPCA& self, py::array Z,
             std::size_t n_draws /* = 1 */) {
            auto z_view = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(Z));
            arma::mat z_mat(z_view);
            arma::cube x_tilde = self.sample_likelihood(z_mat, n_draws);
            return carma::cube_to_arr(x_tilde);
          },
          py::arg("Z"), py::arg("n_draws"))

      .def(
          "sample_missing",
          [](const ppca::PPCA& self, py::array X,
             std::size_t n_draws /* = 1 */) {
            auto x_view = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(X));
            arma::mat x_mat(x_view);
            arma::cube x_tilde = self.sample_missing(x_mat, n_draws);
            return carma::cube_to_arr(x_tilde);
          },
          py::arg("X"), py::arg("n_draws"))

      .def(
          "lmmse_reconstruction",
          [](const ppca::PPCA& self, py::array Ez) {
            auto ez_view = carma::arr_to_mat_view<double>(
                py::cast<py::array_t<double>>(Ez));
            arma::mat ez_mat(ez_view);
            arma::mat x_hat = self.lmmse_reconstruction(ez_mat);
            return carma::mat_to_arr(x_hat);
          },
          py::arg("Ez"))

      .def("get_params",
           [](const ppca::PPCA& self) {
             auto p = self.get_params();
             py::dict d;
             d["components"] = carma::mat_to_arr(p.components);
             d["mean"] = carma::col_to_arr(p.mean);
             d["noise_variance"] = p.noise_variance;
             return d;
           })

      .def(
          "set_params",
          [](ppca::PPCA& self, py::dict params) {
            if (!params.contains("components") || !params.contains("mean") ||
                !params.contains("noise_variance")) {
              throw py::key_error(
                  "set_params requires keys "
                  "'components','mean','noise_variance'");
            }

            py::object components_obj = params["components"];
            py::object mean_obj = params["mean"];
            py::object nv_obj = params["noise_variance"];

            auto components_view =
                carma::arr_to_mat_view<double>(py::cast<py::array_t<double>>(
                    py::cast<py::array>(components_obj)));
            arma::mat components_mat(components_view);

            auto mean_view = carma::arr_to_col_view<double>(
                py::cast<py::array_t<double>>(py::cast<py::array>(mean_obj)));
            arma::vec mean_vec(mean_view);

            double noise_variance = py::cast<double>(nv_obj);

            ppca::PPCA::Params p{components_mat, mean_vec, noise_variance};
            self.set_params(p);
            return &self;
          },
          py::arg("params"), py::return_value_policy::reference_internal)

      .def_property_readonly("components",
                             [](const ppca::PPCA& self) {
                               const arma::mat& comps = self.components();
                               return carma::mat_to_arr(comps);
                             })

      .def_property_readonly("mean",
                             [](const ppca::PPCA& self) {
                               const arma::vec& mean_vec = self.mean();
                               return carma::col_to_arr(mean_vec);
                             })

      .def_property_readonly(
          "noise_variance",
          [](const ppca::PPCA& self) { return self.noise_variance(); })

      .def_property_readonly("explained_variance",
                             [](const ppca::PPCA& self) {
                               const arma::vec& v = self.explained_variance();
                               return carma::col_to_arr(v);
                             })

      .def_property_readonly("explained_variance_ratio",
                             [](const ppca::PPCA& self) {
                               const arma::vec& v =
                                   self.explained_variance_ratio();
                               return carma::col_to_arr(v);
                             })

      .def_property_readonly(
          "n_samples", [](const ppca::PPCA& self) { return self.n_samples(); })

      .def_property_readonly(
          "n_features_in",
          [](const ppca::PPCA& self) { return self.n_features_in(); })

      .def_property_readonly("n_components", [](const ppca::PPCA& self) {
        return self.n_components();
      });
}
