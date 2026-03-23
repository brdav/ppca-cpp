// Copyright 2025 brdav

// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <armadillo>
#include <string>

#include "ppca.hpp"

namespace py = pybind11;

namespace {

arma::mat numpy_to_arma_mat(const py::array& array, const char* name) {
  py::array_t<double, py::array::forcecast> arr(array);
  if (arr.ndim() != 2) {
    throw py::value_error(std::string(name) + " must be a 2D array");
  }

  const auto view = arr.unchecked<2>();
  arma::mat out(view.shape(0), view.shape(1));
  for (py::ssize_t r = 0; r < view.shape(0); ++r) {
    for (py::ssize_t c = 0; c < view.shape(1); ++c) {
      out(static_cast<arma::uword>(r), static_cast<arma::uword>(c)) =
          view(r, c);
    }
  }
  return out;
}

arma::vec numpy_to_arma_vec(const py::array& array, const char* name) {
  py::array_t<double, py::array::forcecast> arr(array);

  if (arr.ndim() == 1) {
    const auto view = arr.unchecked<1>();
    arma::vec out(view.shape(0));
    for (py::ssize_t i = 0; i < view.shape(0); ++i) {
      out(static_cast<arma::uword>(i)) = view(i);
    }
    return out;
  }

  if (arr.ndim() == 2 && (arr.shape(0) == 1 || arr.shape(1) == 1)) {
    const auto view = arr.unchecked<2>();
    const py::ssize_t n = view.shape(0) * view.shape(1);
    arma::vec out(n);
    py::ssize_t i = 0;
    for (py::ssize_t r = 0; r < view.shape(0); ++r) {
      for (py::ssize_t c = 0; c < view.shape(1); ++c) {
        out(static_cast<arma::uword>(i++)) = view(r, c);
      }
    }
    return out;
  }

  throw py::value_error(std::string(name) +
                        " must be a 1D array or a 2D single-column/row array");
}

py::array_t<double> arma_mat_to_numpy(const arma::mat& mat) {
  py::array_t<double> out({static_cast<py::ssize_t>(mat.n_rows),
                           static_cast<py::ssize_t>(mat.n_cols)});
  auto view = out.mutable_unchecked<2>();
  for (arma::uword r = 0; r < mat.n_rows; ++r) {
    for (arma::uword c = 0; c < mat.n_cols; ++c) {
      view(static_cast<py::ssize_t>(r), static_cast<py::ssize_t>(c)) =
          mat(r, c);
    }
  }
  return out;
}

py::array_t<double> arma_vec_to_numpy_col(const arma::vec& vec) {
  py::array_t<double> out(
      {static_cast<py::ssize_t>(vec.n_elem), static_cast<py::ssize_t>(1)});
  auto view = out.mutable_unchecked<2>();
  for (arma::uword i = 0; i < vec.n_elem; ++i) {
    view(static_cast<py::ssize_t>(i), 0) = vec(i);
  }
  return out;
}

py::array_t<double> arma_cube_to_numpy(const arma::cube& cube) {
  py::array_t<double> out({static_cast<py::ssize_t>(cube.n_rows),
                           static_cast<py::ssize_t>(cube.n_cols),
                           static_cast<py::ssize_t>(cube.n_slices)});
  auto view = out.mutable_unchecked<3>();
  for (arma::uword s = 0; s < cube.n_slices; ++s) {
    for (arma::uword c = 0; c < cube.n_cols; ++c) {
      for (arma::uword r = 0; r < cube.n_rows; ++r) {
        view(static_cast<py::ssize_t>(r), static_cast<py::ssize_t>(c),
             static_cast<py::ssize_t>(s)) = cube(r, c, s);
      }
    }
  }
  return out;
}

}  // namespace

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
            arma::mat x_mat = numpy_to_arma_mat(X, "X");
            self.fit(x_mat);
            return &self;
          },
          py::return_value_policy::reference_internal)

      .def("score_samples",
           [](const ppca::PPCA& self, py::array X) {
             arma::mat x_mat = numpy_to_arma_mat(X, "X");
             const arma::vec& ll_vec = self.score_samples(x_mat);
             return arma_vec_to_numpy_col(ll_vec);
           })

      .def("score",
           [](const ppca::PPCA& self, py::array X) {
             arma::mat x_mat = numpy_to_arma_mat(X, "X");
             return self.score(x_mat);
           })

      .def("get_covariance",
           [](const ppca::PPCA& self) {
             arma::mat cov = self.get_covariance();
             return arma_mat_to_numpy(cov);
           })

      .def("get_precision",
           [](const ppca::PPCA& self) {
             arma::mat prec = self.get_precision();
             return arma_mat_to_numpy(prec);
           })

      .def(
          "posterior_latent",
          [](const ppca::PPCA& self, py::array X) {
            arma::mat x_mat = numpy_to_arma_mat(X, "X");
            auto [ez, cz] = self.posterior_latent(x_mat);
            return py::make_tuple(arma_mat_to_numpy(ez),
                                  arma_cube_to_numpy(cz));
          },
          py::arg("X"))

      .def(
          "likelihood",
          [](const ppca::PPCA& self, py::array Z) {
            arma::mat z_mat = numpy_to_arma_mat(Z, "Z");
            auto [x_hat, cov_mat] = self.likelihood(z_mat);
            return py::make_tuple(arma_mat_to_numpy(x_hat),
                                  arma_cube_to_numpy(cov_mat));
          },
          py::arg("Z"))

      .def(
          "impute_missing",
          [](const ppca::PPCA& self, py::array X) {
            arma::mat x_mat = numpy_to_arma_mat(X, "X");
            auto [x_hat, cov_mat] = self.impute_missing(x_mat);
            return py::make_tuple(arma_mat_to_numpy(x_hat),
                                  arma_cube_to_numpy(cov_mat));
          },
          py::arg("X"))

      .def(
          "sample_posterior_latent",
          [](const ppca::PPCA& self, py::array X,
             std::size_t n_draws /* = 1 */) {
            arma::mat x_mat = numpy_to_arma_mat(X, "X");
            arma::cube z_tilde = self.sample_posterior_latent(x_mat, n_draws);
            return arma_cube_to_numpy(z_tilde);
          },
          py::arg("X"), py::arg("n_draws"))

      .def(
          "sample_likelihood",
          [](const ppca::PPCA& self, py::array Z,
             std::size_t n_draws /* = 1 */) {
            arma::mat z_mat = numpy_to_arma_mat(Z, "Z");
            arma::cube x_tilde = self.sample_likelihood(z_mat, n_draws);
            return arma_cube_to_numpy(x_tilde);
          },
          py::arg("Z"), py::arg("n_draws"))

      .def(
          "sample_missing",
          [](const ppca::PPCA& self, py::array X,
             std::size_t n_draws /* = 1 */) {
            arma::mat x_mat = numpy_to_arma_mat(X, "X");
            arma::cube x_tilde = self.sample_missing(x_mat, n_draws);
            return arma_cube_to_numpy(x_tilde);
          },
          py::arg("X"), py::arg("n_draws"))

      .def(
          "lmmse_reconstruction",
          [](const ppca::PPCA& self, py::array Ez) {
            arma::mat ez_mat = numpy_to_arma_mat(Ez, "Ez");
            arma::mat x_hat = self.lmmse_reconstruction(ez_mat);
            return arma_mat_to_numpy(x_hat);
          },
          py::arg("Ez"))

      .def("get_params",
           [](const ppca::PPCA& self) {
             auto p = self.get_params();
             py::dict d;
             d["components"] = arma_mat_to_numpy(p.components);
             d["mean"] = arma_vec_to_numpy_col(p.mean);
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

            arma::mat components_mat = numpy_to_arma_mat(
                py::cast<py::array>(components_obj), "params['components']");
            arma::vec mean_vec = numpy_to_arma_vec(
                py::cast<py::array>(mean_obj), "params['mean']");

            double noise_variance = py::cast<double>(nv_obj);

            ppca::PPCA::Params p{components_mat, mean_vec, noise_variance};
            self.set_params(p);
            return &self;
          },
          py::arg("params"), py::return_value_policy::reference_internal)

      .def_property_readonly("components",
                             [](const ppca::PPCA& self) {
                               const arma::mat& comps = self.components();
                               return arma_mat_to_numpy(comps);
                             })

      .def_property_readonly("mean",
                             [](const ppca::PPCA& self) {
                               const arma::vec& mean_vec = self.mean();
                               return arma_vec_to_numpy_col(mean_vec);
                             })

      .def_property_readonly(
          "noise_variance",
          [](const ppca::PPCA& self) { return self.noise_variance(); })

      .def_property_readonly("explained_variance",
                             [](const ppca::PPCA& self) {
                               const arma::vec& v = self.explained_variance();
                               return arma_vec_to_numpy_col(v);
                             })

      .def_property_readonly("explained_variance_ratio",
                             [](const ppca::PPCA& self) {
                               const arma::vec& v =
                                   self.explained_variance_ratio();
                               return arma_vec_to_numpy_col(v);
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
