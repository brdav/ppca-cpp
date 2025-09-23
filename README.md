
# lib-ppca

[![Build Wheels](https://img.shields.io/github/actions/workflow/status/brdav/lib-ppca/.github/workflows/build.yml?branch=main)](https://github.com/brdav/lib-ppca/actions)
[![PyPI version](https://img.shields.io/pypi/v/lib-ppca.svg)](https://pypi.org/project/lib-ppca)
[![Python Versions](https://img.shields.io/pypi/pyversions/lib-ppca.svg)](https://pypi.org/project/lib-ppca)
[![License](https://img.shields.io/github/license/brdav/lib-ppca.svg)](LICENSE)

Probabilistic PCA (PPCA) with missing-data support — fast C++ core, clean Python API.

<img src="https://raw.githubusercontent.com/brdav/lib-ppca/main/docs/teaser.jpg" width="500" alt="teaser"/>

## Overview

**lib-ppca** implements Probabilistic Principal Component Analysis (PPCA) as described by Tipping & Bishop (1999), with a focus on speed, usability, and robust handling of missing data. The core is written in C++ (Armadillo + CARMA + pybind11), exposed via a simple Python interface.

### Key Features

- **Handles missing values natively:** No need for manual imputation—just use `np.nan` for missing entries.
- **Familiar API:** Drop-in replacement for scikit-learn PCA with attributes like `components_`, `explained_variance_`, etc.
- **Probabilistic modeling:** Compute log-likelihoods, posterior latent variable distributions, multiple imputations, and more.
- **Fast and scalable:** Optimized C++ backend for large datasets.
- **Flexible:** Supports both batch and online (mini-batch) EM.

## Quick Start

```bash
pip install lib-ppca
```

Note: pre-built wheels are produced only for Linux and macOS (CI builds target ubuntu-latest and macos-latest). On other platforms (e.g. Windows) you will need to build from source (see further below).

Usage example:

```python
import numpy as np
from lib_ppca import PPCA

X_train = np.random.randn(600, 10)
X_train[::7, 3] = np.nan                  # missing values
X_test = np.random.randn(100, 10)
X_test[::7, 2] = np.nan                   # missing values

model = PPCA(n_components=3, batch_size=200)
model.fit(X_train)

mZ, covZ = model.posterior_latent(X_test) # latent representation
mX, covX = model.likelihood(mZ)           # reconstruction
ll = model.score(X_test)                  # mean log likelihood

# multiple imputation
X_imputed = model.sample_missing(X_test, n_draws=5)

print("Components shape:", model.components_.shape)
print("Explained variance ratio:", model.explained_variance_ratio_)
```

For more examples and PPCA functionalities see the [demo notebook](https://github.com/brdav/lib-ppca/blob/main/notebooks/demo.ipynb).

## Installation from Source

Minimum requirements

* CMake >= 3.18
* Python >= 3.9 (+ development headers)
* C++20-capable compiler (clang on macOS, gcc on Linux, MSVC on Windows)
* BLAS/LAPACK implementation (OpenBLAS, MKL, or Accelerate on macOS)
* git (to fetch submodules)
* Network access (CMake will download Armadillo into extern by default) or provide extern/armadillo-\<version\>/ or a system Armadillo install

Quick install (fresh clone)

```bash
git clone https://github.com/brdav/lib-ppca.git
cd lib-ppca
git submodule update --init --recursive   # ensure extern/carma is present
python -m pip install .                   # build and install
```

Editable install for development

```bash
git clone https://github.com/brdav/lib-ppca.git
cd lib-ppca
git submodule update --init --recursive
python -m pip install -e '.[dev]'         # editable install
pre-commit install                        # optional: register hooks
```

### Windows

Builds on Windows are untested in CI. You can attempt a Windows build but expect manual steps:

* Install Visual Studio with the C++ toolchain (or a supported MinGW) and CMake.
* Provide BLAS/LAPACK (OpenBLAS, MKL) and point CMake to their libraries.
* Provide Armadillo sources (extern/armadillo-\<version\>/) or install Armadillo system-wide and set ARMADILLO_ROOT_DIR/use find_package.
* You may need to adjust linker/rpath settings or prefer static linking to avoid missing DLLs at runtime.

## Internals

PPCA uses an Expectation-Maximization (EM) algorithm to learn parameters through maximum likelihood estimation. For details see the reference paper listed below. The equations for the EM algorithm in the presence of missing values are listed in [EQUATIONS.md](https://github.com/brdav/lib-ppca/blob/main/docs/EQUATIONS.md).

## Citing

If you use this code academically, cite the original PPCA paper:

* M. Tipping & C. Bishop. Probabilistic Principal Component Analysis. JRSS B, 1999.

You may also reference the library name or URL.

## License

MIT License — see `LICENSE`.

---
Questions or requests? Open an issue.
