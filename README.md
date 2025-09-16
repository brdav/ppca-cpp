
# lib-ppca

Probabilistic PCA (PPCA) with missing-data support — fast C++ core, clean Python API.

<img src="./docs/teaser.jpg" width="720" alt="teaser"/>

## Overview

**lib-ppca** implements Probabilistic Principal Component Analysis (PPCA) as described by Tipping & Bishop (1999), with a focus on speed, usability, and robust handling of missing data. The core is written in C++ (Armadillo + CARMA + pybind11), exposed via a simple Python interface.

### Key Features

- **Handles missing values natively:** No need for manual imputation—just use `np.nan` for missing entries.
- **Familiar API:** Drop-in replacement for PCA with attributes like `components_`, `explained_variance_`, etc.
- **Probabilistic modeling:** Compute log-likelihoods, posterior latent variables, and reconstructions.
- **Fast and scalable:** Optimized C++ backend for large datasets.
- **Flexible:** Supports both batch and online (mini-batch) EM.

### Main Functionalities

- Fit a PPCA model to data, even with missing values.
- Transform data to a lower-dimensional latent space.
- Reconstruct or impute missing values from the latent space.
- Compute data log-likelihood and evaluate model fit.
- Access principal axes, explained variance, and more.

## Quick Start

```bash
pip install lib-ppca
```

Usage example:

```python
import numpy as np
from lib_ppca import PPCA

X_train = np.random.randn(600, 10)
X_train[::7, 3] = np.nan  # introduce missing values
X_test = np.random.randn(100, 10)
X_test[::7, 2] = np.nan  # introduce missing values

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

For more examples and PPCA functionalities see the [demo notebook](https://github.com/brdav/lib-ppca/notebooks/demo.ipynb).

## Installation from Source

Install from a fresh clone (initialise the CARMA submodule first – otherwise the build will fail):

```bash
git clone https://github.com/brdav/lib-ppca.git
cd lib-ppca
git submodule update --init --recursive   # pulls extern/carma
pip install .                             # build and install
```

Use an editable install for development:

```bash
git clone https://github.com/brdav/lib-ppca.git
cd lib-ppca
git submodule update --init --recursive   # pulls extern/carma
pip install -e '.[dev]'                   # editable install
pre-commit install                        # register pre-commit hooks
```

## Internals

PPCA uses an Expectation-Maximization (EM) algorithm to learn parameters through maximum likelihood estimation. For details see the reference paper listed below. The equations for the EM algorithm in the presence of missing values are listed in [EQUATIONS.md](https://github.com/brdav/lib-ppca/EQUATIONS.md).

## Citing

If you use this code academically, cite the original PPCA paper:

* M. Tipping & C. Bishop. Probabilistic Principal Component Analysis. JRSS B, 1999.

You may also reference the repository URL.

## License

MIT License — see `LICENSE`.

---
Questions or requests? Open an issue.
