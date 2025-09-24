# EM for Probabilistic PCA with Missing Values

Many open-source implementations of PCA with missing values rely on the EM imputation algorithm described by Roweis [3, Sec. 3.2] in the noise-free PCA setting. While Roweis also briefly mentions the probabilistic PCA (PPCA) model in his paper, the focus there is not on handling missing values within the full probabilistic framework. As a result, most implementations based on [3] do not provide a direct way to, for example, project new, unseen data with missing entries into the latent space.

In this repository, we instead adopt the full PPCA formulation, following the original derivation of Tipping and Bishop [2] and the treatment in Bishop’s textbook [1, Chapter 12.2]. These works provide a clear derivation of PPCA and its optimization using the EM algorithm. However, the case with missing values is not spelled out in detail. To fill this gap, we summarize the key equations needed to extend PPCA to missing data. We assume the reader is already familiar with [1] and [2].

## Step 1: Handling Missing Values in the Likelihood

For each observation $\mathbf{x}_n$, let:

* $\mathbf{x}_n^\text{o}$ = observed entries

* $\mathbf{x}_n^\text{m}$ = missing entries

Because PPCA assumes an isotropic Gaussian noise model, the likelihood factorizes across dimensions. This means we can integrate out the missing entries directly:

```math
p(\mathbf{X}|\mathbf{\mu}, \mathbf{W}, \sigma^2) = \prod_{n=1}^N \int p(\mathbf{z}_n) p(\mathbf{x}_n | \mathbf{z}_n, \mathbf{\mu}, \mathbf{W}, \sigma^2) \text{d}\mathbf{z}_n
```

which simplifies to

```math
= \prod_{n=1}^N p(\mathbf{x}_n^\text{o}|\mathbf{\mu}, \mathbf{W}, \sigma^2)
```

This tells us that only the observed entries contribute to the marginal likelihood.

From this expression, the maximum-likelihood estimate of the mean vector $\mu$ follows directly as the per-feature average over the observed entries:

```math
\mathbf{\mu}_{\text{ML}} = \frac{1}{\sum_{m=1}^N \iota_{mi}} \sum_{n=1}^N\iota_{ni} x_{ni}
```

where $\iota_{ni} = 1$ if $x_{ni}$ is observed and $0$ otherwise.
Estimating the weight matrix $W$ and noise variance $\sigma^2$ is more involved and requires EM.

## Step 2: The EM Algorithm

The EM algorithm alternates between:

* E-step: Estimate the posterior distribution of the latent variables $\mathbf{z}_n$, given the current parameters and observed data.

* M-step: Maximize the expected complete-data log-likelihood with respect to the parameters $W$ and $\sigma^2$.

### E-step: Posterior of the Latents

For each observation $\mathbf{x}_n$, define:

* $\mathbf{y}_n$: observed entries of $\mathbf{x}_n - \mu$

* $W_n$: rows of $W$ corresponding to the observed dimensions

* $M_n = W_n^\top W_n + \sigma^2 I$

The posterior mean and covariance of $\mathbf{z}_n$ are:

```math
\mathbb{E} [\mathbf{z}_n] = \mathbf{M}_n^{-1} \mathbf{W}_n^\text{T} \mathbf{y}_n
```

```math
\mathbb{E} [\mathbf{z}_n \mathbf{z}_n^\text{T}] = \sigma^2 \mathbf{M}_n^{-1} + \mathbb{E}[\mathbf{z}_n] \mathbb{E}[\mathbf{z}_n]^\text{T}
```

These expectations serve as the sufficient statistics for the M-step.

### M-step: Parameter Updates

Given the posterior expectations, we maximize the complete-data log likelihood to update $W$ and $\sigma^2$.

Update for the weights:

```math
\mathbf{W}_\text{new} = [\sum_{n=1}^N \mathbf{y}_n \mathbb{E}[\mathbf{z}_n]^\text{T}] [\sum_{n=1}^N \mathbb{E}[\mathbf{z}_n \mathbf{z}_n^\text{T}]]^{-1}
```

Update for the noise variance:

```math
\sigma^2_\text{new} = \frac{1}{\sum_{n=1}^N \sum_{i=1}^D \iota_{ni}} \sum_{n=1}^N \sum_{i=1}^D \iota_{ni} \{ (x_{ni} - \mu_{\text{ML}, i})^2 - 2 \mathbb{E}[\mathbf{z}_n]^\text{T} \mathbf{w}_i^\text{new} (x_{ni} - \mu_{\text{ML}, i}) + \text{Tr}(\mathbb{E}[\mathbf{z}_n \mathbf{z}_n^\text{T}] \mathbf{w}_i^\text{new} (\mathbf{w}_i^\text{new})^\text{T}) \}
```

Here $w_i^\text{new}$ is the $i$-th row of the updated weight matrix.

## Summary

* Missing entries are naturally handled in PPCA by marginalizing them out.

* The EM algorithm can then be applied almost unchanged, except that only observed entries contribute in both the E-step (through $W_n$ and $\mathbf{y}_n$) and the M-step (through the indicator variables $\iota_{ni}$).

* This yields a principled way to deal with missing data in PPCA, while still allowing consistent inference on new samples with missing entries.

## References

[1] Bishop, C. M. Pattern Recognition and Machine Learning. Springer, 2006.

[2] Tipping, M. E. and Bishop, C. M. Probabilistic Principal Component Analysis. JRSS B, 1999.

[3] Roweis, S. EM algorithms for PCA and SPCA. NIPS 10, 1998.
