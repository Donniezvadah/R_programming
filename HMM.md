
---
title: "Hidden Markov Models for Time Series"
author: "Donald Zvada"
date: ""
---

# Hidden Markov Models for Time Series

## 1. Quick reference: core objects

| Symbol (or name) | Meaning / definition |
|---|---|
| `m` (or `\mState`) | Number of states in the latent (hidden) Markov chain (finite state-space). |
| `\{C_t: t\in\mathbb{N}\}` | Latent (hidden) **Markov chain** (parameter process). `C_t \in \{1,2,\dots,m\}`. Markov property: `P(C_{t+1} | C_t, ... , C_1) = P(C_{t+1} | C_t)`. |
| `\{X_t: t\in\mathbb{N}\}` | Observed process (state-dependent process). `P(X_t | X_{1:t-1}, C_{1:t}) = P(X_t | C_t)`. |
| `\pi` or `u(t)` | Row vector of marginal state probabilities at time `t`: `u(t) = ( P(C_t=1), ..., P(C_t=m) )`. The initial distribution `u(1)` is often `\pi`. |
| `\delta` | Stationary distribution of the Markov chain (row vector) satisfying `\delta \Gamma = \delta` and `\delta \mathbf{1}^\top = 1`. |
| `\Gamma` | One-step transition probability matrix (t.p.m.), `\Gamma = (\gamma_{ij})_{i,j=1}^m`. |
| `\gamma_{ij}` | Transition probability from latent state `i` to state `j`. |
| `p_i(x)` | State-dependent pmf or density for `X_t` when `C_t = i`: `p_i(x) = P(X_t = x | C_t = i)`. |
| `P(x)` | `m x m` diagonal matrix with `i`th diagonal `p_i(x)` (or density). |
| `\mathbf{1}` | Column vector of ones (dimension m). |
| `\ell(\theta)` | Log-likelihood of parameters `\theta` given observed data `x_{1:T}`. |
| `\lambda, \lambda_i` | Often used for Poisson means in Poisson–HMMs. |
| `\eta_i, \tau_i` | Working (reparameterized) parameters for unconstrained optimisation. |
| `f_{ij}` | Transition count: number of observed transitions from state `i` to `j`. |
| `u(t)` | `u(t) = u(1) \Gamma^{t-1}`. |
| `v, V` | Numeric labels and diagonal matrix used in covariance calculations. |
| `\Omega, U` | Eigen-decomposition matrices for `\Gamma = U \Omega U^{-1}`. |

---

## 2. Key formulas and definitions

### 2.1 Markov chain / transition matrix

A finite-state homogeneous Markov chain `{C_t}` with `m` states has one-step transition probability matrix
\[
\Gamma = (\gamma_{ij})_{i,j=1}^m,\qquad
\gamma_{ij} = \Pr(C_{t+1} = j \mid C_t = i),
\]
with each row summing to `1`. Chapman–Kolmogorov gives `\Gamma^{(t+u)} = \Gamma^{(t)} \Gamma^{(u)}`.

### 2.2 Stationary distribution

`\delta` is stationary if
\[
\delta \Gamma = \delta, \qquad \delta \mathbf{1}^\top = 1.
\]

### 2.3 Hidden Markov model (basic HMM)

The HMM consists of latent states `C_t` forming a Markov chain and observations `X_t` with
\[
\Pr(C_t \mid C_{1:t-1}) = \Pr(C_t \mid C_{t-1}),\qquad
\Pr(X_t \mid X_{1:t-1}, C_{1:t}) = \Pr(X_t \mid C_t).
\]

### 2.4 Marginal distributions of observations

Define `P(x) := \mathrm{diag}(p_1(x),\dots,p_m(x))`. Then for a homogeneous chain with initial `u(1)`,

\[
\Pr(X_t = x) = u(1) \Gamma^{t-1} P(x) \mathbf{1}.
\]

If stationary:

\[
\Pr(X_t = x) = \delta P(x) \mathbf{1}.
\]

### 2.5 Joint distribution of \(X_{1:T}\)

\[
\Pr(X_{1:T} = x_{1:T}) = u(1) P(x_1) \Gamma P(x_2) \Gamma \cdots \Gamma P(x_T) \mathbf{1}.
\]

### 2.6 Likelihood and log-likelihood

\[
\ell(\theta) = \log\big( u(1) P(x_1) \Gamma P(x_2) \cdots P(x_T) \mathbf{1} \big).
\]

### 2.7 Mixture notation

\[
f(x) = \sum_{i=1}^m \delta_i p_i(x), \qquad \sum_{i=1}^m \delta_i = 1.
\]

### 2.8 Reparametrisation

\[
\eta_i = \log(\lambda_i),\qquad
\tau_i = \log\!\Big(\frac{\delta_i}{1-\sum_{j=2}^m \delta_j}\Big).
\]

### 2.9 Empirical ML for \(\Gamma\) (states observed)

\[
\widehat{\gamma}_{ij} = \frac{f_{ij}}{\sum_{k=1}^m f_{ik}}.
\]

---

## 3. Terminology / short glossary

- **Basic HMM:** univariate observations, homogeneous chain.  
- **State-dependent distribution:** `p_i(x)`.  
- **t.p.m.:** transition probability matrix `\Gamma`.  
- **Decoding:** Viterbi (global) or local decoding.  
- **EM algorithm:** Expectation–Maximization for latent variable ML.  
- **Poisson–HMM:** HMM with Poisson state-dependent distributions.

---

## 4. Forward, backward and decoding

### 4.1 Forward recursion

\[
\alpha_t = \delta P(x_1) \prod_{s=2}^t \Gamma P(x_s), \quad
\alpha_t = \alpha_{t-1} \Gamma P(x_t).
\]

### 4.2 Backward recursion

\[
\beta_t = \Gamma P(x_{t+1}) \Gamma P(x_{t+2}) \cdots \Gamma P(x_T) \mathbf{1}, \quad
\beta_T = \mathbf{1}.
\]

### 4.3 Local decoding

\[
\Pr(C_t = i \mid X^{(T)} = x^{(T)}) = \frac{\alpha_t(i) \beta_t(i)}{L_T}.
\]

### 4.4 Viterbi

\[
\xi_{t,j} = \left( \max_i (\xi_{t-1,i} \gamma_{ij}) \right) p_j(x_t).
\]

---

## 5. R implementations — Poisson HMM (parameter transforms & likelihood)

Below are R functions implementing transforms and the scaled negative log-likelihood for Poisson HMMs.

### 5.1 Transform natural -> working (pn2pw)

```r
pois.HMM.pn2pw <- function(m, lambda, gamma, delta = NULL, stationary = TRUE) {
  tlambda <- log(lambda)
  if (m == 1) return(tlambda)
  foo <- log(gamma / diag(gamma))
  tgamma <- as.vector(foo[!diag(m)])
  if (stationary) {
    tdelta <- NULL
  } else {
    tdelta <- log(delta[-1] / delta[1])
  }
  parvect <- c(tlambda, tgamma, tdelta)
  return(parvect)
}
```

### 5.2 Transform working -> natural (pw2pn)

```r
pois.HMM.pw2pn <- function(m, parvect, stationary = TRUE) {
  lambda <- exp(parvect[1:m])
  gamma <- diag(m)
  if (m == 1) return(list(lambda = lambda, gamma = gamma, delta = 1))
  gamma[!gamma] <- exp(parvect[(m+1):(m*m)])
  gamma <- gamma / apply(gamma, 1, sum)
  if (stationary) {
    delta <- solve(t(diag(m) - gamma + 1), rep(1, m))
  } else {
    foo <- c(1, exp(parvect[(m*m+1):(m*m+m-1)]))
    delta <- foo / sum(foo)
  }
  return(list(lambda = lambda, gamma = gamma, delta = delta))
}
```

### 5.3 Minus log-likelihood with scaling (mllk)

```r
pois.HMM.mllk <- function(parvect, x, m, stationary = TRUE, ...) {
  if (m == 1) return(-sum(dpois(x, exp(parvect), log = TRUE)))
  n <- length(x)
  pn <- pois.HMM.pw2pn(m, parvect, stationary = stationary)
  P1 <- dpois(x[1], pn$lambda)
  foo <- pn$delta * P1
  sumfoo <- sum(foo)
  lscale <- log(sumfoo)
  foo <- foo / sumfoo
  if (n >= 2) {
    for (i in 2:n) {
      if (!is.na(x[i])) {
        P <- dpois(x[i], pn$lambda)
      } else {
        P <- rep(1, m)
      }
      foo <- (foo %*% pn$gamma) * P
      sumfoo <- sum(foo)
      lscale <- lscale + log(sumfoo)
      foo <- foo / sumfoo
    }
  }
  mllk <- -lscale
  return(mllk)
}
```

---

## 6. Example: fitting a Poisson HMM via optim()

```r
fit_poisson_hmm <- function(x, m = 2, stationary = TRUE) {
  lambda_start <- quantile(x, probs = seq(0, 1, length.out = m + 1))[-1]
  gamma_start <- matrix(0.1, nrow = m, ncol = m)
  diag(gamma_start) <- 0.8
  if (stationary) {
    delta_start <- NULL
  } else {
    delta_start <- rep(1/m, m)
  }
  start_par <- pois.HMM.pn2pw(m, lambda_start, gamma_start, delta = delta_start, stationary = stationary)
  opt <- optim(start_par, fn = pois.HMM.mllk, x = x, m = m, stationary = stationary, method = "BFGS", control = list(maxit = 1000))
  est <- pois.HMM.pw2pn(m, opt$par, stationary = stationary)
  return(list(parvect = opt$par, loglik = -opt$value, est = est, convergence = opt$convergence))
}
```

---

## 7. Practical tips & numerical issues

- Use scaling in forward recursion to avoid underflow.  
- Reparameterise constrained parameters via log / logit transforms.  
- Consider EM (Baum–Welch) as an alternative to direct optimisation.  
- Check identifiability and use sensible starting values.  
- Use AIC/BIC and predictive checks for model selection.  
- For inference, obtain standard errors via Hessian or bootstrap.

---

## 8. Decoding & inference post-fit

- Compute scaled forward/backward to get local state probabilities.  
- Run Viterbi (log-domain) for global decoding.  
- Compute one-step-ahead predictive distributions using estimated parameters.

---

## 9. References

- Zucchini, W., MacDonald, I. L., & Langrock, R. (2016). *Hidden Markov Models for Time Series: An Introduction Using R.* Chapman & Hall/CRC.  
- Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications. *Proceedings of the IEEE*.  
- Bishop, C. M. (2006). *Pattern Recognition and Machine Learning.* Springer.

