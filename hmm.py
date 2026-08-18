"""
Gaussian Hidden Markov Model — pure NumPy/SciPy implementation.
================================================================
Drop-in replacement for hmmlearn.hmm.GaussianHMM, using only numpy and
scipy.special.logsumexp.  No compiled C extensions, so it runs on any
Python version including 3.14 on Streamlit Cloud.

Implements:
    - Baum–Welch (EM) for fitting
    - Viterbi decoding  (predict)
    - Forward–backward posterior probabilities (predict_proba)
    - Log-likelihood scoring (score)

Reference: Rabiner, L.R. (1989). "A Tutorial on Hidden Markov Models
and Selected Applications in Speech Recognition."
"""

import numpy as np
from scipy.special import logsumexp


class GaussianHMM:
    """
    Gaussian-emission Hidden Markov Model with full covariance matrices.

    Parameters
    ----------
    n_components : int
        Number of hidden states (regimes).
    n_iter : int
        Maximum EM iterations.
    tol : float
        Convergence threshold on log-likelihood improvement.
    random_state : int or None
        Seed for reproducibility.
    covariance_type : str
        Accepted for API compatibility; only 'full' is implemented.
    """

    def __init__(
        self,
        n_components: int = 3,
        n_iter: int = 100,
        tol: float = 1e-2,
        random_state: int | None = None,
        covariance_type: str = "full",
    ):
        self.n_components = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.random_state = random_state
        self.covariance_type = covariance_type

    # ── initialisation ───────────────────────────────────────────────

    def _init_params(self, X: np.ndarray) -> None:
        n_samples, n_features = X.shape
        rng = np.random.RandomState(self.random_state)

        # Uniform start probabilities
        self.startprob_ = np.ones(self.n_components) / self.n_components

        # Slightly diagonal-heavy transition matrix
        self.transmat_ = np.full(
            (self.n_components, self.n_components), 0.1 / self.n_components
        )
        np.fill_diagonal(self.transmat_, 0.7)
        self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

        # Pick random data points as initial means
        idx = rng.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[idx].copy()

        # Identity-ish initial covariances
        self.covars_ = np.array(
            [np.eye(n_features) for _ in range(self.n_components)]
        )

    # ── emission probabilities ───────────────────────────────────────

    def _log_emission_probs(self, X: np.ndarray) -> np.ndarray:
        """Log N(x | μ_k, Σ_k) for every (sample, component) pair."""
        n_samples, n_features = X.shape
        log_probs = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            diff = X - self.means_[k]
            cov = self.covars_[k] + 1e-6 * np.eye(n_features)

            try:
                L = np.linalg.cholesky(cov)
                log_det = 2.0 * np.sum(np.log(np.diag(L)))
                solved = np.linalg.solve(L, diff.T).T
                mahal = np.sum(solved ** 2, axis=1)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(cov)
                eigvals = np.maximum(eigvals, 1e-6)
                log_det = np.sum(np.log(eigvals))
                mahal = np.sum((diff @ eigvecs) ** 2 / eigvals, axis=1)

            log_probs[:, k] = -0.5 * (
                n_features * np.log(2.0 * np.pi) + log_det + mahal
            )

        return log_probs

    # ── forward / backward ───────────────────────────────────────────

    def _forward(self, log_emis: np.ndarray) -> np.ndarray:
        T, K = log_emis.shape
        log_alpha = np.full((T, K), -np.inf)
        log_trans = np.log(self.transmat_ + 1e-300)

        log_alpha[0] = np.log(self.startprob_ + 1e-300) + log_emis[0]

        for t in range(1, T):
            for j in range(K):
                log_alpha[t, j] = (
                    logsumexp(log_alpha[t - 1] + log_trans[:, j])
                    + log_emis[t, j]
                )

        return log_alpha

    def _backward(self, log_emis: np.ndarray) -> np.ndarray:
        T, K = log_emis.shape
        log_beta = np.zeros((T, K))
        log_trans = np.log(self.transmat_ + 1e-300)

        for t in range(T - 2, -1, -1):
            for j in range(K):
                log_beta[t, j] = logsumexp(
                    log_trans[j] + log_emis[t + 1] + log_beta[t + 1]
                )

        return log_beta

    # ── fitting (Baum–Welch / EM) ────────────────────────────────────

    def fit(self, X: np.ndarray) -> "GaussianHMM":
        self._init_params(X)
        n_samples, n_features = X.shape
        prev_ll = -np.inf

        for _ in range(self.n_iter):
            # ── E-step ──
            log_emis = self._log_emission_probs(X)
            log_alpha = self._forward(log_emis)
            log_beta = self._backward(log_emis)

            log_ll = logsumexp(log_alpha[-1])
            if abs(log_ll - prev_ll) < self.tol:
                break
            prev_ll = log_ll

            # Posterior state probabilities  γ(t, k)
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)

            # Posterior transition probabilities  ξ(t, i, j)
            log_trans = np.log(self.transmat_ + 1e-300)
            xi = np.zeros((n_samples - 1, self.n_components, self.n_components))
            for t in range(n_samples - 1):
                for i in range(self.n_components):
                    for j in range(self.n_components):
                        xi[t, i, j] = (
                            log_alpha[t, i]
                            + log_trans[i, j]
                            + log_emis[t + 1, j]
                            + log_beta[t + 1, j]
                        )
                xi[t] -= logsumexp(xi[t].ravel())
            xi = np.exp(xi)

            # ── M-step ──
            self.startprob_ = gamma[0]
            self.startprob_ /= self.startprob_.sum()

            self.transmat_ = xi.sum(axis=0)
            self.transmat_ /= self.transmat_.sum(axis=1, keepdims=True)

            for k in range(self.n_components):
                w = gamma[:, k]
                total = w.sum()
                if total < 1e-10:
                    continue
                self.means_[k] = np.average(X, axis=0, weights=w)
                diff = X - self.means_[k]
                self.covars_[k] = (
                    diff.T @ (diff * w[:, np.newaxis])
                ) / total
                self.covars_[k] += 1e-3 * np.eye(n_features)

        self._log_likelihood = log_ll
        return self

    # ── decoding & scoring ───────────────────────────────────────────

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Viterbi decoding — most likely state sequence."""
        log_emis = self._log_emission_probs(X)
        T, K = log_emis.shape
        log_trans = np.log(self.transmat_ + 1e-300)

        V = np.full((T, K), -np.inf)
        bp = np.zeros((T, K), dtype=int)
        V[0] = np.log(self.startprob_ + 1e-300) + log_emis[0]

        for t in range(1, T):
            for j in range(K):
                scores = V[t - 1] + log_trans[:, j]
                bp[t, j] = np.argmax(scores)
                V[t, j] = scores[bp[t, j]] + log_emis[t, j]

        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(V[-1])
        for t in range(T - 2, -1, -1):
            states[t] = bp[t + 1, states[t + 1]]
        return states

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Posterior state probabilities via forward–backward."""
        log_emis = self._log_emission_probs(X)
        log_alpha = self._forward(log_emis)
        log_beta = self._backward(log_emis)

        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def score(self, X: np.ndarray) -> float:
        """Log-likelihood of the observation sequence."""
        log_emis = self._log_emission_probs(X)
        log_alpha = self._forward(log_emis)
        return float(logsumexp(log_alpha[-1]))
