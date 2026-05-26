# =====================
# CONSTANTS DEFINITIONS
# =====================
_PRECISION = 1e-8
_LINE = "_" * 9
_INDENTATION = " " * 9

# Multiplier used to compute nu (threshold on duality gap)
_ACCURACY_MUL = 0.5

# Parameters for learning rate shrinking
_REGRET_CHECK_START_T = 5
_REGRET_CHECK_INCREASE_T = 1.6
_SHRINK_REGRET = 0.8
_SHRINK_ETA = 0.8

# Minimum number of iterations before termination is considered
_MIN_ITER = 5

# =====================================
# LAGRANGIAN CLASS AND HELPER CLASSES
# =====================================

import logging
from collections.abc import Callable
from time import time
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn import clone
from sklearn.dummy import DummyClassifier
from sklearn.base import BaseEstimator, MetaEstimatorMixin
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted

from fairlearn.reductions._moments import ClassificationMoment
from fairlearn.reductions._moments.moment import Moment

logger = logging.getLogger(__name__)

_MESSAGE_BAD_OBJECTIVE = "Objective needs to be of the same type as constraints. Objective is {}, constraints are {}."


# ==============================================================================
# This class wraps any sklearn classifier so we can use it like a function h(X)
# Instead of writing clf.predict(X), we just write h(X)
# ==============================================================================
class _PredictorAsCallable:
    def __init__(self, classifier):
        self._classifier = classifier

    def __call__(self, X):
        return self._classifier.predict(X)


# ===============================================================================
# This class stores the Lagrangian value and its upper and lower bounds,
# along with the constraint violation (gamma) and error.
# It is used to compute the duality gap ν_t as defined in Algorithm 1
# ================================================================================
class _GapResult:
    def __init__(self, L, L_low, L_high, gamma, error):
        self.L = L  # Current Lagrangian value: L( Q̄_t, λ̄_t )
        self.L_low = L_low  # Lower bound on the Lagrangian
        self.L_high = L_high  # Upper bound on the Lagrangian
        self.gamma = gamma  # Constraint violation vector
        self.error = error  # Objective value (e.g., classification error)

    def gap(self):
        # Duality gap: ν_t = max{ L - L_low, L_high - L }
        return max(self.L - self.L_low, self.L_high - self.L)


# ==============================================================================
# Lagrangian object encapsulates the fairness optimization problem:
#   L(h, λ) = error(h) + Σ λ_j (γ_j(h) - b_j)
# It tracks all trained classifiers h, their errors, constraint violations (gamma),
# and λ vectors used for training. It also manages data loading and logging stats.
# ==================================================================================
class _Lagrangian:
    def __init__(
        self,
        *,
        X,
        y,
        estimator,
        constraints: Moment,
        B: float,
        objective: Moment | None = None,
        opt_lambda: bool = True,
        sample_weight_name: str = "sample_weight",
        **kwargs,
    ):
        self.constraints = constraints
        self.constraints.load_data(X, y, **kwargs)
        if objective is None:
            self.obj = self.constraints.default_objective()
        elif objective._moment_type() == constraints._moment_type():
            self.obj = objective
        else:
            raise ValueError(
                _MESSAGE_BAD_OBJECTIVE.format(
                    objective._moment_type(), constraints._moment_type()
                )
            )
        self.obj.load_data(X, y, **kwargs)
        self.estimator = estimator
        self.B = B
        self.opt_lambda = opt_lambda

        self.hs = pd.Series(dtype="object")
        self.predictors = pd.Series(dtype="object")
        self.errors = pd.Series(dtype="float64")
        self.gammas = pd.DataFrame()
        self.lambdas = pd.DataFrame()
        self.n_oracle_calls = 0
        self.oracle_execution_times = []
        self.n_oracle_calls_dummy_returned = 0
        self.last_linprog_n_hs = 0
        self.last_linprog_result = None
        self.sample_weight_name = sample_weight_name

    # ============================================================================
    # This function implements the empirical Lagrangian:
    #   L(Q, λ) = error(Q) + λᵀ (gamma(Q) - bound)
    # where Q is either:
    #   - a single classifier h: X → {0,1,...,K}, or
    #   - a randomized classifier (i.e., a distribution over classifiers)
    #
    # This corresponds to equation (3) from the paper:
    #   min_Q err(Q)  s.t. M θ(Q) ≤ c
    # by using Lagrangian relaxation to combine the objective and constraints.
    # =======================================================================================
    def _eval(self, Q, lambda_vec):
        if callable(Q):
            error = self.obj.gamma(Q).iloc[0]
            gamma = self.constraints.gamma(Q)
        else:
            error = self.errors[Q.index].dot(Q)
            gamma = self.gammas[Q.index].dot(Q)
        if self.opt_lambda:
            lambda_vec = self.constraints.project_lambda(lambda_vec)
        L = error + np.sum(lambda_vec * (gamma - self.constraints.bound()))
        max_constraint = (gamma - self.constraints.bound()).max()
        L_high = error
        if max_constraint > 0:
            L_high += self.B * max_constraint
        return L, L_high, gamma, error

    # ============================================
    # Evaluate the duality gap for a given Q and λ:
    #   gap = max{ L(Q, λ) - L_low, L_high - L(Q, λ) }
    # Starts with current Q and λ to compute L and L_high.
    # Then searches for a better lower bound by training new h's
    # using scaled versions of λ: mul * λ.
    # If any h leads to lower L, updates L_low.
    # Stops early if the gap becomes sufficiently large.
    # ===================================================
    def eval_gap(self, Q, lambda_hat, nu):
        L, L_high, gamma, error = self._eval(Q, lambda_hat)
        result = _GapResult(L, L, L_high, gamma, error)
        for mul in [1.0, 2.0, 5.0, 10.0]:
            # Clamp to [0, B] so multiplied lambdas stay in the valid dual feasible set
            clamped_lambda = (mul * lambda_hat).clip(upper=self.B)
            _, h_hat_idx, _ = self.best_h(clamped_lambda)
            logger.debug("%smul=%.0f", _INDENTATION, mul)
            L_low_mul, _, _, _ = self._eval(pd.Series({h_hat_idx: 1.0}), lambda_hat)
            if L_low_mul < result.L_low:
                result.L_low = L_low_mul
            if result.gap() > nu + _PRECISION:
                break
        return result

    # ================================================
    # Solve a linear program to find the best randomized classifier Q over existing h's,
    # minimizing the Lagrangian objective under fairness constraints.
    #
    # The primal LP optimizes:
    #   min_Q   ∑ Q[i] * error(h_i) + B * η
    #   s.t.    gamma(Q) - bound ≤ η
    #           ∑ Q[i] = 1
    #
    # The dual LP is solved to obtain lambda_vec (Lagrange multipliers).
    # Returns: (Q, lambda_vec, gap_result)
    # The best λ-response to a given Q (BEST_λ(Q)) corresponds to placing all weight B
    # ================================================
    def solve_linprog(self, nu):
        n_hs = len(self.hs)
        n_constraints = len(self.constraints.index)
        if self.last_linprog_n_hs == n_hs and self.last_linprog_result is not None:
            return self.last_linprog_result
        c = np.concatenate((self.errors, [self.B]))
        A_ub = np.concatenate(
            (
                self.gammas.sub(self.constraints.bound(), axis=0),
                -np.ones((n_constraints, 1)),
            ),
            axis=1,
        )
        b_ub = np.zeros(n_constraints)
        A_eq = np.concatenate((np.ones((1, n_hs)), np.zeros((1, 1))), axis=1)
        b_eq = np.ones(1)
        result = opt.linprog(
            c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, method="highs-ds"
        )
        Q = pd.Series(result.x[:-1], self.hs.index)
        dual_c = np.concatenate((b_ub, -b_eq))
        dual_A_ub = np.concatenate((-A_ub.transpose(), A_eq.transpose()), axis=1)
        dual_b_ub = c
        dual_bounds = [
            (None, None) if i == n_constraints else (0, None)
            for i in range(n_constraints + 1)
        ]
        result_dual = opt.linprog(
            dual_c,
            A_ub=dual_A_ub,
            b_ub=dual_b_ub,
            bounds=dual_bounds,
            method="highs-ds",
        )
        lambda_vec = pd.Series(result_dual.x[:-1], self.constraints.index)
        self.last_linprog_n_hs = n_hs
        self.last_linprog_result = (Q, lambda_vec, self.eval_gap(Q, lambda_vec, nu))
        return self.last_linprog_result

    # ===============================
    # Oracle: trains a classifier h in response to the current λ vector.
    # Computes signed weights combining accuracy loss and fairness constraint violations.
    # Generate new training labels (redY) based on fairness constraints
    # ===============================
    def _call_oracle(self, lambda_vec):
        signed_weights = self.obj.signed_weights() + self.constraints.signed_weights(
            lambda_vec
        )
        signed_y = self.constraints._y_as_series.copy()
        signed_weight_series = signed_weights.copy()
        unique_classes = np.sort(np.unique(signed_y))

        # y_p is None for "All" constraints that iterate over every label
        y_p = getattr(self.constraints, "y_p", None)

        if len(unique_classes) == 2:
            # Binary classification
            # When y_p is unknown, pick unique_classes[1] as positive arbitrarily
            pos_class = y_p if y_p is not None else unique_classes[1]
            neg_class = (
                unique_classes[0]
                if unique_classes[1] == pos_class
                else unique_classes[1]
            )
            redY = pd.Series(
                np.where(signed_weight_series > 0, pos_class, neg_class),
                index=signed_y.index,
            )
        else:
            if y_p is not None:
                # Single-label constraint: push toward y_p when weight > 0, else keep true label
                redY = pd.Series(
                    np.where(signed_weight_series > 0, y_p, signed_y),
                    index=signed_y.index,
                )
            else:
                # "All" constraints: no single positive label.
                # Positive weight  → push toward the true label (correct prediction reduces loss).
                # Negative weight  → push toward the *next* class in sorted order (mod K),
                # cycling across all K classes so every class can be targeted.
                K = len(unique_classes)
                cls_to_idx = {c: i for i, c in enumerate(unique_classes)}
                next_class = np.array(
                    [unique_classes[(cls_to_idx[c] + 1) % K] for c in signed_y.values]
                )
                redY = pd.Series(
                    np.where(
                        signed_weight_series.values > 0, signed_y.values, next_class
                    ),
                    index=signed_y.index,
                )

        redW = signed_weight_series.abs()
        w_sum = redW.sum()
        if w_sum < _PRECISION:
            # All signed weights are zero: fall back to uniform sample weights
            redW = pd.Series(1.0 / len(redW), index=redW.index)
        else:
            redW = self.constraints.total_samples * redW / w_sum
        redY_unique = np.unique(redY)

        # Train the estimator (oracle) using redY and redW
        if len(redY_unique) == 1:
            logger.debug("redY had single value. Using DummyClassifier")
            estimator = DummyClassifier(strategy="constant", constant=redY_unique[0])
            self.n_oracle_calls_dummy_returned += 1
        else:
            estimator = clone(estimator=self.estimator, safe=False)

        oracle_call_start_time = time()
        estimator.fit(self.constraints.X, redY, **{self.sample_weight_name: redW})
        self.oracle_execution_times.append(time() - oracle_call_start_time)
        self.n_oracle_calls += 1
        return estimator

    # =========================================================
    # best_h: Computes the best classifier h for current lambda
    # ---------------------------------------------------------
    # - Trains a new classifier using _call_oracle(lambda_vec)
    # - Computes its error and fairness violation (gamma)
    # - Evaluates its Lagrangian value: error + lambda^T * gamma
    # - If this classifier improves the current best, we store it
    # - Returns the best classifier (among all iterations so far)
    # =========================================================
    def best_h(self, lambda_vec):
        classifier = self._call_oracle(lambda_vec)
        h = _PredictorAsCallable(classifier)
        h_error = self.obj.gamma(h).iloc[0]
        h_gamma = self.constraints.gamma(h)
        # Always store the newly trained classifier so the LP pool stays complete
        new_h_idx = len(self.hs)
        self.hs.at[new_h_idx] = h
        self.predictors.at[new_h_idx] = classifier
        self.errors.at[new_h_idx] = h_error
        self.gammas[new_h_idx] = h_gamma
        self.lambdas[new_h_idx] = lambda_vec.copy()
        # Find the best classifier among all stored (including the one just added)
        values = self.errors + self.gammas.transpose().dot(lambda_vec)
        best_idx = values.idxmin()
        logger.debug(
            "%sbest_h: new=%d best=%d val=%.6f",
            _LINE,
            new_h_idx,
            best_idx,
            values[best_idx],
        )
        return self.hs[best_idx], best_idx, new_h_idx


# =====================================
# EXPONENTIATED GRADIENT CLASS (full)
# =====================================


class GeneralizedExponentiatedGradient(BaseEstimator, MetaEstimatorMixin):
    def __init__(
        self,
        estimator,
        constraints: Moment,
        *,
        objective: Moment | None = None,
        eps: float = 0.01,
        max_iter: int = 50,
        nu: float | None = None,
        eta0: float = 2.0,
        run_linprog_step: bool = True,
        sample_weight_name: str = "sample_weight",
        positive_label=None,
    ):
        self.estimator = estimator
        self.constraints = constraints
        self.objective = objective
        self.eps = eps
        self.max_iter = max_iter
        self.nu = nu
        self.eta0 = eta0
        self.run_linprog_step = run_linprog_step
        self.sample_weight_name = sample_weight_name
        # positive_label can be None when using All-label constraints (DP/EO over all classes)
        self.positive_label = positive_label

    def fit(self, X, y, **kwargs):
        self.lambda_vecs_EG_ = pd.DataFrame()
        self.lambda_vecs_LP_ = pd.DataFrame()

        logger.debug("...Exponentiated Gradient STARTING")

        B = 1 / self.eps
        lagrangian = _Lagrangian(
            X=X,
            y=y,
            estimator=self.estimator,
            constraints=self.constraints,
            B=B,
            objective=self.objective,
            sample_weight_name=self.sample_weight_name,
            **kwargs,
        )

        theta = pd.Series(0, lagrangian.constraints.index)
        Qsum = pd.Series(dtype="float64")
        gaps_EG = []
        gaps = []
        Qs = []

        last_regret_checked = _REGRET_CHECK_START_T
        last_gap = np.inf

        for t in range(self.max_iter):
            logger.debug("...iter=%03d", t)

            lambda_vec = B * np.exp(theta) / (1 + np.exp(theta).sum())
            self.lambda_vecs_EG_[t] = lambda_vec
            lambda_EG = self.lambda_vecs_EG_.mean(axis=1)

            h, h_idx, new_h_idx = lagrangian.best_h(lambda_vec)

            if t == 0:
                if self.nu is None:
                    # Use 0/1 residuals so nu is well-defined for any label type
                    residuals = (h(X) != self.constraints._y_as_series).astype(float)
                    self.nu = (
                        _ACCURACY_MUL
                        * residuals.std()
                        / np.sqrt(self.constraints.total_samples)
                    )
                eta = self.eta0 / B
                logger.debug(
                    "...eps=%.3f, B=%.1f, nu=%.6f, max_iter=%d",
                    self.eps,
                    B,
                    self.nu,
                    self.max_iter,
                )

            # Track the current oracle output (not the best historical) in the mixture
            # and use its gamma for the theta gradient, as required by Algorithm 1
            if new_h_idx not in Qsum.index:
                Qsum.at[new_h_idx] = 0.0
            Qsum[new_h_idx] += 1.0
            gamma = lagrangian.gammas[new_h_idx]
            Q_EG = Qsum / Qsum.sum()
            result_EG = lagrangian.eval_gap(Q_EG, lambda_EG, self.nu)
            gap_EG = result_EG.gap()
            gaps_EG.append(gap_EG)

            if t == 0 or not self.run_linprog_step:
                gap_LP = np.inf
            else:
                Q_LP, self.lambda_vecs_LP_[t], result_LP = lagrangian.solve_linprog(
                    self.nu
                )
                gap_LP = result_LP.gap()

            if gap_EG < gap_LP:
                Qs.append(Q_EG)
                gaps.append(gap_EG)
            else:
                Qs.append(Q_LP)
                gaps.append(gap_LP)

            logger.debug(
                "%seta=%.6f, gap=%.6f, gap_LP=%.6f", _INDENTATION, eta, gap_EG, gap_LP
            )

            if (gaps[t] < self.nu) and (t >= _MIN_ITER):
                break

            if t >= last_regret_checked * _REGRET_CHECK_INCREASE_T:
                best_gap = min(gaps_EG)
                if best_gap > last_gap * _SHRINK_REGRET:
                    eta *= _SHRINK_ETA
                last_regret_checked = t
                last_gap = best_gap

            theta += eta * (gamma - self.constraints.bound())

        gaps_series = pd.Series(gaps)
        gaps_best = gaps_series[gaps_series <= gaps_series.min() + _PRECISION]
        self.best_iter_ = gaps_best.index[-1]
        self.best_gap_ = gaps[self.best_iter_]
        self.weights_ = Qs[self.best_iter_]
        self._hs = lagrangian.hs

        # Ensure all classifiers have weights (even if zero)
        for h_idx in self._hs.index:
            if h_idx not in self.weights_.index:
                self.weights_.at[h_idx] = 0.0

        # Normalize weights to sum to 1
        self.weights_ = self.weights_ / self.weights_.sum()

        self.last_iter_ = len(Qs) - 1
        self.predictors_ = lagrangian.predictors
        self.n_oracle_calls_ = lagrangian.n_oracle_calls
        self.n_oracle_calls_dummy_returned_ = lagrangian.n_oracle_calls_dummy_returned
        self.oracle_execution_times_ = lagrangian.oracle_execution_times
        self.lambda_vecs_ = lagrangian.lambdas

        return self

    def _pmf_predict(self, X):
        check_is_fitted(self)

        # Get all unique classes first by collecting predictions from all classifiers
        all_classes = set()
        for t in self._hs.index:
            pred = self._hs[t](X)
            all_classes.update(pred)
        all_classes = np.sort(np.array(list(all_classes)))

        # Initialize probability matrix
        class_probs = np.zeros((len(X), len(all_classes)))

        # Calculate weighted probabilities
        for t in self._hs.index:
            pred = self._hs[t](X)
            for i, c in enumerate(all_classes):
                class_probs[:, i] += (pred == c).astype(float) * self.weights_[t]

        return class_probs, all_classes

    def predict(self, X, random_state=None):
        check_is_fitted(self)
        random_state = check_random_state(random_state)
        probs, all_classes = self._pmf_predict(X)

        # When positive_label is None (All-label constraints) or missing, use argmax directly.
        if self.positive_label is None or self.positive_label not in all_classes:
            if (
                self.positive_label is not None
                and self.positive_label not in all_classes
            ):
                # Single y_p was set but never predicted: keep existing behaviour of adding it
                # back with zero probability so downstream code stays consistent.
                all_classes = np.sort(np.append(all_classes, self.positive_label))
                new_probs = np.zeros((len(X), len(all_classes)))
                orig_classes = all_classes[all_classes != self.positive_label]
                for i, cls in enumerate(all_classes):
                    if cls != self.positive_label:
                        orig_idx = np.where(orig_classes == cls)[0][0]
                        new_probs[:, i] = probs[:, orig_idx]
                probs = new_probs
            # For All-label constraints (positive_label is None): pure argmax over all classes.
            return all_classes[np.argmax(probs, axis=1)]

        # Use argmax for all cases: deterministic, reproducible, consistent with binary path
        return all_classes[np.argmax(probs, axis=1)]
