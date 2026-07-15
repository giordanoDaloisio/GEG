# Our proposed code for Generalized the simple version(for Binary and multi-class)
# ------------------------
# Imports and constants
# ------------------------
import pandas as pd
import numpy as np
from fairlearn.reductions._moments.moment import ClassificationMoment
from fairlearn.reductions._moments.moment import (
    _GROUP_ID,
    _LABEL,
    _PREDICTION,
    _ALL,
    _EVENT,
    _SIGN,
)
from fairlearn.utils._input_validation import (
    _validate_and_reformat_input,
    _MESSAGE_RATIO_NOT_IN_RANGE,
)

_UPPER_BOUND_DIFF = "upper_bound_diff"
_LOWER_BOUND_DIFF = "lower_bound_diff"
_MESSAGE_INVALID_BOUNDS = "Only one of difference_bound and ratio_bound can be used."
_DEFAULT_DIFFERENCE_BOUND = 0.01
_CTRL_EVENT_FORMAT = "control={0},{1}"


# ------------------------
# Helper functions
# ------------------------
def _combine_event_and_control(event: str, control: str) -> str:
    if pd.notnull(control):
        return _CTRL_EVENT_FORMAT.format(control, event)
    else:
        return event


def _merge_event_and_control_columns(event_col, control_col):
    if control_col is None:
        return event_col
    else:
        return event_col.combine(control_col, _combine_event_and_control)


# ------------------------
# General Error Rate class
# -------------------
class GeneralErrorRate1(ClassificationMoment):
    """Misclassification error for both binary and multi-class classification."""

    short_name = "GenErr1"

    def __init__(self, y_p=1):
        super().__init__()
        self.y_p = y_p

    @property
    def index(self):
        return self._index

    def load_data(self, X, y, *, sensitive_features, control_features=None):
        """Load the specified data into the object."""
        _, y_train, sf_train, cf_train = _validate_and_reformat_input(
            X,
            y,
            sensitive_features=sensitive_features,
            control_features=control_features,
        )
        super().load_data(X, y_train, sensitive_features=sf_train)
        self._index = [_ALL]

    def gamma(self, predictor):
        """Return the misclassification error rate of the predictor."""
        pred = predictor(self.X)
        if isinstance(pred, np.ndarray):
            pred = np.squeeze(pred)
        error = pd.Series(data=(self.tags[_LABEL] != pred).mean(), index=self.index)
        self._gamma_descr = str(error)
        self.tags[_PREDICTION] = pred  # Needed for signed_weights
        return error

    def project_lambda(self, lambda_vec):
        """Return the lambda values (no projection needed)."""
        return lambda_vec

    def signed_weights(self, lambda_vec=None):
        """Return signed weights for binary or multi-class using positive label self.y_p."""
        y = self.tags[_LABEL]

        # For error minimization, we want:
        # - Positive weight when we correctly predict the positive class (y == y_p and pred == y_p)
        # - Negative weight when we incorrectly predict the positive class (y != y_p and pred == y_p)
        # Since we're in the context of training, we use the true labels to compute the sign
        indicator = (y == self.y_p).astype(float)
        # Flip the sign: +1 for positive class, -1 for negative class
        # This ensures that minimizing the weighted error actually minimizes error
        weights = 2 * indicator - 1

        if lambda_vec is None:
            return weights
        else:
            return lambda_vec[_ALL] * weights


# ------------------------
# General Utility Builder (multi-class)
# ------------------------
def build_pred_based_utilities(y: pd.Series, y_p: int) -> np.ndarray:
    classes = sorted(y.unique())
    n_classes = len(classes)
    n_samples = len(y)

    utilities = np.zeros((n_samples, n_classes), dtype=np.float64)
    for i, cls in enumerate(classes):
        if cls == y_p:
            utilities[:, i] = 1.0  # utility = 1 if prediction == y_p
    return utilities


# ------------------------
# Symmetric (class-agnostic) Error Rate for "All" constraints
# ------------------------
class SymmetricErrorRate(ClassificationMoment):
    """Standard misclassification error that treats every class equally.

    Used as the default objective for All-label constraints so no single
    class is favoured.  The signed_weights are +1 for every sample,
    letting the constraint term drive the oracle direction.
    """

    short_name = "SymErr"

    @property
    def index(self):
        return self._index

    def load_data(self, X, y, *, sensitive_features, control_features=None):
        _, y_train, sf_train, _ = _validate_and_reformat_input(
            X,
            y,
            sensitive_features=sensitive_features,
            control_features=control_features,
        )
        super().load_data(X, y_train, sensitive_features=sf_train)
        self._index = [_ALL]

    def gamma(self, predictor):
        pred = predictor(self.X)
        if isinstance(pred, np.ndarray):
            pred = np.squeeze(pred)
        error = pd.Series(data=(self.tags[_LABEL] != pred).mean(), index=self.index)
        self._gamma_descr = str(error)
        return error

    def project_lambda(self, lambda_vec):
        return lambda_vec

    def signed_weights(self, lambda_vec=None):
        """Return uniform +1 weights so every sample is treated equally."""
        weights = pd.Series(1.0, index=self.tags.index)
        if lambda_vec is None:
            return weights
        return lambda_vec[_ALL] * weights


# ------------------------
# GeneralUtilityParity class (multi-class and Binary)
# ------------------------
class GeneralUtilityParity1(ClassificationMoment):
    def __init__(
        self,
        *,
        difference_bound=None,
        ratio_bound=None,
        ratio_bound_slack=0.0,
        y_p=None,
    ):
        super(GeneralUtilityParity1, self).__init__()
        self.y_p = y_p
        if (difference_bound is None) and (ratio_bound is None):
            self.eps = _DEFAULT_DIFFERENCE_BOUND
            self.ratio = 1.0
        elif (difference_bound is not None) and (ratio_bound is None):
            self.eps = difference_bound
            self.ratio = 1.0
        elif (difference_bound is None) and (ratio_bound is not None):
            self.eps = ratio_bound_slack
            if not (0 < ratio_bound <= 1):
                raise ValueError(_MESSAGE_RATIO_NOT_IN_RANGE)
            self.ratio = ratio_bound
        else:
            raise ValueError(_MESSAGE_INVALID_BOUNDS)

    @property
    def index(self):
        return self._index

    def default_objective(self):
        return GeneralErrorRate1(y_p=self.y_p)

    def load_data(
        self,
        X,
        y: pd.Series,
        *,
        sensitive_features: pd.Series,
        event: pd.Series = None,
        utilities=None,
    ):
        super().load_data(X, y, sensitive_features=sensitive_features)
        self.tags[_EVENT] = event

        # ===> Generalized utility matrix based on y_p
        if utilities is None:
            if self.y_p is None:
                raise ValueError("y_p must be specified to build the utility matrix")
            utilities = build_pred_based_utilities(y, y_p=self.y_p)

        self.utilities = utilities

        self.classes_ = sorted(np.unique(self.tags[_LABEL]))
        self.y_p_index = self.classes_.index(self.y_p)

        self.prob_event = self.tags.groupby(_EVENT).size() / self.total_samples
        self.prob_group_event = (
            self.tags.groupby([_EVENT, _GROUP_ID]).size() / self.total_samples
        )
        signed = pd.concat(
            [self.prob_group_event, self.prob_group_event],
            keys=["+", "-"],
            names=[_SIGN, _EVENT, _GROUP_ID],
        )
        self._index = signed.index
        self.default_objective_lambda_vec = None

        event_vals = self.tags[_EVENT].dropna().unique()
        group_vals = self.tags[_GROUP_ID].unique()
        self.pos_basis = pd.DataFrame(index=self.index)
        self.neg_basis = pd.DataFrame(index=self.index)
        self.neg_basis_present = pd.Series(dtype="float64")
        zero_vec = pd.Series(0.0, self.index)
        i = 0
        for event_val in event_vals:
            for group in group_vals[:-1]:
                self.pos_basis[i] = 0 + zero_vec
                self.neg_basis[i] = 0 + zero_vec
                self.pos_basis.loc[("+", event_val, group), i] = 1
                self.neg_basis.loc[("-", event_val, group), i] = 1
                self.neg_basis_present.at[i] = True
                i += 1

    def gamma(self, predictor):
        predictions = predictor(self.X)
        predictions = np.squeeze(predictions)
        pred = (predictions == self.y_p).astype(float)
        self.tags[_PREDICTION] = pred

        expect_event = self.tags.groupby(_EVENT)[[_PREDICTION]].mean()
        expect_group_event = (
            self.tags.groupby([_EVENT, _GROUP_ID])[_PREDICTION].mean().to_frame()
        )

        expect_group_event[_UPPER_BOUND_DIFF] = (
            self.ratio * expect_group_event[_PREDICTION]
            - expect_event[_PREDICTION]
            .reindex(expect_group_event.index.get_level_values(0))
            .values
        )
        expect_group_event[_LOWER_BOUND_DIFF] = (
            -expect_group_event[_PREDICTION]
            + self.ratio
            * expect_event[_PREDICTION]
            .reindex(expect_group_event.index.get_level_values(0))
            .values
        )

        g_signed = pd.concat(
            [
                expect_group_event[_UPPER_BOUND_DIFF],
                expect_group_event[_LOWER_BOUND_DIFF],
            ],
            keys=["+", "-"],
            names=[_SIGN, _EVENT, _GROUP_ID],
        )

        self._gamma_descr = str(
            expect_group_event[[_PREDICTION, _UPPER_BOUND_DIFF, _LOWER_BOUND_DIFF]]
        )
        return g_signed

    def bound(self):
        return pd.Series(self.eps, index=self.index)

    def project_lambda(self, lambda_vec):
        if self.ratio == 1.0:
            lambda_pos = lambda_vec["+"] - lambda_vec["-"]
            lambda_neg = -lambda_pos
            lambda_pos[lambda_pos < 0.0] = 0.0
            lambda_neg[lambda_neg < 0.0] = 0.0
            return pd.concat(
                [lambda_pos, lambda_neg],
                keys=["+", "-"],
                names=[_SIGN, _EVENT, _GROUP_ID],
            )
        return lambda_vec

    def signed_weights(self, lambda_vec):
        lambda_event = (lambda_vec["+"] - self.ratio * lambda_vec["-"]).groupby(
            level=_EVENT
        ).sum() / self.prob_event
        lambda_group_event = (
            self.ratio * lambda_vec["+"] - lambda_vec["-"]
        ) / self.prob_group_event
        adjust = (
            lambda_event.reindex(lambda_group_event.index.get_level_values(0)).values
            - lambda_group_event.values
        )
        adjust_series = pd.Series(adjust, index=lambda_group_event.index)

        signed_weights = self.tags.apply(
            lambda row: (
                0
                if pd.isna(row[_EVENT])
                else adjust_series[row[_EVENT], row[_GROUP_ID]]
            ),
            axis=1,
        )

        utility_diff = self.utilities[:, self.y_p_index]
        return utility_diff * signed_weights


# ---------------------------------------------------------------------------------
#  General DemographicParity1 class, can works for binary and multi-class using y_P
# ---------------------------------------------------------------------------------
class GeneralDemographicParity1(GeneralUtilityParity1):
    short_name = "GeneralDemographicParity1"

    def __init__(self, *, y_p=None, difference_bound=None):
        super().__init__(y_p=y_p, difference_bound=difference_bound)

    def load_data(self, X, y, *, sensitive_features, control_features=None):
        _, y_train, sf_train, cf_train = _validate_and_reformat_input(
            X,
            y,
            sensitive_features=sensitive_features,
            control_features=control_features,
        )
        base_event = pd.Series(data=_ALL, index=y_train.index)
        event = _merge_event_and_control_columns(base_event, cf_train)
        super().load_data(X, y_train, event=event, sensitive_features=sf_train)


# ------------------------------------------------------------------------------------
#  General EqualizedOdds1 class
# ------------------------------------------------------------------------------------
class GeneralEqualizedOdds1(GeneralUtilityParity1):
    short_name = "GeneralEqualizedOdds1"

    def __init__(self, *, y_p=None, difference_bound=None):
        super().__init__(y_p=y_p, difference_bound=difference_bound)

    def load_data(self, X, y, *, sensitive_features, control_features=None):
        _, y_train, sf_train, cf_train = _validate_and_reformat_input(
            X,
            y,
            sensitive_features=sensitive_features,
            control_features=control_features,
        )

        # Define the event as the label itself: for each y_i, event = "label=y_i"
        base_event = y_train.apply(lambda v: _LABEL + "=" + str(v))
        event = _merge_event_and_control_columns(base_event, cf_train)

        super().load_data(X, y_train, event=event, sensitive_features=sf_train)


# -----------------------------------------
# Combined Fairness Constraint Class (DP + EO)
# -----------------------------------------
class CombinedParityGeneral1(ClassificationMoment):
    """Combined demographic parity and equalized odds constraints for classification."""

    short_name = "CombinedParityGeneral1"

    def __init__(
        self,
        *,
        y_p,
        use_dp=True,
        use_eo=True,
        dp_bound=None,
        eo_bound=None,
        dp_ratio_bound=None,
        eo_ratio_bound=None,
        ratio_bound_slack=0.0,
    ):
        super().__init__()
        if not use_dp and not use_eo:
            raise ValueError("At least one of use_dp or use_eo must be True")
        if (dp_bound is not None and dp_ratio_bound is not None) or (
            eo_bound is not None and eo_ratio_bound is not None
        ):
            raise ValueError(_MESSAGE_INVALID_BOUNDS)

        self.y_p = y_p
        self.use_dp = use_dp
        self.use_eo = use_eo
        self.dp_bound = dp_bound if dp_bound is not None else _DEFAULT_DIFFERENCE_BOUND
        self.eo_bound = eo_bound if eo_bound is not None else _DEFAULT_DIFFERENCE_BOUND
        self.dp_ratio_bound = dp_ratio_bound
        self.eo_ratio_bound = eo_ratio_bound
        self.ratio_bound_slack = ratio_bound_slack

    @property
    def index(self):
        return self._index

    def default_objective(self):
        """Return the default objective (error rate)."""
        return GeneralErrorRate1(y_p=self.y_p)

    def load_data(
        self, X, y, *, sensitive_features, control_features=None, utilities=None
    ):
        """Load the specified data into the object."""
        _, y_train, sf_train, cf_train = _validate_and_reformat_input(
            X,
            y,
            sensitive_features=sensitive_features,
            control_features=control_features,
        )
        self.tags = pd.DataFrame({_LABEL: y_train, _GROUP_ID: sf_train})
        self.X = X
        self._y = y_train
        self._total_samples = len(y_train)

        # Build utilities if not provided
        if utilities is None:
            utilities = build_pred_based_utilities(y_train, self.y_p)
        self.utilities = utilities
        self.classes_ = sorted(np.unique(y_train))
        self.y_p_index = self.classes_.index(self.y_p)

        # Initialize events and bounds
        events = []
        bounds = []
        ratios = {}

        if self.use_dp:
            self.tags["dp_event"] = pd.Series(_ALL, index=y_train.index)
            events.append("dp_event")
            bounds.append(self.dp_bound)
            ratios["dp_event"] = (
                self.dp_ratio_bound if self.dp_ratio_bound is not None else 1.0
            )

        if self.use_eo:
            self.tags["eo_event"] = y_train.apply(lambda v: f"{_LABEL}={v}")
            events.append("eo_event")
            bounds.append(self.eo_bound)
            ratios["eo_event"] = (
                self.eo_ratio_bound if self.eo_ratio_bound is not None else 1.0
            )

        # Create index and probability distributions
        _idx_tuples = []
        bound_vals = []
        self.prob_event = {}
        self.prob_group_event = {}

        for ev_col, bound in zip(events, bounds):
            ev_vals = self.tags[ev_col].unique()
            self.prob_event[ev_col] = (
                self.tags.groupby(ev_col).size() / self._total_samples
            )
            self.prob_group_event[ev_col] = (
                self.tags.groupby([ev_col, _GROUP_ID]).size() / self._total_samples
            )

            for ev in ev_vals:
                for g in self.tags[_GROUP_ID].unique():
                    _idx_tuples.append(("+", ev, g))
                    _idx_tuples.append(("-", ev, g))
                    bound_vals.append(bound)
                    bound_vals.append(bound)

        self._index = pd.MultiIndex.from_tuples(
            _idx_tuples, names=[_SIGN, _EVENT, _GROUP_ID]
        )
        self.bound_ = pd.Series(bound_vals, index=self._index)
        self.ratios = ratios

    def bound(self):
        """Return the bound values."""
        return self.bound_

    def gamma(self, predictor):
        """Calculate gamma values for the current predictor."""
        predictions = np.squeeze(predictor(self.X))
        pred = (predictions == self.y_p).astype(float)
        self.tags[_PREDICTION] = pred

        gamma_list = []
        for ev_col in ["dp_event", "eo_event"]:
            if ev_col not in self.tags.columns:
                continue

            ratio = self.ratios[ev_col]
            mean_event = self.tags.groupby(ev_col)[_PREDICTION].mean()
            mean_group_event = self.tags.groupby([ev_col, _GROUP_ID])[
                _PREDICTION
            ].mean()

            upper = (
                ratio * mean_group_event
                - mean_event.reindex(mean_group_event.index.get_level_values(0)).values
            )
            lower = (
                -mean_group_event
                + ratio
                * mean_event.reindex(mean_group_event.index.get_level_values(0)).values
            )

            g = pd.concat(
                [upper, lower], keys=["+", "-"], names=[_SIGN, _EVENT, _GROUP_ID]
            )
            gamma_list.append(g)

        gamma_final = pd.concat(gamma_list).reindex(self.index).fillna(0)
        self._gamma_descr = str(gamma_final)
        return gamma_final

    def project_lambda(self, lambda_vec):
        """Project lambda values according to constraints."""
        dp_ratio_1 = self.dp_ratio_bound is None or self.dp_ratio_bound == 1.0
        eo_ratio_1 = self.eo_ratio_bound is None or self.eo_ratio_bound == 1.0

        if dp_ratio_1 and eo_ratio_1:
            lambda_pos = lambda_vec["+"] - lambda_vec["-"]
            lambda_neg = -lambda_pos
            lambda_pos[lambda_pos < 0.0] = 0.0
            lambda_neg[lambda_neg < 0.0] = 0.0
            return pd.concat(
                [lambda_pos, lambda_neg],
                keys=["+", "-"],
                names=[_SIGN, _EVENT, _GROUP_ID],
            )
        return lambda_vec

    def signed_weights(self, lambda_vec):
        """Compute signed weights for the classifier."""
        signed_weights = pd.Series(0.0, index=self.tags.index)

        for ev_col in ["dp_event", "eo_event"]:
            if ev_col not in self.tags.columns:
                continue

            ratio = self.ratios[ev_col]
            prob_e = self.prob_event[ev_col]
            prob_ge = self.prob_group_event[ev_col]

            lambda_event = (
                (lambda_vec["+"] - ratio * lambda_vec["-"]).groupby(level=_EVENT).sum()
            )
            lambda_group_event = ratio * lambda_vec["+"] - lambda_vec["-"]

            for e, g in prob_ge.index:
                adjust = (
                    lambda_event[e] / prob_e[e]
                    - lambda_group_event[(e, g)] / prob_ge[(e, g)]
                )
                mask = (self.tags[ev_col] == e) & (self.tags[_GROUP_ID] == g)
                signed_weights[mask] += adjust

        utility_diff = self.utilities[:, self.y_p_index]
        return utility_diff * signed_weights

    def __repr__(self):
        return (
            f"CombinedParityGeneral1(y_p={self.y_p}, use_dp={self.use_dp}, use_eo={self.use_eo}, "
            f"dp_bound={self.dp_bound}, eo_bound={self.eo_bound}, "
            f"dp_ratio_bound={self.dp_ratio_bound}, eo_ratio_bound={self.eo_ratio_bound})"
        )


# ---------------------------------------------------------------------------------
#  Multi-label helpers shared by the "All" variants
# ---------------------------------------------------------------------------------
def _build_all_label_index(
    event_col_name: str, ev_vals, group_vals, sign_vals=("+", "-")
):
    """Return a MultiIndex over (sign, event_with_label, group) for all labels."""
    tuples = []
    for sign in sign_vals:
        for ev in ev_vals:
            for g in group_vals:
                tuples.append((sign, ev, g))
    return pd.MultiIndex.from_tuples(tuples, names=[_SIGN, _EVENT, _GROUP_ID])


# ---------------------------------------------------------------------------------
#  GeneralDemographicParityAll
#  DP applied for every class label y simultaneously:
#    P(ŷ = y | A = a) = P(ŷ = y)  ∀ a ∈ A, y ∈ Y
# ---------------------------------------------------------------------------------
class GeneralDemographicParityAll(ClassificationMoment):
    """Demographic parity generalised to all class labels simultaneously."""

    short_name = "GeneralDemographicParityAll"

    def __init__(
        self, *, difference_bound=None, ratio_bound=None, ratio_bound_slack=0.0
    ):
        super().__init__()
        if (difference_bound is not None) and (ratio_bound is not None):
            raise ValueError(_MESSAGE_INVALID_BOUNDS)
        if ratio_bound is not None:
            if not (0 < ratio_bound <= 1):
                raise ValueError(_MESSAGE_RATIO_NOT_IN_RANGE)
            self.eps = ratio_bound_slack
            self.ratio = ratio_bound
        else:
            self.eps = (
                difference_bound
                if difference_bound is not None
                else _DEFAULT_DIFFERENCE_BOUND
            )
            self.ratio = 1.0

    def default_objective(self):
        # Use label 0 as placeholder; error rate is label-agnostic for binary objectives
        return SymmetricErrorRate()

    def load_data(self, X, y, *, sensitive_features, control_features=None):
        _, y_train, sf_train, _ = _validate_and_reformat_input(
            X,
            y,
            sensitive_features=sensitive_features,
            control_features=control_features,
        )
        super().load_data(X, y_train, sensitive_features=sf_train)

        self._classes = sorted(np.unique(y_train))
        group_vals = self.tags[_GROUP_ID].unique()

        # One event column per label: "dp_label=<y>"
        self._dp_event_cols = []
        for cls in self._classes:
            col = f"dp_label={cls}"
            self.tags[col] = _ALL  # same event value for all samples (DP)
            self._dp_event_cols.append(col)

        # Pre-compute probabilities and build index
        self._prob_event = {}
        self._prob_group_event = {}
        index_tuples = []
        bound_vals = []

        for col in self._dp_event_cols:
            self._prob_event[col] = self.tags.groupby(col).size() / self.total_samples
            self._prob_group_event[col] = (
                self.tags.groupby([col, _GROUP_ID]).size() / self.total_samples
            )
            for ev in self.tags[col].unique():
                for g in group_vals:
                    index_tuples.append(("+", f"{col}|{ev}", g))
                    index_tuples.append(("-", f"{col}|{ev}", g))
                    bound_vals.extend([self.eps, self.eps])

        self._index = pd.MultiIndex.from_tuples(
            index_tuples, names=[_SIGN, _EVENT, _GROUP_ID]
        )
        self._bound_series = pd.Series(bound_vals, index=self._index)

        # Utility matrices: one per label (1 where true label == cls, else 0)
        n_classes = len(self._classes)
        n_samples = len(y_train)
        self._utilities = np.zeros((n_samples, n_classes), dtype=np.float64)
        for i, cls in enumerate(self._classes):
            self._utilities[:, i] = (y_train == cls).astype(float)

        self._y_p_indices = {cls: self._classes.index(cls) for cls in self._classes}
        self._group_vals = group_vals

    @property
    def index(self):
        return self._index

    def gamma(self, predictor):
        predictions = np.squeeze(predictor(self.X))
        gamma_list = []

        for cls, col in zip(self._classes, self._dp_event_cols):
            pred_cls = (predictions == cls).astype(float)
            tmp = self.tags[[_GROUP_ID, col]].copy()
            tmp[_PREDICTION] = pred_cls

            mean_event = tmp.groupby(col)[_PREDICTION].mean()
            mean_group_event = tmp.groupby([col, _GROUP_ID])[_PREDICTION].mean()

            upper = (
                self.ratio * mean_group_event
                - mean_event.reindex(mean_group_event.index.get_level_values(0)).values
            )
            lower = (
                -mean_group_event
                + self.ratio
                * mean_event.reindex(mean_group_event.index.get_level_values(0)).values
            )

            # Re-key using the composite event name used in self.index
            ev_vals = tmp[col].unique()
            upper.index = pd.MultiIndex.from_tuples(
                [(f"{col}|{ev}", g) for (ev, g) in upper.index],
                names=[_EVENT, _GROUP_ID],
            )
            lower.index = upper.index

            g = pd.concat(
                [upper, lower], keys=["+", "-"], names=[_SIGN, _EVENT, _GROUP_ID]
            )
            gamma_list.append(g)

        gamma_final = pd.concat(gamma_list).reindex(self.index).fillna(0.0)
        self._gamma_descr = str(gamma_final)
        return gamma_final

    def bound(self):
        return self._bound_series

    def project_lambda(self, lambda_vec):
        if self.ratio == 1.0:
            lambda_pos = lambda_vec["+"] - lambda_vec["-"]
            lambda_neg = -lambda_pos
            lambda_pos[lambda_pos < 0.0] = 0.0
            lambda_neg[lambda_neg < 0.0] = 0.0
            return pd.concat(
                [lambda_pos, lambda_neg],
                keys=["+", "-"],
                names=[_SIGN, _EVENT, _GROUP_ID],
            )
        return lambda_vec

    def signed_weights(self, lambda_vec):
        signed_weights = pd.Series(0.0, index=self.tags.index)

        for cls, col in zip(self._classes, self._dp_event_cols):
            prob_e = self._prob_event[col]
            prob_ge = self._prob_group_event[col]

            ev_vals = self.tags[col].unique()
            for ev in ev_vals:
                ev_key = f"{col}|{ev}"
                try:
                    lp_ev = lambda_vec["+"].loc[ev_key]  # Series indexed by _GROUP_ID
                    lm_ev = lambda_vec["-"].loc[ev_key]
                    le = float((lp_ev - self.ratio * lm_ev).sum()) / float(prob_e[ev])
                except KeyError:
                    le = 0.0
                    lp_ev = pd.Series(dtype=float)
                    lm_ev = pd.Series(dtype=float)
                for g in self._group_vals:
                    pge = float(prob_ge.get((ev, g), 0.0))
                    if pge < 1e-8:
                        continue
                    lp_g = float(lp_ev.get(g, 0.0))
                    lm_g = float(lm_ev.get(g, 0.0))
                    adjust = le - (self.ratio * lp_g - lm_g) / pge
                    mask = (self.tags[col] == ev) & (self.tags[_GROUP_ID] == g)
                    signed_weights[mask] += (
                        adjust * self._utilities[:, self._y_p_indices[cls]][mask]
                    )

        return signed_weights

    def per_class_rewards(self, lambda_vec):
        """Per-sample, per-class Lagrangian rewards.

        Entry (i, k) is the coefficient of the indicator 1{h(x_i)=k} in
        -L(h, lambda) (a *gain*): predicting class k for sample i changes the
        Lagrangian by -reward/n. Unlike ``signed_weights`` -- which collapses
        the constraint pressure onto each sample's true label -- this keeps
        the pressure on every class, so the oracle can relabel toward the
        specific class the constraints push for (proper cost-sensitive
        multiclass reduction).
        """
        groups = self.tags[_GROUP_ID]
        rewards = pd.DataFrame(0.0, index=self.tags.index, columns=self._classes)

        for cls, col in zip(self._classes, self._dp_event_cols):
            prob_e = self._prob_event[col]
            prob_ge = self._prob_group_event[col]
            for ev in self.tags[col].dropna().unique():
                ev_key = f"{col}|{ev}"
                try:
                    lp_ev = lambda_vec["+"].loc[ev_key]
                    lm_ev = lambda_vec["-"].loc[ev_key]
                except KeyError:
                    continue
                le = float((lp_ev - self.ratio * lm_ev).sum()) / float(prob_e[ev])
                for g in self._group_vals:
                    pge = float(prob_ge.get((ev, g), 0.0))
                    if pge < 1e-8:
                        continue
                    lp_g = float(lp_ev.get(g, 0.0))
                    lm_g = float(lm_ev.get(g, 0.0))
                    adjust = le - (self.ratio * lp_g - lm_g) / pge
                    mask = (self.tags[col] == ev) & (groups == g)
                    rewards.loc[mask, cls] += adjust

        return rewards

    def __repr__(self):
        return f"GeneralDemographicParityAll(eps={self.eps}, ratio={self.ratio})"


# ---------------------------------------------------------------------------------
#  GeneralEqualizedOddsAll
#  EO applied for every class label y simultaneously:
#    P(ŷ = y | Y = y, A = a) = P(ŷ = y | Y = y)  ∀ a ∈ A, y ∈ Y
# ---------------------------------------------------------------------------------
class GeneralEqualizedOddsAll(ClassificationMoment):
    """Equalized odds generalised to all class labels simultaneously."""

    short_name = "GeneralEqualizedOddsAll"

    def __init__(
        self, *, difference_bound=None, ratio_bound=None, ratio_bound_slack=0.0
    ):
        super().__init__()
        if (difference_bound is not None) and (ratio_bound is not None):
            raise ValueError(_MESSAGE_INVALID_BOUNDS)
        if ratio_bound is not None:
            if not (0 < ratio_bound <= 1):
                raise ValueError(_MESSAGE_RATIO_NOT_IN_RANGE)
            self.eps = ratio_bound_slack
            self.ratio = ratio_bound
        else:
            self.eps = (
                difference_bound
                if difference_bound is not None
                else _DEFAULT_DIFFERENCE_BOUND
            )
            self.ratio = 1.0

    def default_objective(self):
        return SymmetricErrorRate()

    def load_data(self, X, y, *, sensitive_features, control_features=None):
        _, y_train, sf_train, _ = _validate_and_reformat_input(
            X,
            y,
            sensitive_features=sensitive_features,
            control_features=control_features,
        )
        super().load_data(X, y_train, sensitive_features=sf_train)

        self._classes = sorted(np.unique(y_train))
        group_vals = self.tags[_GROUP_ID].unique()
        self._group_vals = group_vals

        # One event column per label: event = "label=<y>" restricted to samples where Y=y
        # We encode: eo_label=<y> has the true label as value (so groupby isolates Y=y rows)
        self._eo_event_cols = []
        for cls in self._classes:
            col = f"eo_label={cls}"
            # Only samples where Y == cls get the event; others get NaN (ignored by fairlearn)
            self.tags[col] = y_train.where(y_train == cls).apply(
                lambda v: f"{_LABEL}={v}" if pd.notnull(v) else np.nan
            )
            self._eo_event_cols.append(col)

        self._prob_event = {}
        self._prob_group_event = {}
        index_tuples = []
        bound_vals = []

        for col in self._eo_event_cols:
            valid = self.tags[col].notna()
            self._prob_event[col] = (
                self.tags[valid].groupby(col).size() / self.total_samples
            )
            self._prob_group_event[col] = (
                self.tags[valid].groupby([col, _GROUP_ID]).size() / self.total_samples
            )
            for ev in self.tags[col].dropna().unique():
                for g in group_vals:
                    index_tuples.append(("+", f"{col}|{ev}", g))
                    index_tuples.append(("-", f"{col}|{ev}", g))
                    bound_vals.extend([self.eps, self.eps])

        self._index = pd.MultiIndex.from_tuples(
            index_tuples, names=[_SIGN, _EVENT, _GROUP_ID]
        )
        self._bound_series = pd.Series(bound_vals, index=self._index)

        n_classes = len(self._classes)
        n_samples = len(y_train)
        self._utilities = np.zeros((n_samples, n_classes), dtype=np.float64)
        for i, cls in enumerate(self._classes):
            self._utilities[:, i] = (y_train == cls).astype(float)
        self._y_p_indices = {cls: self._classes.index(cls) for cls in self._classes}

    @property
    def index(self):
        return self._index

    def gamma(self, predictor):
        predictions = np.squeeze(predictor(self.X))
        gamma_list = []

        for cls, col in zip(self._classes, self._eo_event_cols):
            pred_cls = (predictions == cls).astype(float)
            valid = self.tags[col].notna()
            tmp = self.tags.loc[valid, [_GROUP_ID, col]].copy()
            tmp[_PREDICTION] = pred_cls[valid]

            mean_event = tmp.groupby(col)[_PREDICTION].mean()
            mean_group_event = tmp.groupby([col, _GROUP_ID])[_PREDICTION].mean()

            upper = (
                self.ratio * mean_group_event
                - mean_event.reindex(mean_group_event.index.get_level_values(0)).values
            )
            lower = (
                -mean_group_event
                + self.ratio
                * mean_event.reindex(mean_group_event.index.get_level_values(0)).values
            )

            upper.index = pd.MultiIndex.from_tuples(
                [(f"{col}|{ev}", g) for (ev, g) in upper.index],
                names=[_EVENT, _GROUP_ID],
            )
            lower.index = upper.index

            g = pd.concat(
                [upper, lower], keys=["+", "-"], names=[_SIGN, _EVENT, _GROUP_ID]
            )
            gamma_list.append(g)

        gamma_final = pd.concat(gamma_list).reindex(self.index).fillna(0.0)
        self._gamma_descr = str(gamma_final)
        return gamma_final

    def bound(self):
        return self._bound_series

    def project_lambda(self, lambda_vec):
        if self.ratio == 1.0:
            lambda_pos = lambda_vec["+"] - lambda_vec["-"]
            lambda_neg = -lambda_pos
            lambda_pos[lambda_pos < 0.0] = 0.0
            lambda_neg[lambda_neg < 0.0] = 0.0
            return pd.concat(
                [lambda_pos, lambda_neg],
                keys=["+", "-"],
                names=[_SIGN, _EVENT, _GROUP_ID],
            )
        return lambda_vec

    def signed_weights(self, lambda_vec):
        signed_weights = pd.Series(0.0, index=self.tags.index)

        for cls, col in zip(self._classes, self._eo_event_cols):
            prob_e = self._prob_event[col]
            prob_ge = self._prob_group_event[col]

            for ev in self.tags[col].dropna().unique():
                ev_key = f"{col}|{ev}"
                pe = float(prob_e.get(ev, 0.0))
                if pe < 1e-8:
                    continue
                try:
                    lp_ev = lambda_vec["+"].loc[ev_key]  # Series indexed by _GROUP_ID
                    lm_ev = lambda_vec["-"].loc[ev_key]
                    le = float((lp_ev - self.ratio * lm_ev).sum()) / pe
                except KeyError:
                    le = 0.0
                    lp_ev = pd.Series(dtype=float)
                    lm_ev = pd.Series(dtype=float)
                for g in self._group_vals:
                    pge = float(prob_ge.get((ev, g), 0.0))
                    if pge < 1e-8:
                        continue
                    lp_g = float(lp_ev.get(g, 0.0))
                    lm_g = float(lm_ev.get(g, 0.0))
                    adjust = le - (self.ratio * lp_g - lm_g) / pge
                    mask = (self.tags[col] == ev) & (self.tags[_GROUP_ID] == g)
                    signed_weights[mask] += (
                        adjust * self._utilities[:, self._y_p_indices[cls]][mask]
                    )

        return signed_weights

    def per_class_rewards(self, lambda_vec):
        """Per-sample, per-class Lagrangian rewards (see GeneralDemographicParityAll).

        For EO the event of class k only contains samples with Y=k, so each
        sample receives constraint pressure on its true-label column only.
        """
        groups = self.tags[_GROUP_ID]
        rewards = pd.DataFrame(0.0, index=self.tags.index, columns=self._classes)

        for cls, col in zip(self._classes, self._eo_event_cols):
            prob_e = self._prob_event[col]
            prob_ge = self._prob_group_event[col]
            for ev in self.tags[col].dropna().unique():
                ev_key = f"{col}|{ev}"
                pe = float(prob_e.get(ev, 0.0))
                if pe < 1e-8:
                    continue
                try:
                    lp_ev = lambda_vec["+"].loc[ev_key]
                    lm_ev = lambda_vec["-"].loc[ev_key]
                except KeyError:
                    continue
                le = float((lp_ev - self.ratio * lm_ev).sum()) / pe
                for g in self._group_vals:
                    pge = float(prob_ge.get((ev, g), 0.0))
                    if pge < 1e-8:
                        continue
                    lp_g = float(lp_ev.get(g, 0.0))
                    lm_g = float(lm_ev.get(g, 0.0))
                    adjust = le - (self.ratio * lp_g - lm_g) / pge
                    mask = (self.tags[col] == ev) & (groups == g)
                    rewards.loc[mask, cls] += adjust

        return rewards

    def __repr__(self):
        return f"GeneralEqualizedOddsAll(eps={self.eps}, ratio={self.ratio})"


# ---------------------------------------------------------------------------------
#  CombinedParityGeneralAll
#  DP + EO applied simultaneously for ALL class labels
# ---------------------------------------------------------------------------------
class CombinedParityGeneralAll(ClassificationMoment):
    """Combined DP + EO constraints for all class labels simultaneously."""

    short_name = "CombinedParityGeneralAll"

    def __init__(
        self,
        *,
        use_dp=True,
        use_eo=True,
        dp_bound=None,
        eo_bound=None,
    ):
        super().__init__()
        if not use_dp and not use_eo:
            raise ValueError("At least one of use_dp or use_eo must be True")
        self.use_dp = use_dp
        self.use_eo = use_eo
        self.dp_bound = dp_bound if dp_bound is not None else _DEFAULT_DIFFERENCE_BOUND
        self.eo_bound = eo_bound if eo_bound is not None else _DEFAULT_DIFFERENCE_BOUND

    def default_objective(self):
        return SymmetricErrorRate()

    def load_data(self, X, y, *, sensitive_features, control_features=None):
        _, y_train, sf_train, _ = _validate_and_reformat_input(
            X,
            y,
            sensitive_features=sensitive_features,
            control_features=control_features,
        )
        super().load_data(X, y_train, sensitive_features=sf_train)

        self._classes = sorted(np.unique(y_train))
        self._group_vals = self.tags[_GROUP_ID].unique()

        # Build sub-constraints
        self._dp = None
        self._eo = None

        if self.use_dp:
            self._dp = GeneralDemographicParityAll(difference_bound=self.dp_bound)
            self._dp.load_data(
                X,
                y,
                sensitive_features=sensitive_features,
                control_features=control_features,
            )

        if self.use_eo:
            self._eo = GeneralEqualizedOddsAll(difference_bound=self.eo_bound)
            self._eo.load_data(
                X,
                y,
                sensitive_features=sensitive_features,
                control_features=control_features,
            )

        # Combine indices
        indices = []
        if self._dp is not None:
            indices.append(self._dp.index.to_frame(index=False))
        if self._eo is not None:
            indices.append(self._eo.index.to_frame(index=False))
        combined = pd.concat(indices, ignore_index=True).drop_duplicates()
        self._index = pd.MultiIndex.from_frame(combined)

        bounds = []
        if self._dp is not None:
            bounds.append(self._dp.bound())
        if self._eo is not None:
            bounds.append(self._eo.bound())
        self._bound_series = pd.concat(bounds)

    @property
    def index(self):
        return self._index

    def bound(self):
        return self._bound_series

    def gamma(self, predictor):
        parts = []
        if self._dp is not None:
            parts.append(self._dp.gamma(predictor))
        if self._eo is not None:
            parts.append(self._eo.gamma(predictor))
        result = pd.concat(parts).reindex(self.index).fillna(0.0)
        self._gamma_descr = str(result)
        return result

    def project_lambda(self, lambda_vec):
        lambda_pos = lambda_vec["+"] - lambda_vec["-"]
        lambda_neg = -lambda_pos
        lambda_pos[lambda_pos < 0.0] = 0.0
        lambda_neg[lambda_neg < 0.0] = 0.0
        return pd.concat(
            [lambda_pos, lambda_neg], keys=["+", "-"], names=[_SIGN, _EVENT, _GROUP_ID]
        )

    def signed_weights(self, lambda_vec):
        weights = pd.Series(0.0, index=self.tags.index)
        if self._dp is not None:
            weights += self._dp.signed_weights(lambda_vec)
        if self._eo is not None:
            weights += self._eo.signed_weights(lambda_vec)
        return weights

    def per_class_rewards(self, lambda_vec):
        """Sum of the DP and EO per-sample, per-class Lagrangian rewards."""
        rewards = pd.DataFrame(0.0, index=self.tags.index, columns=self._classes)
        if self._dp is not None:
            rewards += self._dp.per_class_rewards(lambda_vec)
        if self._eo is not None:
            rewards += self._eo.per_class_rewards(lambda_vec)
        return rewards

    def __repr__(self):
        return (
            f"CombinedParityGeneralAll(use_dp={self.use_dp}, use_eo={self.use_eo}, "
            f"dp_bound={self.dp_bound}, eo_bound={self.eo_bound})"
        )
