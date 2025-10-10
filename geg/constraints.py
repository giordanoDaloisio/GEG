# Our proposed code for Generalized the simple version(for Binary and multi-class)
# ------------------------
# Imports and constants
# ------------------------
import pandas as pd
import numpy as np
from fairlearn.reductions._moments.moment import ClassificationMoment
from fairlearn.reductions._moments.moment import _GROUP_ID, _LABEL, _PREDICTION, _ALL, _EVENT, _SIGN
from fairlearn.utils._input_validation import _validate_and_reformat_input, _MESSAGE_RATIO_NOT_IN_RANGE
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

    def load_data(self, X, y, *, sensitive_features, control_features=None):
        """Load the specified data into the object."""
        _, y_train, sf_train, cf_train = _validate_and_reformat_input(
            X, y,
            sensitive_features=sensitive_features,
            control_features=control_features
        )
        super().load_data(X, y_train, sensitive_features=sf_train)
        self.index = [_ALL]

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

        indicator = (y == self.y_p).astype(float)
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
# GeneralUtilityParity class (multi-class and Binary)
# ------------------------
class GeneralUtilityParity1(ClassificationMoment):
    def __init__(self, *, difference_bound=None, ratio_bound=None, ratio_bound_slack=0.0, y_p=None):
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

    def default_objective(self):
        return GeneralErrorRate1(y_p=self.y_p)

    def load_data(self, X, y: pd.Series, *, sensitive_features: pd.Series, event: pd.Series = None, utilities=None):
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
        self.prob_group_event = self.tags.groupby([_EVENT, _GROUP_ID]).size() / self.total_samples
        signed = pd.concat([self.prob_group_event, self.prob_group_event],
                           keys=["+", "-"],
                           names=[_SIGN, _EVENT, _GROUP_ID])
        self.index = signed.index
        self.default_objective_lambda_vec = None

        event_vals = self.tags[_EVENT].dropna().unique()
        group_vals = self.tags[_GROUP_ID].unique()
        self.pos_basis = pd.DataFrame(index=self.index)
        self.neg_basis = pd.DataFrame(index=self.index)
        self.neg_basis_present = pd.Series(dtype='float64')
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
        expect_group_event = self.tags.groupby([_EVENT, _GROUP_ID])[_PREDICTION].mean().to_frame()

        expect_group_event[_UPPER_BOUND_DIFF] = (
            self.ratio * expect_group_event[_PREDICTION] -
            expect_event[_PREDICTION].reindex(expect_group_event.index.get_level_values(0)).values
        )
        expect_group_event[_LOWER_BOUND_DIFF] = (
            - expect_group_event[_PREDICTION] +
            self.ratio * expect_event[_PREDICTION].reindex(expect_group_event.index.get_level_values(0)).values
        )

        g_signed = pd.concat(
            [expect_group_event[_UPPER_BOUND_DIFF], expect_group_event[_LOWER_BOUND_DIFF]],
            keys=["+", "-"],
            names=[_SIGN, _EVENT, _GROUP_ID]
        )

        self._gamma_descr = str(expect_group_event[[_PREDICTION, _UPPER_BOUND_DIFF, _LOWER_BOUND_DIFF]])
        return g_signed

    def bound(self):
        return pd.Series(self.eps, index=self.index)

    def project_lambda(self, lambda_vec):
        if self.ratio == 1.0:
            lambda_pos = lambda_vec["+"] - lambda_vec["-"]
            lambda_neg = -lambda_pos
            lambda_pos[lambda_pos < 0.0] = 0.0
            lambda_neg[lambda_neg < 0.0] = 0.0
            return pd.concat([lambda_pos, lambda_neg], keys=["+", "-"], names=[_SIGN, _EVENT, _GROUP_ID])
        return lambda_vec

    def signed_weights(self, lambda_vec):
        lambda_event = (lambda_vec["+"] - self.ratio * lambda_vec["-"]).groupby(level=_EVENT).sum() / self.prob_event
        lambda_group_event = (self.ratio * lambda_vec["+"] - lambda_vec["-"]) / self.prob_group_event
        adjust = lambda_event.reindex(lambda_group_event.index.get_level_values(0)).values - lambda_group_event.values
        adjust_series = pd.Series(adjust, index=lambda_group_event.index)

        signed_weights = self.tags.apply(
            lambda row: 0 if pd.isna(row[_EVENT]) else adjust_series[row[_EVENT], row[_GROUP_ID]], axis=1
        )

        utility_diff = self.utilities[:, self.y_p_index]
        return utility_diff * signed_weights

# ---------------------------------------------------------------------------------
#  General DemographicParity1 class, can works for binary and multi-class using y_P
# ---------------------------------------------------------------------------------
class GeneralDemographicParity1(GeneralUtilityParity1):
    short_name = "GeneralDemographicParity1"

    def __init__(self, *, y_p=None, difference_bound=None):
        super().__init__(y_p=y_p,difference_bound=difference_bound)

    def load_data(self, X, y, *, sensitive_features, control_features=None):
        _, y_train, sf_train, cf_train = _validate_and_reformat_input(
            X, y,
            sensitive_features=sensitive_features,
            control_features=control_features
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
        super().__init__(y_p=y_p,difference_bound=difference_bound)

    def load_data(self, X, y, *, sensitive_features, control_features=None):
        _, y_train, sf_train, cf_train = _validate_and_reformat_input(
            X,
            y,
            sensitive_features=sensitive_features,
            control_features=control_features
        )

        # Define the event as the label itself: for each y_i, event = "label=y_i"
        base_event = y_train.apply(lambda v: _LABEL + "=" + str(v))
        event = _merge_event_and_control_columns(base_event, cf_train)

        super().load_data(X, y_train, event=event, sensitive_features=sf_train)
