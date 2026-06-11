import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from geg.geg import GeneralizedExponentiatedGradient
from geg.constraints import (
    GeneralDemographicParity1,
    GeneralEqualizedOdds1,
    CombinedParityGeneral1,
    GeneralDemographicParityAll,
    GeneralEqualizedOddsAll,
    CombinedParityGeneralAll,
)
from metrics import Metrics
from blackbox.balancers import MulticlassBalancer
from demv import DEMV
from ml_debiaser.reduce_to_binary import Reduce2Binary

np.random.seed(42)


class XGBClassifierWrapper(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper around ``xgboost.XGBClassifier``.

    ``XGBClassifier`` requires class labels to be contiguous integers ``0..K-1``,
    but the datasets here use arbitrary label values (e.g. wine quality ``[4,5,6,7]``,
    crime ``[100,200,...]``) and GEG relabels targets to those *original* values in its
    oracle. This wrapper label-encodes ``y`` on ``fit`` and decodes predictions back to
    the original labels on ``predict``, while forwarding ``sample_weight`` (which GEG's
    oracle passes on every refit). Being a ``BaseEstimator`` keeps it cloneable, as GEG
    relies on ``sklearn.clone``.
    """

    def __init__(
        self,
        n_estimators=100,
        random_state=42,
        max_depth=None,
        learning_rate=None,
        min_child_weight=None,
        subsample=None,
        reg_lambda=None,
    ):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_child_weight = min_child_weight
        self.subsample = subsample
        self.reg_lambda = reg_lambda

    def fit(self, X, y, sample_weight=None):
        self._encoder = LabelEncoder()
        y_enc = self._encoder.fit_transform(y)
        self.classes_ = self._encoder.classes_
        # Optional regularization knobs; None falls back to the XGBoost default
        # so existing callers keep byte-identical behavior.
        extra = {
            k: v
            for k, v in {
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "min_child_weight": self.min_child_weight,
                "subsample": self.subsample,
                "reg_lambda": self.reg_lambda,
            }.items()
            if v is not None
        }
        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            tree_method="hist",
            **extra,
        )
        self._model.fit(X, y_enc, sample_weight=sample_weight)
        return self

    def predict(self, X):
        return self._encoder.inverse_transform(self._model.predict(X))


def get_values(dataset: str):
    """
    Input: dataset name
    Returns: the label name, positive prediction, privileged group, unprivileged group
    """

    if "adult" in dataset:
        return ("income", 1, {"sex": 1}, {"sex": 0})
    if "cmc" in dataset:
        return ("contr_use", 2, {"wife_religion": 1}, {"wife_religion": 0})
    if "compas" in dataset:
        return ("two_year_recid", 0, {"race": 1}, {"race": 0})
    if "german" in dataset:
        return ("credit", 1, {"age": 1}, {"age": 0})
    if "crime" in dataset:
        return ("ViolentCrimesClass", 100, {"black_people": 1}, {"black_people": 0})
    if "drug" in dataset:
        return ("y", 0, {"race": 1}, {"race": 0})
    if "law" in dataset:
        return ("gpa", 2, {"race": 1}, {"race": 0})
    if "obesity" in dataset:
        return ("y", 0, {"Age": 1}, {"Age": 0})
    if "park" in dataset:
        return ("score_cut", 0, {"sex": 1}, {"sex": 0})
    if "wine" in dataset:
        return ("quality", 6, {"type": 1}, {"type": 0})


def run_experiment(dataset: str, data: pd.DataFrame, n_splits=10, model_name="lr"):
    label, pos_label, priv_group, unpriv_group = get_values(dataset)
    X = data.drop(columns=[label])
    y = data[label]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    results = []

    for train_index, test_index in kf.split(X):
        print(f"Fold {fold}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if model_name == "rf":
            model = RandomForestClassifier(random_state=42)
        elif model_name == "svm":
            # Use balanced class weights for better handling of imbalanced data
            model = SVC(class_weight="balanced", random_state=42)
        elif model_name == "mlp":
            model = MLPClassifier(random_state=42)
        elif model_name == "xgb":
            model = XGBClassifierWrapper()
        elif model_name == "lr":
            model = LogisticRegression(random_state=42)
        else:
            raise ValueError(
                "Invalid model type. Choose from 'rf', 'svm', 'mlp', 'xgb', or 'lr'."
            )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_data = X_test.copy()
        test_data[label] = y_test
        test_data["pred"] = y_pred

        metrics = Metrics(test_data, "pred", label, pos_label)
        accuracy = metrics.accuracy()
        precision = metrics.precision()
        recall = metrics.recall()
        f1_score = metrics.f1()
        statistical_parity = metrics.statistical_parity(unpriv_group)
        equal_opportunity = metrics.equal_opportunity(unpriv_group)
        average_odds = metrics.average_odds(unpriv_group)
        wc_statistical_parity = metrics.wc_statistical_parity(unpriv_group)
        wc_equal_opportunity = metrics.wc_equal_opportunity(unpriv_group)
        wc_average_odds = metrics.wc_average_odds(unpriv_group)

        results.append(
            {
                "fold": fold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "statistical_parity": statistical_parity,
                "equal_opportunity": equal_opportunity,
                "average_odds": average_odds,
                "wc_statistical_parity": wc_statistical_parity,
                "wc_equal_opportunity": wc_equal_opportunity,
                "wc_average_odds": wc_average_odds,
                "constraint": model_name,
            }
        )

        fold += 1

    return pd.DataFrame(results)


def run_experiment_geg(
    dataset: str,
    data: pd.DataFrame,
    constraint_type: str,
    n_splits=10,
    model_name="lr",
    difference_bound=0.005,
    ratio_bound_slack=1e-7,
):
    label, pos_label, priv_group, unpriv_group = get_values(dataset)
    X = data.drop(columns=[label])
    y = data[label]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    results = []

    for train_index, test_index in kf.split(X):
        print(f"Fold {fold}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if constraint_type == "dp":
            constraint = GeneralDemographicParity1(
                y_p=pos_label, difference_bound=difference_bound
            )
        elif constraint_type == "eo":
            constraint = GeneralEqualizedOdds1(
                y_p=pos_label, difference_bound=difference_bound
            )
        elif constraint_type == "cp":
            constraint = CombinedParityGeneral1(
                use_dp=True,
                use_eo=True,
                y_p=pos_label,
                dp_bound=0.05,
                eo_bound=0.05,
                ratio_bound_slack=ratio_bound_slack,
            )
        else:
            raise ValueError(
                "Invalid constraint type. Choose from 'dp', 'eo', or 'cp'."
            )

        if model_name == "rf":
            estimator = RandomForestClassifier(random_state=42)
        elif model_name == "svm":
            # Use balanced class weights and probability estimates for better handling of imbalanced data
            estimator = SVC(class_weight="balanced", probability=True, random_state=42)
        elif model_name == "mlp":
            estimator = MLPClassifier(random_state=42)
        elif model_name == "xgb":
            estimator = XGBClassifierWrapper()
        elif model_name == "lr":
            estimator = LogisticRegression(random_state=42)
        else:
            raise ValueError(
                "Invalid model type. Choose from 'rf', 'svm', 'mlp', 'xgb', or 'lr'."
            )

        geg = GeneralizedExponentiatedGradient(
            estimator=estimator,
            constraints=constraint,
            eps=1e-5,
            positive_label=pos_label,
        )

        geg.fit(
            X_train.values,
            y_train.values,
            sensitive_features=X_train[priv_group.keys()].values,
        )
        y_pred = geg.predict(X_test.values)

        test_data = X_test.copy()
        test_data[label] = y_test
        test_data["pred"] = y_pred

        metrics = Metrics(test_data, "pred", label, pos_label)
        accuracy = metrics.accuracy()
        precision = metrics.precision()
        recall = metrics.recall()
        f1_score = metrics.f1()
        statistical_parity = metrics.statistical_parity(unpriv_group)
        equal_opportunity = metrics.equal_opportunity(unpriv_group)
        average_odds = metrics.average_odds(unpriv_group)
        wc_statistical_parity = metrics.wc_statistical_parity(unpriv_group)
        wc_equal_opportunity = metrics.wc_equal_opportunity(unpriv_group)
        wc_average_odds = metrics.wc_average_odds(unpriv_group)

        results.append(
            {
                "fold": fold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "statistical_parity": statistical_parity,
                "equal_opportunity": equal_opportunity,
                "average_odds": average_odds,
                "wc_statistical_parity": wc_statistical_parity,
                "wc_equal_opportunity": wc_equal_opportunity,
                "wc_average_odds": wc_average_odds,
                "constraint": constraint_type,
            }
        )

        fold += 1

    return pd.DataFrame(results)


def run_experiment_geg_multi(
    dataset: str,
    data: pd.DataFrame,
    constraint_type: str,
    n_splits=10,
    model_name="lr",
    difference_bound=0.05,
    ratio_bound_slack=1e-7,
    eta=2.0,
    eps=1e-5,
    estimator_params=None,
):
    label, pos_label, priv_group, unpriv_group = get_values(dataset)
    X = data.drop(columns=[label])
    y = data[label]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    results = []

    for train_index, test_index in kf.split(X):
        print(f"Fold {fold}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if constraint_type == "dp":
            constraint = GeneralDemographicParityAll(
                difference_bound=difference_bound,
            )
        elif constraint_type == "eo":
            constraint = GeneralEqualizedOddsAll(
                difference_bound=difference_bound,
            )
        elif constraint_type == "cp":
            constraint = CombinedParityGeneralAll(
                use_dp=True,
                use_eo=True,
                dp_bound=difference_bound,
                eo_bound=difference_bound,
            )
        else:
            raise ValueError(
                "Invalid constraint type. Choose from 'dp', 'eo', or 'cp'."
            )

        if model_name == "rf":
            # estimator_params lets callers regularize the oracle (e.g. max_depth,
            # min_samples_leaf, class_weight) so a strong RF stops fitting the
            # relabeled train set perfectly -- which is what makes the fairness
            # constraint transfer to test. Default None preserves prior behavior.
            estimator = RandomForestClassifier(
                random_state=42, **(estimator_params or {})
            )
        elif model_name == "svm":
            # Use balanced class weights and probability estimates for better handling of imbalanced data
            estimator = SVC(class_weight="balanced", probability=True, random_state=42)
        elif model_name == "mlp":
            estimator = MLPClassifier(random_state=42)
        elif model_name == "xgb":
            # estimator_params plays the same role as for RF: regularize the
            # boosting oracle (max_depth, learning_rate, ...) so it cannot
            # memorize the relabeled train set.
            estimator = XGBClassifierWrapper(**(estimator_params or {}))
        elif model_name == "lr":
            estimator = LogisticRegression(random_state=42)
        else:
            raise ValueError(
                "Invalid model type. Choose from 'rf', 'svm', 'mlp', 'xgb', or 'lr'."
            )

        geg = GeneralizedExponentiatedGradient(
            estimator=estimator,
            constraints=constraint,
            eps=eps,
            positive_label=pos_label,
            eta0=eta,
        )

        geg.fit(
            X_train.values,
            y_train.values,
            sensitive_features=X_train[priv_group.keys()].values,
        )
        y_pred = geg.predict(X_test.values)

        test_data = X_test.copy()
        test_data[label] = y_test
        test_data["pred"] = y_pred

        metrics = Metrics(test_data, "pred", label, pos_label)
        accuracy = metrics.accuracy()
        precision = metrics.precision()
        recall = metrics.recall()
        f1_score = metrics.f1()
        statistical_parity = metrics.statistical_parity(unpriv_group)
        equal_opportunity = metrics.equal_opportunity(unpriv_group)
        average_odds = metrics.average_odds(unpriv_group)
        wc_statistical_parity = metrics.wc_statistical_parity(unpriv_group)
        wc_equal_opportunity = metrics.wc_equal_opportunity(unpriv_group)
        wc_average_odds = metrics.wc_average_odds(unpriv_group)

        results.append(
            {
                "fold": fold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "statistical_parity": statistical_parity,
                "equal_opportunity": equal_opportunity,
                "average_odds": average_odds,
                "wc_statistical_parity": wc_statistical_parity,
                "wc_equal_opportunity": wc_equal_opportunity,
                "wc_average_odds": wc_average_odds,
                "constraint": constraint_type,
            }
        )

        fold += 1

    return pd.DataFrame(results)


def run_experiment_blackbox(
    dataset: str, data: pd.DataFrame, n_splits=10, model_name="lr"
):
    label, pos_label, priv_group, unpriv_group = get_values(dataset)
    X = data.drop(columns=[label])
    y = data[label]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    results = []

    for train_index, test_index in kf.split(X):
        print(f"Fold {fold}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if model_name == "lr":
            model = LogisticRegression(random_state=42)
        elif model_name == "rf":
            model = RandomForestClassifier(random_state=42)
        elif model_name == "xgb":
            model = XGBClassifierWrapper()
        else:
            raise ValueError("Invalid model type. Choose from 'lr', 'rf', or 'xgb'.")

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test.values)

        test_data = X_test.copy()
        test_data[label] = y_test
        test_data["pred"] = y_pred

        print(test_data["pred"].value_counts(), test_data["pred"].shape)
        print(test_data[label].value_counts(), test_data[label].shape)
        pb = MulticlassBalancer(
            y=label, y_="pred", a=list(priv_group.keys())[0], data=test_data
        )
        y_adj = pb.adjust(cv=True, summary=False)
        print(y_adj)
        test_data["pred"] = y_adj

        metrics = Metrics(test_data, "pred", label, pos_label)
        accuracy = metrics.accuracy()
        precision = metrics.precision()
        recall = metrics.recall()
        f1_score = metrics.f1()
        statistical_parity = metrics.statistical_parity(unpriv_group)
        equal_opportunity = metrics.equal_opportunity(unpriv_group)
        average_odds = metrics.average_odds(unpriv_group)
        wc_statistical_parity = metrics.wc_statistical_parity(unpriv_group)
        wc_equal_opportunity = metrics.wc_equal_opportunity(unpriv_group)
        wc_average_odds = metrics.wc_average_odds(unpriv_group)

        results.append(
            {
                "fold": fold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "statistical_parity": statistical_parity,
                "equal_opportunity": equal_opportunity,
                "average_odds": average_odds,
                "wc_statistical_parity": wc_statistical_parity,
                "wc_equal_opportunity": wc_equal_opportunity,
                "wc_average_odds": wc_average_odds,
            }
        )

        fold += 1

    return pd.DataFrame(results)


def run_experiment_demv(dataset: str, data: pd.DataFrame, n_splits=10, model_name="lr"):
    label, pos_label, priv_group, unpriv_group = get_values(dataset)
    X = data.drop(columns=[label])
    y = data[label]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    results = []
    demv = DEMV(list(priv_group.keys()))

    for train_index, test_index in kf.split(X):
        print(f"Fold {fold}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        if model_name == "lr":
            model = LogisticRegression(random_state=42)
        elif model_name == "rf":
            model = RandomForestClassifier(random_state=42)
        elif model_name == "xgb":
            model = XGBClassifierWrapper()
        else:
            raise ValueError("Invalid model type. Choose from 'lr', 'rf', or 'xgb'.")

        X_train, y_train = demv.fit_transform(X_train, y_train)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test.values)

        test_data = X_test.copy()
        test_data[label] = y_test
        test_data["pred"] = y_pred

        metrics = Metrics(test_data, "pred", label, pos_label)
        accuracy = metrics.accuracy()
        precision = metrics.precision()
        recall = metrics.recall()
        f1_score = metrics.f1()
        statistical_parity = metrics.statistical_parity(unpriv_group)
        equal_opportunity = metrics.equal_opportunity(unpriv_group)
        average_odds = metrics.average_odds(unpriv_group)
        wc_statistical_parity = metrics.wc_statistical_parity(unpriv_group)
        wc_equal_opportunity = metrics.wc_equal_opportunity(unpriv_group)
        wc_average_odds = metrics.wc_average_odds(unpriv_group)

        results.append(
            {
                "fold": fold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "statistical_parity": statistical_parity,
                "equal_opportunity": equal_opportunity,
                "average_odds": average_odds,
                "wc_statistical_parity": wc_statistical_parity,
                "wc_equal_opportunity": wc_equal_opportunity,
                "wc_average_odds": wc_average_odds,
            }
        )

        fold += 1

    return pd.DataFrame(results)


def run_experiment_r2b(dataset: str, data: pd.DataFrame, n_splits=10):
    label, pos_label, priv_group, unpriv_group = get_values(dataset)
    X = data.drop(columns=[label])
    y = data[label]

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold = 1
    results = []
    r2b = Reduce2Binary(num_classes=len(y.unique()))

    for train_index, test_index in kf.split(X):
        print(f"Fold {fold}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        model = LogisticRegression(random_state=42)
        y_one_hot = pd.get_dummies(y_train).values
        y_one_hot = r2b.fit(
            y_one_hot, group_feature=X_train[list(priv_group.keys())[0]].values
        )
        y_train = np.argmax(y_one_hot, axis=1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test.values)

        test_data = X_test.copy()
        test_data[label] = y_test
        test_data["pred"] = y_pred

        metrics = Metrics(test_data, "pred", label, pos_label)
        accuracy = metrics.accuracy()
        precision = metrics.precision()
        recall = metrics.recall()
        f1_score = metrics.f1()
        statistical_parity = metrics.statistical_parity(unpriv_group)
        equal_opportunity = metrics.equal_opportunity(unpriv_group)
        average_odds = metrics.average_odds(unpriv_group)
        wc_statistical_parity = metrics.wc_statistical_parity(unpriv_group)
        wc_equal_opportunity = metrics.wc_equal_opportunity(unpriv_group)
        wc_average_odds = metrics.wc_average_odds(unpriv_group)

        results.append(
            {
                "fold": fold,
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1_score,
                "statistical_parity": statistical_parity,
                "equal_opportunity": equal_opportunity,
                "average_odds": average_odds,
                "wc_statistical_parity": wc_statistical_parity,
                "wc_equal_opportunity": wc_equal_opportunity,
                "wc_average_odds": wc_average_odds,
            }
        )

        fold += 1

    return pd.DataFrame(results)
