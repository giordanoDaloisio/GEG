import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from geg.geg import GeneralizedExponentiatedGradient
from geg.constraints import GeneralDemographicParity1, GeneralEqualizedOdds1, CombinedParityGeneral1
from metrics import Metrics
from blackbox.balancers import MulticlassBalancer
from demv import DEMV
from ml_debiaser.reduce_to_binary import Reduce2Binary

np.random.seed(42)

def get_values(dataset: str):

    '''
    Input: dataset name
    Returns: the label name, positive prediction, privileged group, unprivileged group
    '''

    if "adult" in dataset:
        return ("income", 1, {'sex': 1}, {'sex': 0})
    if "cmc" in dataset:
        return ("contr_use", 2, {'wife_religion': 1}, {'wife_religion': 0})
    if "compas" in dataset:
        return ("two_year_recid", 0, {'race': 1}, {'race': 0})
    if "german" in dataset:
        return ("credit", 1, {'age': 1}, {'age': 0})
    if "crime" in dataset:
        return ("ViolentCrimesClass", 100, {'black_people': 1}, {'black_people': 0})
    if "drug" in dataset:
        return ("y", 0, {'race': 1}, {'race': 0})
    if "law" in dataset:
        return ("gpa", 2, {'race': 1}, {'race': 0})
    if "obesity" in dataset:
        return ("y", 0, {'Age': 1}, {'Age': 0})
    if "park" in dataset:
        return ("score_cut", 0, {'sex': 1}, {'sex': 0})
    if "wine" in dataset:
        return ("quality", 6, {'type': 1}, {'type': 0})
    
def run_experiment(dataset: str, data: pd.DataFrame, n_splits=10, model_name='lr'):
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

        if model_name == 'rf':
            model = RandomForestClassifier()
        elif model_name == 'svm':
            # Use balanced class weights for better handling of imbalanced data
            model = SVC(class_weight='balanced')
        elif model_name == 'mlp':
            model = MLPClassifier()
        elif model_name == 'xgb':
            model = GradientBoostingClassifier()
        elif model_name == 'lr':
            model = LogisticRegression()
        else:
            raise ValueError("Invalid model type. Choose from 'rf', 'svm', 'mlp', 'xgb', or 'lr'.")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        test_data = X_test.copy()
        test_data[label] = y_test
        test_data['pred'] = y_pred

        metrics = Metrics(test_data, 'pred', label, pos_label)
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

        results.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'statistical_parity': statistical_parity,
            'equal_opportunity': equal_opportunity,
            'average_odds': average_odds,
            'wc_statistical_parity': wc_statistical_parity,
            'wc_equal_opportunity': wc_equal_opportunity,
            'wc_average_odds': wc_average_odds,
            'constraint': model_name
        })

        fold += 1

    return pd.DataFrame(results)

def run_experiment_geg(dataset: str, data: pd.DataFrame, constraint_type: str, n_splits=10, model_name='lr'):
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
            constraint = GeneralDemographicParity1(y_p=pos_label, difference_bound=0.005)
        elif constraint_type == "eo":
            constraint = GeneralEqualizedOdds1(y_p=pos_label, difference_bound=0.005)
        elif constraint_type == "cp":
            constraint = CombinedParityGeneral1( 
                use_dp=True,
                use_eo=True,
                y_p=pos_label,
                dp_bound=0.05,
                eo_bound=0.05,
                ratio_bound_slack=1e-7
            )
        else:
            raise ValueError("Invalid constraint type. Choose from 'dp', 'eo', or 'cp'.")

        if model_name == 'rf':
            estimator = RandomForestClassifier()
        elif model_name == 'svm':
            # Use balanced class weights and probability estimates for better handling of imbalanced data
            estimator = SVC(class_weight='balanced', probability=True)
        elif model_name == 'mlp':
            estimator = MLPClassifier()
        elif model_name == 'xgb':
            estimator = GradientBoostingClassifier()
        elif model_name == 'lr':
            estimator = LogisticRegression()
        else:
            raise ValueError("Invalid model type. Choose from 'rf', 'svm', 'mlp', 'xgb', or 'lr'.")

        geg = GeneralizedExponentiatedGradient(
            estimator=estimator,
            constraints=constraint,
            eps=1e-5,
            positive_label=pos_label
        )

        geg.fit(X_train.values, y_train.values, sensitive_features=X_train[priv_group.keys()].values)
        y_pred = geg.predict(X_test.values)

        test_data = X_test.copy()
        test_data[label] = y_test
        test_data['pred'] = y_pred

        metrics = Metrics(test_data, 'pred', label, pos_label)
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

        results.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'statistical_parity': statistical_parity,
            'equal_opportunity': equal_opportunity,
            'average_odds': average_odds,
            'wc_statistical_parity': wc_statistical_parity,
            'wc_equal_opportunity': wc_equal_opportunity,
            'wc_average_odds': wc_average_odds,
            'constraint': constraint_type
        })

        fold += 1

    return pd.DataFrame(results)

def run_experiment_blackbox(dataset: str, data: pd.DataFrame, n_splits=10):
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
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test.values)

        test_data = X_test.copy()
        test_data[label] = y_test
        test_data['pred'] = y_pred

        print(test_data['pred'].value_counts(), test_data['pred'].shape)
        print(test_data[label].value_counts(), test_data[label].shape)
        pb = MulticlassBalancer(y = label, y_ = 'pred', a = list(priv_group.keys())[0], data = test_data)
        y_adj = pb.adjust(cv = True, summary = False)
        print(y_adj)
        test_data['pred'] = y_adj

        metrics = Metrics(test_data, 'pred', label, pos_label)
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

        results.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'statistical_parity': statistical_parity,
            'equal_opportunity': equal_opportunity,
            'average_odds': average_odds,
            'wc_statistical_parity': wc_statistical_parity,
            'wc_equal_opportunity': wc_equal_opportunity,
            'wc_average_odds': wc_average_odds,
        })

        fold += 1

    return pd.DataFrame(results)

def run_experiment_demv(dataset: str, data: pd.DataFrame, n_splits=10):
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
        
        model = LogisticRegression()
        X_train, y_train = demv.fit_transform(X_train, y_train)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test.values)

        test_data = X_test.copy()
        test_data[label] = y_test
        test_data['pred'] = y_pred

        metrics = Metrics(test_data, 'pred', label, pos_label)
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

        results.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'statistical_parity': statistical_parity,
            'equal_opportunity': equal_opportunity,
            'average_odds': average_odds,
            'wc_statistical_parity': wc_statistical_parity,
            'wc_equal_opportunity': wc_equal_opportunity,
            'wc_average_odds': wc_average_odds,
        })

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
        
        model = LogisticRegression()
        y_one_hot = pd.get_dummies(y_train).values
        y_one_hot = r2b.fit(y_one_hot, group_feature=X_train[list(priv_group.keys())[0]].values)
        y_train = np.argmax(y_one_hot, axis=1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test.values)

        test_data = X_test.copy()
        test_data[label] = y_test
        test_data['pred'] = y_pred

        metrics = Metrics(test_data, 'pred', label, pos_label)
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

        results.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'statistical_parity': statistical_parity,
            'equal_opportunity': equal_opportunity,
            'average_odds': average_odds,
            'wc_statistical_parity': wc_statistical_parity,
            'wc_equal_opportunity': wc_equal_opportunity,
            'wc_average_odds': wc_average_odds,
        })

        fold += 1

    return pd.DataFrame(results)