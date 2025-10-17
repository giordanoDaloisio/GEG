import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from geg.geg import GeneralizedExponentiatedGradient
from geg.constraints import GeneralDemographicParity1, GeneralEqualizedOdds1, CombinedParityGeneral1
from metrics import Metrics
from utils import get_values
import os

def run_experiment(dataset: str, data: pd.DataFrame, n_splits=10):
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

        results.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'statistical_parity': statistical_parity,
            'equal_opportunity': equal_opportunity,
            'average_odds': average_odds
        })

        fold += 1

    return pd.DataFrame(results)

def run_experiment_geg(dataset: str, data: pd.DataFrame, constraint_type: str, n_splits=10):
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
            constraint = GeneralDemographicParity1(y_p=pos_label, difference_bound=0.05)
        elif constraint_type == "eo":
            constraint = GeneralEqualizedOdds1(y_p=pos_label, difference_bound=0.05)
        elif constraint_type == "cp":
            constraint = CombinedParityGeneral1( 
                use_dp=True,
                use_eo=True,
                y_p=pos_label,
                dp_bound=0.005,
                eo_bound=0.005,
                ratio_bound_slack=1e-7
            )
        else:
            raise ValueError("Invalid constraint type. Choose from 'dp', 'eo', or 'cp'.")

        geg = GeneralizedExponentiatedGradient(
            estimator=LogisticRegression(),
            constraints=constraint,
            positive_label=pos_label
        )


        geg.fit(X_train.values.astype(int), y_train.values.astype(int), sensitive_features=X_train[priv_group.keys()].values.astype(int))
        y_pred = geg.predict(X_test.values.astype(int))

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

        results.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'statistical_parity': statistical_parity,
            'equal_opportunity': equal_opportunity,
            'average_odds': average_odds,
            'constraint': constraint_type
        })

        fold += 1

    return pd.DataFrame(results)

if __name__ == "__main__":
    
    # binary_data = ['adult', 'compas.csv', 'german.csv']

    binary_data = ['compas.csv']

    for data in os.listdir('data'):
        if data.endswith('.csv'):
            if data in binary_data:
              dataset_name = data[:-4]
              print(f"Processing dataset: {dataset_name}")
              df = pd.read_csv(os.path.join('data', data), index_col=0)

              print("Running baseline experiment...")
              baseline_results = run_experiment(dataset_name, df)
              os.makedirs('results_baseline_rq2', exist_ok=True)
              baseline_results.to_csv(f'results_baseline_rq2/{dataset_name}_baseline_results.csv', index=False)

              for constraint in ['dp', 'eo', 'cp']:
                  print(f"Running GEG experiment with constraint: {constraint}")
                  geg_results = run_experiment_geg(dataset_name, df, constraint)
                  os.makedirs('results_geg_rq2', exist_ok=True)
                  geg_results.to_csv(f'results_geg_rq2/{dataset_name}_geg_{constraint}_results.csv', index=False)