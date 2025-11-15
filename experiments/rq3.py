import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from geg.geg import GeneralizedExponentiatedGradient
from geg.constraints import GeneralDemographicParity1, GeneralEqualizedOdds1, CombinedParityGeneral1
from metrics import Metrics
from utils import get_values
from demv import DEMV
from fairlearn.reductions import ExponentiatedGradient, DemographicParity, EqualizedOdds
import os

def run_experiment_eg(dataset: str, data: pd.DataFrame, n_splits=10):
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

        for c in ['dp', 'eo']:
            if c == 'dp':
                constraint = DemographicParity()
            elif c == 'eo':
                constraint = EqualizedOdds()

            model = ExponentiatedGradient(
                LogisticRegression(),
                constraints=constraint,
                eps=0.01
            )

            model.fit(X_train, y_train, sensitive_features=X_train[priv_group.keys()].values)
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
                'average_odds': average_odds,
                'constraint': c
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

        results.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'statistical_parity': statistical_parity,
            'equal_opportunity': equal_opportunity,
            'average_odds': average_odds,
        })

        fold += 1

    return pd.DataFrame(results)

if __name__ == "__main__":
    
    multiclass_data = ['cmc.csv', 'crime.csv', 'drug.csv', 'law.csv', 'obesity.csv', 'park.csv', 'wine.csv']

    for data in os.listdir('data'):
        if data.endswith('.csv'):
            if data in multiclass_data:   
              dataset_name = data[:-4]
              print(f"Processing dataset: {dataset_name}")
              df = pd.read_csv(os.path.join('data', data))

              print("Running baseline experiment...")
              baseline_results = run_experiment_demv(dataset_name, df)
              os.makedirs('results_demv', exist_ok=True)
              baseline_results.to_csv(f'results_demv/{dataset_name}_results.csv', index=False)

    binary_data = ['adult.csv', 'compas.csv', 'german.csv']

    for data in os.listdir('data'):
        if data.endswith('.csv'):
            if data in binary_data:   
              dataset_name = data[:-4]
              print(f"Processing dataset: {dataset_name}")
              df = pd.read_csv(os.path.join('data', data))

              print("Running baseline experiment...")
              baseline_results = run_experiment_eg(dataset_name, df)
              os.makedirs('results_eg', exist_ok=True)
              baseline_results.to_csv(f'results_eg/{dataset_name}_results.csv', index=False)