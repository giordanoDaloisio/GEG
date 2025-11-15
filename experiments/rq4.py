import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from geg.geg import GeneralizedExponentiatedGradient
from geg.constraints import GeneralDemographicParity1, GeneralEqualizedOdds1, CombinedParityGeneral1
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from metrics import Metrics
from utils import get_values
import os

def run_experiment(dataset: str, data: pd.DataFrame, model_name, n_splits=10):
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
        else:
            raise ValueError("Invalid model type. Choose from 'rf', 'svm', or 'mlp'.")

        # Standardize features by removing the mean and scaling to unit variance
        # scaler = StandardScaler()
        # X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        # X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

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

def run_experiment_geg(dataset: str, data: pd.DataFrame, constraint_type: str, model_name, n_splits=10):
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
        else:
            raise ValueError("Invalid model type. Choose from 'rf', 'svm', or 'mlp'.")
        
        geg = GeneralizedExponentiatedGradient(
            estimator=estimator,
            constraints=constraint,
            eps=1e-5,
            positive_label=pos_label
        )

        # scaler = StandardScaler()
        # X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        # X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)

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
    
    multiclass_data = ['cmc.csv', 'crime.csv', 'drug.csv', 'law.csv', 'park.csv', 'wine.csv', 'obesity.csv']

    for data in os.listdir('data'):
        if data.endswith('.csv'):
            if data in multiclass_data:   
              dataset_name = data[:-4]
              print(f"Processing dataset: {dataset_name}")
              df = pd.read_csv(os.path.join('data', data))

              for model in ['rf','xgb']:
                  print(f"Running experiment with model: {model}")
                  model_results = run_experiment(dataset_name, df, model)
                  os.makedirs('results_models', exist_ok=True)
                  model_results.to_csv(f'results_models/{dataset_name}_{model}_results.csv', index=False)

                  for constraint in ['dp','eo', 'cp']:
                    print(f"Running GEG experiment with constraint: {constraint}")
                    geg_results = run_experiment_geg(dataset_name, df, constraint, model)
                    os.makedirs('results_geg_rq4', exist_ok=True)
                    geg_results.to_csv(f'results_geg_rq4/{dataset_name}_geg_{constraint}_{model}_results.csv', index=False)