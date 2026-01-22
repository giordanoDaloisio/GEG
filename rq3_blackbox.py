import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from metrics import Metrics
from utils import get_values
from balancers import MulticlassBalancer
import os

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
    
    multiclass_data = [
        'park.csv', 'drug.csv', 'obesity.csv', 'wine.csv'
        #'cmc.csv', 'crime.csv',  'law.csv'
        ]

    for data in os.listdir('experiments/data'):
        if data.endswith('.csv'):
            if data in multiclass_data:
              print(data)   
              dataset_name = data[:-4]
              print(f"Processing dataset: {dataset_name}")
              df = pd.read_csv(os.path.join('experiments/data', data))

              print("Running baseline experiment...")
              baseline_results = run_experiment_blackbox(dataset_name, df)
              os.makedirs('experiments/results_blackbox', exist_ok=True)
              baseline_results.to_csv(f'experiments/results_blackbox/{dataset_name}_results.csv', index=False)