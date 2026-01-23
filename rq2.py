import pandas as pd
from utils import run_experiment, run_experiment_geg
import os

if __name__ == "__main__":
    
    binary_data = ['adult.csv', 'compas.csv', 'german.csv']

    for data in os.listdir('experiments/data'):
        if data.endswith('.csv'):
            if data in binary_data:
              dataset_name = data[:-4]
              print(f"Processing dataset: {dataset_name}")
              df = pd.read_csv(os.path.join('experiments/data', data), index_col=0)

              print("Running baseline experiment...")
              baseline_results = run_experiment(dataset_name, df)
              os.makedirs('experiments/results_baseline_rq2', exist_ok=True)
              baseline_results.to_csv(f'experiments/results_baseline_rq2/{dataset_name}_baseline_results.csv', index=False)

              for constraint in ['dp', 'eo', 'cp']:
                  print(f"Running GEG experiment with constraint: {constraint}")
                  geg_results = run_experiment_geg(dataset_name, df, constraint)
                  os.makedirs('experiments/results_geg_rq2', exist_ok=True)
                  geg_results.to_csv(f'experiments/results_geg_rq2/{dataset_name}_geg_{constraint}_results.csv', index=False)