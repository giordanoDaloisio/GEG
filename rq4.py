import pandas as pd
from utils import (
    get_values,
    run_experiment,
    run_experiment_geg,
    run_experiment_geg_multi,
)
import os

if __name__ == "__main__":

    multiclass_data = [
        "cmc.csv",
        "crime.csv",
        "drug.csv",
        "law.csv",
        "park.csv",
        "wine.csv",
        "obesity.csv",
    ]
    # multiclass_data = ['park.csv']
    for data in os.listdir("experiments/data"):
        if data.endswith(".csv"):
            if data in multiclass_data:
                dataset_name = data[:-4]
                print(f"Processing dataset: {dataset_name}")
                df = pd.read_csv(os.path.join("experiments/data", data))

                for model in ["rf", "xgb"]:
                    #   print(f"Running experiment with model: {model}")
                    #   model_results = run_experiment(dataset_name, df, model_name=model)
                    #   os.makedirs('experiments/results_models', exist_ok=True)
                    #   model_results.to_csv(f'experiments/results_models/{dataset_name}_{model}_results.csv', index=False)

                    for constraint in ["dp", "eo", "cp"]:
                        # print(f"Running GEG experiment with constraint: {constraint}")
                        # geg_results = run_experiment_geg(dataset_name, df, constraint, model_name=model)
                        # os.makedirs('experiments/results_geg_rq4', exist_ok=True)
                        # geg_results.to_csv(f'experiments/results_geg_rq4/{dataset_name}_geg_{constraint}_{model}_results.csv', index=False)
                        print(
                            f"Running GEG experiment full with constraint: {constraint} and model: {model}"
                        )
                        geg_multi_results = run_experiment_geg_multi(
                            dataset_name, df, constraint, model_name=model
                        )
                        os.makedirs("experiments/results_geg_multi_rq4", exist_ok=True)
                        geg_multi_results.to_csv(
                            f"experiments/results_geg_multi_rq4/{dataset_name}_geg_{constraint}_{model}_results.csv",
                            index=False,
                        )
