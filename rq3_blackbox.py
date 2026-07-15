import pandas as pd
from utils import run_experiment_blackbox
import os

if __name__ == "__main__":

    multiclass_data = [
        "park.csv",
        "drug.csv",
        "obesity.csv",
        "wine.csv",
        "cmc.csv",
        "crime.csv",
        "law.csv",
    ]

    for data in os.listdir("experiments/data"):
        if data.endswith(".csv"):
            if data in multiclass_data:
                print(data)
                dataset_name = data[:-4]
                print(f"Processing dataset: {dataset_name}")
                df = pd.read_csv(os.path.join("experiments/data", data))
                for model in ["rf", "xgb"]:
                    print(f"Running {model} experiment...")
                    baseline_results = run_experiment_blackbox(
                        dataset_name, df, model_name=model
                    )
                    os.makedirs("experiments/results_blackbox", exist_ok=True)
                    baseline_results.to_csv(
                        f"experiments/results_blackbox/{dataset_name}_{model}_results.csv",
                        index=False,
                    )
