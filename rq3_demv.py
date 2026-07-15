import pandas as pd
from utils import run_experiment_demv
import os

if __name__ == "__main__":

    multiclass_data = [
        "cmc.csv",
        "crime.csv",
        "drug.csv",
        "law.csv",
        "obesity.csv",
        "park.csv",
        "wine.csv",
    ]

    for data in os.listdir("experiments/data"):
        if data.endswith(".csv"):
            if data in multiclass_data:
                dataset_name = data[:-4]
                print(f"Processing dataset: {dataset_name}")
                df = pd.read_csv(os.path.join("experiments/data", data))
                for model in ["rf", "xgb"]:
                    print("Running baseline experiment...")
                    baseline_results = run_experiment_demv(
                        dataset_name, df, model_name=model
                    )
                    os.makedirs("experiments/results_demv", exist_ok=True)
                    baseline_results.to_csv(
                        f"experiments/results_demv/{dataset_name}_{model}_results.csv",
                        index=False,
                    )
