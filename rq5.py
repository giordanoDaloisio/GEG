from utils import run_experiment, run_experiment_geg, run_experiment_geg_multi
import pandas as pd
import os
from time import time
import numpy as np

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
    # multiclass_data = ["cmc.csv"]
    for data in os.listdir("experiments/data"):
        if data.endswith(".csv"):
            if data in multiclass_data:
                dataset_name = data[:-4]
                print(f"Processing dataset: {dataset_name}")
                df = pd.read_csv(os.path.join("experiments/data", data))

                for constraint in ["dp", "eo", "cp"]:
                    for i in np.arange(0, 0.0001, 0.0005):
                        print(
                            f"Running GEG experiment full with constraint: {constraint} and ratio_bound_slack: {i}"
                        )
                        start_time = time()
                        geg_multi_results = run_experiment_geg_multi(
                            dataset_name, df, constraint, ratio_bound_slack=i
                        )
                        end_time = time()
                        print(
                            f"GEG experiment with constraint {constraint} took {end_time - start_time} seconds."
                        )
                        os.makedirs(
                            "experiments/results_geg_rq5_ratio_bound_slack",
                            exist_ok=True,
                        )
                        os.makedirs(
                            "experiments/results_geg_rq5_ratio_bound_slack/time",
                            exist_ok=True,
                        )
                        geg_multi_results["ratio_bound_slack"] = i
                        if os.path.exists(
                            f"experiments/results_geg_rq5_ratio_bound_slack/{dataset_name}_geg_{constraint}_results.csv"
                        ):
                            existing_results = pd.read_csv(
                                f"experiments/results_geg_rq5_ratio_bound_slack/{dataset_name}_geg_{constraint}_results.csv"
                            )
                            combined_results = pd.concat(
                                [existing_results, geg_multi_results], ignore_index=True
                            )
                            combined_results.to_csv(
                                f"experiments/results_geg_rq5_ratio_bound_slack/{dataset_name}_geg_{constraint}_results.csv",
                                index=False,
                            )
                        else:
                            geg_multi_results.to_csv(
                                f"experiments/results_geg_rq5_ratio_bound_slack/{dataset_name}_geg_{constraint}_results.csv",
                                index=False,
                            )
                        with open(
                            f"experiments/results_geg_rq5/time/{dataset_name}_geg_{constraint}_time.txt",
                            "a",
                        ) as f:
                            f.write(str(end_time - start_time))
