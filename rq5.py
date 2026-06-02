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
                    for i in np.arange(0.00, 0.1, 0.005):
                        print(
                            f"Running GEG experiment full with constraint: {constraint} and difference_bound: {i}"
                        )
                        start_time = time()
                        geg_multi_results = run_experiment_geg_multi(
                            dataset_name, df, constraint, difference_bound=i
                        )
                        end_time = time()
                        print(
                            f"GEG experiment with constraint {constraint} took {end_time - start_time} seconds."
                        )
                        os.makedirs("experiments/results_geg_rq5", exist_ok=True)
                        os.makedirs("experiments/results_geg_rq5/time", exist_ok=True)
                        geg_multi_results["difference_bound"] = i
                        if os.path.exists(
                            f"experiments/results_geg_rq5/{dataset_name}_geg_{constraint}_results.csv"
                        ):
                            existing_results = pd.read_csv(
                                f"experiments/results_geg_rq5/{dataset_name}_geg_{constraint}_results.csv"
                            )
                            combined_results = pd.concat(
                                [existing_results, geg_multi_results], ignore_index=True
                            )
                            combined_results.to_csv(
                                f"experiments/results_geg_rq5/{dataset_name}_geg_{constraint}_results.csv",
                                index=False,
                            )
                        else:
                            geg_multi_results.to_csv(
                                f"experiments/results_geg_rq5/{dataset_name}_geg_{constraint}_results.csv",
                                index=False,
                            )
                        with open(
                            f"experiments/results_geg_rq5/time/{dataset_name}_geg_{constraint}_time.txt",
                            "a",
                        ) as f:
                            f.write(str(end_time - start_time))
