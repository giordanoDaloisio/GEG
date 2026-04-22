from utils import run_experiment, run_experiment_geg, run_experiment_geg_multi
import pandas as pd
import os
from time import time

if __name__ == "__main__":

    multiclass_data = [
        # "cmc.csv",
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

                #   print("Running baseline experiment...")
                #   start_time = time()
                #   baseline_results = run_experiment(dataset_name, df)
                #   end_time = time()
                #   print(f"Baseline experiment took {end_time - start_time} seconds.")
                #   os.makedirs('experiments/results_baseline', exist_ok=True)
                #   os.makedirs('experiments/results_baseline/time', exist_ok=True)
                #   baseline_results.to_csv(f'experiments/results_baseline/{dataset_name}_baseline_results.csv', index=False)
                #   with open(f'experiments/results_baseline/time/{dataset_name}_baseline_time.txt', 'w') as f:
                #       f.write(str(end_time - start_time))

                for constraint in ["dp", "eo", "cp"]:
                    #   print(f"Running GEG experiment with constraint: {constraint}")
                    #   start_time = time()
                    #   geg_results = run_experiment_geg(dataset_name, df, constraint)
                    #   end_time = time()
                    #   print(f"GEG experiment with constraint {constraint} took {end_time - start_time} seconds.")
                    #   os.makedirs('experiments/results_geg_new', exist_ok=True)
                    #   os.makedirs('experiments/results_geg_new/time', exist_ok=True)
                    #   geg_results.to_csv(f'experiments/results_geg_new/{dataset_name}_geg_{constraint}_results.csv', index=False)
                    #   with open(f'experiments/results_geg_new/time/{dataset_name}_geg_{constraint}_time.txt', 'w') as f:
                    #         f.write(str(end_time - start_time))
                    print(f"Running GEG experiment full with constraint: {constraint}")
                    start_time = time()
                    geg_multi_results = run_experiment_geg_multi(
                        dataset_name, df, constraint
                    )
                    end_time = time()
                    print(
                        f"GEG experiment with constraint {constraint} took {end_time - start_time} seconds."
                    )
                    os.makedirs("experiments/results_geg_multi", exist_ok=True)
                    os.makedirs("experiments/results_geg_multi/time", exist_ok=True)
                    geg_multi_results.to_csv(
                        f"experiments/results_geg_multi/{dataset_name}_geg_{constraint}_results.csv",
                        index=False,
                    )
                    with open(
                        f"experiments/results_geg_multi/time/{dataset_name}_geg_{constraint}_time.txt",
                        "w",
                    ) as f:
                        f.write(str(end_time - start_time))
