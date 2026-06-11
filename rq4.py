import pandas as pd
from utils import (
    get_values,
    run_experiment,
    run_experiment_geg,
    run_experiment_geg_multi,
)
import os

# Oracle regularization per base model. Strong learners (RF/XGB) fit the train
# set perfectly, which makes every fairness constraint look satisfied on train
# (e.g. TPR=1 for every group), so the dual variables never move and GEG
# degenerates to the baseline. A mildly regularized oracle keeps the constraint
# violations visible during training so the correction transfers to test, at a
# small accuracy cost. class_weight="balanced" must NOT be used: it rescales
# the cost-sensitive sample weights GEG passes to the oracle.
GEG_ORACLE_PARAMS = {
    "rf": {"min_samples_leaf": 5},
    "xgb": {"max_depth": 3, "learning_rate": 0.1, "min_child_weight": 5},
}

# B = 1/eps bounds the dual variables. The previous 1e-5 (B=1e5) let LP-dual
# lambdas reach 1e5, producing pure-fairness oracle calls whose classifiers
# were noise (train accuracy < 0.3) yet entered the final mixture.
GEG_EPS = 1e-2

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
    for data in os.listdir("experiments/data"):
        if data.endswith(".csv"):
            if data in multiclass_data:
                dataset_name = data[:-4]
                print(f"Processing dataset: {dataset_name}")
                df = pd.read_csv(os.path.join("experiments/data", data))

                for model in ["rf", "xgb"]:
                    # Regenerate the unmitigated baseline with the current
                    # XGBClassifierWrapper: the saved wine/xgb baseline was
                    # produced by an older wrapper and is ~7pt too low.
                    print(f"Running experiment with model: {model}")
                    model_results = run_experiment(dataset_name, df, model_name=model)
                    os.makedirs("experiments/results_models", exist_ok=True)
                    model_results.to_csv(
                        f"experiments/results_models/{dataset_name}_{model}_results.csv",
                        index=False,
                    )

                    for constraint in ["dp", "eo", "cp"]:
                        # print(f"Running GEG experiment with constraint: {constraint}")
                        # geg_results = run_experiment_geg(dataset_name, df, constraint, model_name=model)
                        # os.makedirs('experiments/results_geg_rq4', exist_ok=True)
                        # geg_results.to_csv(f'experiments/results_geg_rq4/{dataset_name}_geg_{constraint}_{model}_results.csv', index=False)
                        print(
                            f"Running GEG experiment full with constraint: {constraint} and model: {model}"
                        )
                        geg_multi_results = run_experiment_geg_multi(
                            dataset_name,
                            df,
                            constraint,
                            model_name=model,
                            difference_bound=0.05,
                            eps=GEG_EPS,
                            estimator_params=GEG_ORACLE_PARAMS[model],
                        )
                        os.makedirs("experiments/results_geg_multi_rq4", exist_ok=True)
                        geg_multi_results.to_csv(
                            f"experiments/results_geg_multi_rq4/{dataset_name}_geg_{constraint}_{model}_results.csv",
                            index=False,
                        )
