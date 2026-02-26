import numpy as np
import time
import csv
import os
from src.chore_allocation import (
    pEF1_fPO_three_agent_allocation,
    ILP_pEF1_fPO_allocation,
    compute_usw,
    fPO,
    EF1,
    EF_violations,
)


def save_experiment_result(X, D, seed, model, time_CPU, time_elapsed, filename):
    m, n = X.shape
    headers = [
        "seed",
        "model",
        "N",
        "M",
        "USW",
        "EF violations",
        "CPU time",
        "elapsed time",
        "EF1",
        "fPO",
    ]
    file_exists = os.path.isfile(filename)
    with open(filename, mode="a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        if not file_exists:
            writer.writeheader()
        writer.writerow(
            {
                "seed": seed,
                "model": model,
                "N": n,
                "M": m,
                "USW": compute_usw(X, D),
                "EF violations": EF_violations(X, D),
                "CPU time": time_CPU,
                "elapsed time": time_elapsed,
                "EF1": EF1(X, D),
                "fPO": fPO(X, D),
            }
        )


agent_range = [50, 100]
filename = "experiments_ILP.csv"


# chore_range = np.concatenate(
#     [np.arange(1, 11), np.arange(20, 101, 10), np.arange(150, 401, 50)]
# )
chore_range = np.concatenate([np.arange(80, 101, 10), np.arange(150, 401, 50)])

for n in agent_range:
    for m in chore_range:
        for seed in range(10):
            np.random.seed(seed)
            D = np.random.randint(1, 10, size=(m, n)).astype(float)

            start_elapsed = time.perf_counter()
            start_CPU = time.process_time()
            X = ILP_pEF1_fPO_allocation(m, n, D)
            time_CPU = time.process_time() - start_CPU
            time_elapsed = time.perf_counter() - start_elapsed

            save_experiment_result(X, D, seed, "ILP", time_CPU, time_elapsed, filename)
    chore_range = np.concatenate(
        [np.arange(1, 11), np.arange(20, 101, 10), np.arange(150, 401, 50)]
    )
