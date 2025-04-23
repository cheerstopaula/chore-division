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


n = 3
filename = "experiments.csv"


chore_range = np.concatenate(
    [np.arange(1, 11), np.arange(20, 101, 10), np.arange(150, 1001, 50)]
)

for m in chore_range:
    for seed in range(10):
        np.random.seed(seed)
        D = np.random.randint(1, 10, size=(m, n)).astype(float)

        start_elapsed = time.perf_counter()
        start_CPU = time.process_time()
        X = pEF1_fPO_three_agent_allocation(m, n, D)
        time_CPU = time.process_time() - start_CPU
        time_elapsed = time.perf_counter() - start_elapsed

        save_experiment_result(X, D, seed, "3ag", time_CPU, time_elapsed, filename)

        start_elapsed = time.perf_counter()
        start_CPU = time.process_time()
        X = ILP_pEF1_fPO_allocation(m, n, D)
        time_CPU = time.process_time() - start_CPU
        time_elapsed = time.perf_counter() - start_elapsed

        save_experiment_result(X, D, seed, "ILP", time_CPU, time_elapsed, filename)
