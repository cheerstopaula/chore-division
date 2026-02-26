import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt
import time


# ----------------------------
# Demand computation
# ----------------------------


def demand_bundles(utilities, budgets, prices):
    N, M = utilities.shape
    X = np.zeros((N, M), dtype=int)

    model = gp.Model()
    model.setParam("OutputFlag", 0)

    x = model.addVars(M, vtype=GRB.BINARY, name="x")

    budget_constr = model.addConstr(
        gp.quicksum(prices[j] * x[j] for j in range(M)) <= 0, name="budget"
    )

    for i in range(N):

        model.setObjective(
            gp.quicksum(utilities[i, j] * x[j] for j in range(M)), GRB.MAXIMIZE
        )

        budget_constr.RHS = float(budgets[i])

        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"No feasible solution for agent {i}")

        for j in range(M):
            X[i, j] = int(x[j].X)

    return X


def clearing_error(X, quantities):
    total_demand = X.sum(axis=0)
    return total_demand - quantities


# ----------------------------
# Tatonnement loop
# ----------------------------


def tatonnement_loop(
    utilities, budgets, quantities, price_stepsize, max_iter=200, tol=1e-9
):
    N, M = utilities.shape
    prices = np.zeros(M)

    history = []

    start_time = time.time()

    for it in range(max_iter):

        X = demand_bundles(utilities, budgets, prices)
        z = clearing_error(X, quantities)

        error_value = np.sum(np.abs(z))
        history.append(error_value)

        print(f"Tatonnement iter {it}: {error_value}")

        if np.max(np.abs(z)) <= tol:
            break

        prices = prices + price_stepsize * z
        prices = np.maximum(prices, 0.0)

    elapsed = time.time() - start_time

    return prices, z, history, elapsed


# ----------------------------
# Candidate bundle generation
# ----------------------------


def generate_candidate_bundles(utilities, budgets, prices, epsilon, stepsize):
    N, _ = utilities.shape

    candidates = []
    budget_grid = []

    for i in range(N):

        base = budgets[i]
        grid = np.arange(base - epsilon, base + epsilon + 1e-9, stepsize)

        bundles_i = []
        budget_i = []

        for b in grid:
            X_i = demand_bundles(utilities[i : i + 1], np.array([b]), prices)[0]

            bundles_i.append(X_i)
            budget_i.append(b)

        candidates.append(np.array(bundles_i))
        budget_grid.append(np.array(budget_i))

    return candidates, budget_grid


# ----------------------------
# Global ILP
# ----------------------------


def minimize_clearing_error(candidates, budget_grid, original_budgets, quantities):
    N = len(candidates)
    M = len(quantities)

    model = gp.Model()
    model.setParam("OutputFlag", 0)

    y = {}

    for i in range(N):
        for k in range(candidates[i].shape[0]):
            y[i, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}")

    for i in range(N):
        model.addConstr(
            gp.quicksum(y[i, k] for k in range(candidates[i].shape[0])) == 1
        )

    D = {}
    for j in range(M):
        D[j] = gp.quicksum(
            candidates[i][k, j] * y[i, k]
            for i in range(N)
            for k in range(candidates[i].shape[0])
        )

    z_plus = model.addVars(M, lb=0)
    z_minus = model.addVars(M, lb=0)

    for j in range(M):
        model.addConstr(D[j] - quantities[j] == z_plus[j] - z_minus[j])

    clearing_obj = gp.quicksum(z_plus[j] + z_minus[j] for j in range(M))

    deviation_obj = gp.quicksum(
        abs(budget_grid[i][k] - original_budgets[i]) * y[i, k]
        for i in range(N)
        for k in range(candidates[i].shape[0])
    )

    model.setObjectiveN(clearing_obj, index=0, priority=2)
    model.setObjectiveN(deviation_obj, index=1, priority=1)

    model.optimize()

    chosen_bundles = []
    chosen_budgets = []

    for i in range(N):
        for k in range(candidates[i].shape[0]):
            if y[i, k].X > 0.5:
                chosen_bundles.append(candidates[i][k])
                chosen_budgets.append(budget_grid[i][k])
                break

    return np.array(chosen_bundles), np.array(chosen_budgets)


# ----------------------------
# Price adjustment loop
# ----------------------------


def price_adjustment_loop(
    utilities,
    original_budgets,
    prices,
    quantities,
    epsilon,
    grid_stepsize,
    price_stepsize,
    max_iter=200,
    tol=1e-9,
):

    history = []

    start_time = time.time()

    for it in range(max_iter):

        candidates, budget_grid = generate_candidate_bundles(
            utilities, original_budgets, prices, epsilon, grid_stepsize
        )

        chosen_bundles, chosen_budgets = minimize_clearing_error(
            candidates, budget_grid, original_budgets, quantities
        )

        z = clearing_error(chosen_bundles, quantities)

        error_value = np.sum(np.abs(z))
        history.append(error_value)

        print(f"Adjustment iter {it}: {error_value}")

        if np.max(np.abs(z)) <= tol:
            break

        prices = prices + price_stepsize * z
        prices = np.maximum(prices, 0.0)

    elapsed = time.time() - start_time

    return prices, chosen_budgets, chosen_bundles, z, history, elapsed


# ----------------------------
# MAIN SCRIPT
# ----------------------------

np.random.seed(0)

N = 20
M = 80

utilities = np.random.randint(1, 7, size=(N, M))
quantities = np.random.randint(1, 3, size=M)

beta = 0.08
price_stepsize = 0.002
grid_stepsize = price_stepsize
epsilon = beta / 4

initial_payments = np.random.uniform(1.0 + beta / 4, 1.0 + 3 * beta / 4, size=N)


# --- Tatonnement warm-up ---

prices, z, tat_history, tat_time = tatonnement_loop(
    utilities, initial_payments, quantities, price_stepsize, max_iter=100
)

print("\nTatonnement time:", tat_time)


# --- Price adjustment phase ---

prices, chosen_payments, chosen_bundles, z, adj_history, adj_time = (
    price_adjustment_loop(
        utilities,
        initial_payments,
        prices,
        quantities,
        epsilon,
        grid_stepsize,
        price_stepsize / 10,
    )
)

print("Adjustment time:", adj_time)


# ----------------------------
# Plot combined history
# ----------------------------

plt.figure()

# Tatonnement (dotted)
plt.plot(
    range(len(tat_history)),
    tat_history,
    linestyle="dotted",
    label="Tatonnement (warm-up)",
)

# Adjustment (solid, continues index)
offset = len(tat_history)
plt.plot(
    range(offset, offset + len(adj_history)), adj_history, label="Price adjustment"
)

plt.xlabel("Iteration")
plt.ylabel("Sum of absolute clearing error")
plt.title("Clearing error over iterations")
plt.legend()

plt.show()
