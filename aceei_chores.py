import numpy as np
import gurobipy as gp
from gurobipy import GRB


def demand_bundles(disutilities, min_payments, prices):
    """
    Solve one >= knapsack per agent:
        min   sum_j d[i,j] x_j
        s.t.  sum_j p[j] x_j >= min_payment[i]
              x_j in {0,1}
    """

    N, M = disutilities.shape
    X = np.zeros((N, M), dtype=int)

    # Build model ONCE
    model = gp.Model()
    model.setParam("OutputFlag", 0)

    # Variables
    x = model.addVars(M, vtype=GRB.BINARY, name="x")

    # Add payment constraint (RHS will change per agent)
    payment_constr = model.addConstr(
        gp.quicksum(prices[j] * x[j] for j in range(M)) >= 0, name="payment"
    )

    # Solve per agent
    for i in range(N):

        # Update objective coefficients
        model.setObjective(
            gp.quicksum(disutilities[i, j] * x[j] for j in range(M)), GRB.MINIMIZE
        )

        # Update RHS
        payment_constr.RHS = float(min_payments[i])

        model.optimize()

        if model.status != GRB.OPTIMAL:
            raise RuntimeError(f"No feasible solution for agent {i}")

        # Store solution
        for j in range(M):
            X[i, j] = int(x[j].X)

    return X


def clearing_error(X, quantities):
    """
    Parameters
    ----------
    X : (N, M) numpy array (0/1)
        Allocation / demand matrix
    quantities : (M,) numpy array
        Available supply of each item

    Returns
    -------
    z : (M,) numpy array
        Clearing error vector:
        z[j] = total_demand[j] - quantities[j]
    """
    total_demand = X.sum(axis=0)
    z = total_demand - quantities
    return z


import numpy as np


def generate_candidate_bundles(disutilities, min_payments, prices, epsilon, stepsize):
    N, _ = disutilities.shape

    candidates = []  # list per agent
    payment_grid = []  # store corresponding perturbed payments

    for i in range(N):
        base = min_payments[i]

        grid = np.arange(base - epsilon, base + epsilon + 1e-9, stepsize)
        bundles_i = []
        payments_i = []

        for b in grid:
            X_i = demand_bundles(disutilities[i : i + 1], np.array([b]), prices)[0]

            bundles_i.append(X_i)
            payments_i.append(b)

        candidates.append(np.array(bundles_i))  # shape (K_i, M)
        payment_grid.append(np.array(payments_i))

    return candidates, payment_grid


def minimize_clearing_error(
    candidates, payment_grid, original_min_payments, quantities
):
    N = len(candidates)
    M = len(quantities)

    model = gp.Model()
    model.setParam("OutputFlag", 0)

    # Decision variables y_{i,k}
    y = {}
    for i in range(N):
        K_i = candidates[i].shape[0]
        for k in range(K_i):
            y[i, k] = model.addVar(vtype=GRB.BINARY, name=f"y_{i}_{k}")

    # One bundle per agent
    for i in range(N):
        K_i = candidates[i].shape[0]
        model.addConstr(gp.quicksum(y[i, k] for k in range(K_i)) == 1)

    # Demand per item
    D = {}
    for j in range(M):
        D[j] = gp.quicksum(
            candidates[i][k, j] * y[i, k]
            for i in range(N)
            for k in range(candidates[i].shape[0])
        )

    # L1 clearing error
    z_plus = model.addVars(M, lb=0)
    z_minus = model.addVars(M, lb=0)

    for j in range(M):
        model.addConstr(D[j] - quantities[j] == z_plus[j] - z_minus[j])

    clearing_obj = gp.quicksum(z_plus[j] + z_minus[j] for j in range(M))

    # ---- Tie-breaking deviation term ----
    deviation_obj = gp.quicksum(
        abs(payment_grid[i][k] - original_min_payments[i]) * y[i, k]
        for i in range(N)
        for k in range(candidates[i].shape[0])
    )

    # Multi-objective: priority 1 = clearing error
    model.setObjectiveN(clearing_obj, index=0, priority=2)

    # priority 2 = smallest deviation
    model.setObjectiveN(deviation_obj, index=1, priority=1)

    model.optimize()

    # Extract solution
    chosen_bundles = []
    chosen_payments = []

    for i in range(N):
        K_i = candidates[i].shape[0]
        for k in range(K_i):
            if y[i, k].X > 0.5:
                chosen_bundles.append(candidates[i][k])
                chosen_payments.append(payment_grid[i][k])
                break

    return np.array(chosen_bundles), np.array(chosen_payments)


def tatonnement_loop(
    disutilities,
    original_min_payments,
    quantities,
    price_stepsize,
    max_iter=500,
    tol=1e-9,
):
    """
    Iterative price adjustment with endogenous budget perturbation.

    Returns
    -------
    prices
    chosen_payments
    chosen_bundles
    clearing_error_vector
    """

    N, M = disutilities.shape

    # 1. Initialize prices
    initial_price = np.max(original_min_payments)
    prices = np.full(M, initial_price)

    for it in range(max_iter):

        X = demand_bundles(disutilities, original_min_payments, prices)
        z = clearing_error(X, quantities)

        # print(f"Iteration {it}")
        # print("Prices:", prices)
        print(f"Clearing error iteration {it}:", np.sum(np.abs(z)))
        # print(f"Clearing error iteration {it}:", z)

        # 5. Check convergence
        if np.max(np.abs(z)) <= tol:
            print("Exact clearing achieved.")
            return prices, z

        # 6. Update prices
        prices = prices - price_stepsize * (z - (np.mean(z)))

        # Project to nonnegative prices
        prices = np.maximum(prices, 0.0)

    print("Max iterations reached.")
    return prices, X, z


def price_adjustment_loop(
    disutilities,
    original_min_payments,
    prices,
    quantities,
    epsilon,
    grid_stepsize,
    price_stepsize,
    max_iter=100,
    tol=1e-9,
):
    """
    Iterative price adjustment with endogenous budget perturbation.

    Returns
    -------
    prices
    chosen_payments
    chosen_bundles
    clearing_error_vector
    """

    N, M = disutilities.shape

    for it in range(max_iter):

        # 2. Generate candidate bundles for current prices
        candidates, payment_grid = generate_candidate_bundles(
            disutilities, original_min_payments, prices, epsilon, grid_stepsize
        )

        # 3. Solve global ILP with lexicographic tie-breaking
        chosen_bundles, chosen_payments = minimize_clearing_error(
            candidates, payment_grid, original_min_payments, quantities
        )

        # 4. Compute clearing error
        z = clearing_error(chosen_bundles, quantities)

        # print(f"Iteration {it}")
        # print("Prices:", prices)
        print(f"Clearing error iteration {it}:", np.sum(np.abs(z)))

        # 5. Check convergence
        if np.max(np.abs(z)) <= tol:
            print("Exact clearing achieved.")
            return prices, chosen_payments, chosen_bundles, z

        # 6. Update prices
        prices = prices - price_stepsize * (z - (np.mean(z)))

        # Project to nonnegative prices
        prices = np.maximum(prices, 0.0)

    print("Max iterations reached.")
    return prices, chosen_payments, chosen_bundles, z


# # Reproducibility
# np.random.seed(0)

# # Problem size
# N = 10   # number of agents
# M = 50   # number of items

# # Problem instance
# disutilities = np.random.randint(100, 106, size=(N, M))
# quantities = np.random.randint(1, 3, size=M)

# #Model parameters
# beta = 0.04
# price_stepsize = 0.002
# grid_stepsize = price_stepsize
# epsilon = beta/4

# #Initial minimum payments
# initial_payments = np.random.uniform(1.0+beta/4, 1.0+3*beta/4, size=N)

# # # Prices between 0.2 and 0.6 (so bundles can exceed 1)
# # prices = np.array([0.25,0.5,0.6,0.7,0.8,1,0.9,1])

# # X = demand_bundles(disutilities, initial_payments, prices)
# # print(clearing_error(X, quantities))

# prices, X, z = tatonnement_loop(disutilities,initial_payments,quantities,price_stepsize)
# print(prices,np.sum(np.abs(z)))

# prices, chosen_payments, chosen_bundles, z = price_adjustment_loop(disutilities,initial_payments,prices,quantities,epsilon,grid_stepsize=grid_stepsize, price_stepsize=price_stepsize/10)
# print(chosen_payments,np.sum(np.abs(z)))
# print(prices)

# X= demand_bundles(prices=prices, min_payments=chosen_payments, disutilities=disutilities)
# print(np.sum(np.abs(clearing_error(X, quantities))))
# print(quantities)
# print(clearing_error(X, quantities))
# print(X)
