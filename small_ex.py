import numpy as np
from aceei_chores import (
    demand_bundles,
    clearing_error,
    tatonnement_loop,
    price_adjustment_loop,
)

# Reproducibility
np.random.seed(0)

# Problem size
N = 4  # number of agents
M = 10  # number of items

# Problem instance
disutilities = np.random.randint(10, 20, size=(N, M))
quantities = np.random.randint(1, 3, size=M)

# Model parameters
beta = 0.04
price_stepsize = 0.05
grid_stepsize = price_stepsize / 2
epsilon = beta / 4

# Initial minimum payments
# initial_payments = np.random.uniform(1.0+beta/4, 1.0+3*beta/4, size=N)
initial_payments = [1, 1.1, 1.2, 1.3]

# # Prices between 0.2 and 0.6 (so bundles can exceed 1)
# prices = np.array([0.25,0.5,0.6,0.7,0.8,1,0.9,1])

# X = demand_bundles(disutilities, initial_payments, prices)
# print(clearing_error(X, quantities))

prices, X, z = tatonnement_loop(
    disutilities, initial_payments, quantities, price_stepsize
)
print(prices, np.sum(np.abs(z)))

prices, chosen_payments, chosen_bundles, z = price_adjustment_loop(
    disutilities,
    initial_payments,
    prices,
    quantities,
    epsilon,
    grid_stepsize=grid_stepsize,
    price_stepsize=price_stepsize / 10,
)
print(chosen_payments, np.sum(np.abs(z)))
print(prices)

X = demand_bundles(
    prices=prices, min_payments=chosen_payments, disutilities=disutilities
)
print(np.sum(np.abs(clearing_error(X, quantities))))
print(quantities)
print(clearing_error(X, quantities))
print(X)
