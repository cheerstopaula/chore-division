import numpy as np
from src.chore_allocation import (
    pEF1_fPO_three_agent_allocation,
    ILP_pEF1_fPO_allocation,
    fPO,
    EF1,
    EF_violations,
)

np.random.seed(0)

# Small 3 agent example
m, n = 6, 3
D = np.random.randint(1, 6, size=(m, n)).astype(float)
D[1, 1], D[5, 1] = 3, 1


X = pEF1_fPO_three_agent_allocation(m, n, D)

print(X)
print("Allocation is fPO:", fPO(X, D))
print("Allocation is EF1:", EF1(X, D))
print("EF violations:", EF_violations(X, D))


# Larger example

m, n = 80, 9
D = np.random.randint(1, 6, size=(m, n)).astype(float)

X = ILP_pEF1_fPO_allocation(m, n, D)

print(X)
print("Allocation is fPO:", fPO(X, D))
print("Allocation is EF1:", EF1(X, D))
print("EF violations:", EF_violations(X, D))
