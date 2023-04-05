import numpy as np
import pandas as pd

original_data = pd.read_excel('UC.xlsx', sheet_name=None)

gendata = original_data['GenData']
loaddata = original_data['Load']

pmin = gendata.iloc[:, 3].values.tolist()
pmax = gendata.iloc[:, 2].values.tolist()
a = gendata.iloc[:, 4].values.tolist()
b = gendata.iloc[:, 5].values.tolist()
c = gendata.iloc[:, 6].values.tolist()

load = loaddata.iloc[:, 1].values.tolist()

def lagrange_relaxation_UC_ED(max_iter=100, tol=0.001):
    """
    Solves the unit commitment and economic dispatch problem using the Lagrange relaxation method.
    Inputs:
        max_iter: the maximum number of iterations to perform.
        tol: the convergence tolerance.
    Outputs:
        PED: the power output after economic dispatch.
        J_star: the optimal objective function value.
        q_star: the optimal total power generation.
    """

    # Initialize Lagrange multipliers to zero.
    lmbd = [load[i] * 0.01 for i in range(168)]

    for it in range(max_iter):
        J_new, q_new, PED_new, lmbd_new = lagrange_relaxation_UC_ED_iteration(lmbd)
        opt_new = abs(J_new - q_new)

        if opt_new < tol:
            print('Converged after', it, 'iterations.')
            print('Objective function value:', J_new)
            print('Total power generation:', q_new)
            break
        else:
            lmbd = lmbd_new

    return J_new, q_new, PED_new

def lagrange_relaxation_UC_ED_iteration(lmbd):
    coefficients = list(zip(a, b, c))
    power_limits = list(zip(pmin, pmax))

    PUsum = np.zeros(168)
    J_iter = 0
    q_iter = 0
    PED = []

    for i in range(168):
        power_gen = economic_dispatch(coefficients, power_limits, load[i])
        PED.append(power_gen)
        PUsum[i] = np.sum(power_gen)
        J_iter += np.sum([cost_function(P, a[j], b[j], c[j]) for j, P in enumerate(power_gen)])
        q_iter += np.sum(power_gen)

    lmbd_new = [lmbd[i] + 0.01 * (PUsum[i] - load[i]) for i in range(168)]

    return J_iter, q_iter, PED, lmbd_new

def cost_function(x, a, b, c):
    return a * x ** 2 + b * x + c

def economic_dispatch(coefficients, power_limits, total_load, epsilon=1e-6, max_iter=1000):
    num_generators = len(coefficients)
    lmbd_low = min(coefficients, key=lambda x: x[1])[1]  # Set lambda lower bound to the minimum linear cost
    lmbd_high = max(coefficients, key=lambda x: x[1])[1]  # Set lambda upper bound to the maximum linear cost

    for i in (num_generators):
        a, b, c = coefficients[i]
        P_min, P_max = power_limits[i]
        power_gen[i] = max(min((lmbd_low - b) / (2 * a), P_max), P_min)

    total_gen = np.sum(power_gen)

    if abs(total_gen - total_load) < epsilon:
        return power_gen

    if total_gen > total_load:
        lmbd_high = lmbd_low
    else:
        lmbd_low = (lmbd_low + lmbd_high) / 2

    return power_gen

def main():
    J_star, q_star, PED = lagrange_relaxation_UC_ED()
    print("Optimal objective function value:", J_star)
    print("Optimal total power generation:", q_star)
    print("Power output after economic dispatch:", PED)

if __name__ == "main":
    main()