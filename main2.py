import numpy as np
import pandas as pd
from scipy.optimize import minimize, linprog

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
    # Read data from Excel file.

    # Initialize Lagrange multipliers to zero.
    lmbd = [load[i] * 0.01 for i in range(168)]

    J = 0
    q = 0
    for _ in range(max_iter):
        J_new, q_new, PED_new, lmbd_new = lagrange_relaxation_UC_ED_iteration(lmbd, tol)
        if q_new != 0:
            opt_new = (J_new - q_new) / q_new
        else:
            opt_new = 0
        print(opt_new, J_new, q_new)
        if opt_new < tol:
            print('Converged after', _, 'iterations.')
            print('Objective function value:', J_new)
            print('Total power generation:', q_new)
#            print('Power output after economic dispatch:', PED_new)
            break
        else:
            J = J_new
            q = q_new
            lmbd = lmbd_new

    return J, q, PED_new


def lagrange_relaxation_UC_ED_iteration(lmbd, tol=0.001):
    gen_periods_up = [0] * 11
    P = [[0 for j in range(11)] for i in range(168)]
    U = [[0 for j in range(11)] for i in range(168)]
    PED = [[0 for j in range(11)] for i in range(168)]
    PUsum = [0 for i in range(168)]
    F = [[0 for j in range(11)] for i in range(168)]
    Fprime = [[0 for j in range(11)] for i in range(168)]

    J_iter = 0
    q_iter = 0
    for i in range(168):
        for j in range(11):
            P[i][j] = max(min(max((lmbd[i] - b[j]) / (2 * c[j]), 0), pmax[j]), pmin[j])
            U[i][j] = 1 - min(max(cost_function(P[i][j], a[j], b[j], c[j]) - lmbd[i] * P[i][j], 0), 1)
            P[i][j] = U[i][j] * P[i][j]
            F[i][j] = cost_function(P[i][j], a[j], b[j], c[j])
            Fprime[i][j] = cost_function_derivative(P[i][j], b[j], c[j])
            PUsum[i] += P[i][j] * U[i][j]
            J_iter += F[i][j]
            q_iter += P[i][j]
    lmbd_new = [lmbd[i] + 0.01 * (PUsum[i] - load[i]) for i in range(168)]

    return J_iter, q_iter, P, lmbd_new

def cost_function(x, a, b, c):
    return a * x ** 2 + b * x + c


def cost_function_derivative(x, b, c):
    return 2 * c * x + b


def economic_dispatch(coefficients, power_limits, total_load, epsilon=1e-6, max_iter=1000):
    num_generators = len(coefficients)
    lmbd_low = min(coefficients, key=lambda x: x[1])[1]  # Set lambda lower bound to the minimum linear cost
    lmbd_high = max(coefficients, key=lambda x: x[1])[1]  # Set lambda upper bound to the maximum linear cost

    for _ in range(max_iter):
        power_gen = np.zeros(num_generators)

        for i in range(num_generators):
            a, b, c = coefficients[i]
            P_min, P_max = power_limits[i]
            power_gen[i] = max(min((lmbd_low - b) / (2 * a), P_max), P_min)

        total_gen = np.sum(power_gen)

        if abs(total_gen - total_load) < epsilon:
            break

        if total_gen > total_load:
            lmbd_high = lmbd_low
        else:
            lmbd_low = (lmbd_low + lmbd_high) / 2

    return power_gen

def main():
    J_star, q_star, PED = lagrange_relaxation_UC_ED()
    print("Optimal objective function value:", J_star)
    print("Optimal total power generation:", q_star)
#    print("Power output after economic dispatch:", PED)


if __name__ == "__main__":
    main()