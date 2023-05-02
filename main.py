# import math
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


def lagrange_relaxation_UC_ED(max_iter=100, tol=0.001, J=0, q=0):
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
    lmbd = [load[i]/10 for i in range(168)]

    # # Initialize flag for economic dispatch needed to False.
    # PED_needed = [False for i in range(168)]

    # Initialize generator power outputs to zero, unit commitment variables to zero (off), and
    # power output after economic dispatch to zero.
    P = [[0 for j in range(11)] for i in range(168)]
    U = [[0 for j in range(11)] for i in range(168)]
    PED = [[0 for j in range(11)] for i in range(168)]
    PUsum = [0 for i in range(168)]
    F = [[0 for j in range(11)] for i in range(168)]
    Fprime = [[0 for j in range(11)] for i in range(168)]
    # Add a generator penalty array, where each element represents
    # the number of time periods the generator is up.
    gen_periods_up = [0] * 11
    for i in range(168):
        for j in range(11):
            P[i][j] = max(min(max((lmbd[i] - b[j]) / (2 * c[j]), 0), pmax[j]), pmin[j])
            U[i][j] = 1 - min(max(cost_function(P[i][j], a[j], b[j], c[j]) - lmbd[i] * P[i][j], 0), 1)
            P[i][j] = U[i][j] * P[i][j]
            F[i][j] = cost_function(P[i][j], a[j], b[j], c[j])
            Fprime[i][j] = cost_function_derivative(P[i][j], b[j], c[j])
            PUsum[i] += P[i][j]
        if load[i] < PUsum[i] or (P[i][j] > 0 for j in range(11)):
            PED[i], _ = economic_dispatch(b, c, U[i], pmin, pmax, load[i])
#            print(PED[i])
            J = J + sum(cost_function(PED[i][j], a[j], b[j], c[j]) for j in range(11))
            # Update the generator penalty array
            for j in range(11):
                if PED[i][j] > 0:
                    gen_periods_up[j] += 1
        else:
            J = J + 100000  # Add penalty for not meeting load demand, equal to 10000.
        q = q + sum(cost_function(P[i][j], a[j], b[j], c[j]) for j in range(11)) + lmbd[i] * (load[i] - PUsum[i])

    # Add a soft constraint for generator up-time by penalizing the objective function
    opt = (J - q) / q
    print(opt, J, q)
    for i in range(168):
        if load[i] > PUsum[i]:
            lmbd[i] = lmbd[i] + 0.1 * (load[i] - PUsum[i])
        else:
            lmbd[i] = lmbd[i] - 0.02 * (load[i] - PUsum[i])
    for _ in range(max_iter):
        opt_new, J_new, q_new, PED_new, lmbd_new = lagrange_relaxation_UC_ED_iteration(lmbd, tol, max_iter)
        for i in range(168):
            print(PED_new[i])
        print(opt_new, J_new, q_new)
        if abs(opt_new) < tol:
            print('Converged after', _, 'iterations.')
            print('Objective function value:', J_new)
            print('Total power generation:', q_new)
            print('Power output after economic dispatch:', PED_new)
            break


def lagrange_relaxation_UC_ED_iteration(lmbd, tol=0.001, max_iter=1000, J_iter=0, q_iter=0):
    # Add a generator penalty array, where each element represents
    # the number of time periods the generator is up.
    gen_periods_up = [0] * 11
    P = [[0 for j in range(11)] for i in range(168)]
    U = [[0 for j in range(11)] for i in range(168)]
    PED = [[0 for j in range(11)] for i in range(168)]
    PUsum = [0 for i in range(168)]
    F = [[0 for j in range(11)] for i in range(168)]
    Fprime = [[0 for j in range(11)] for i in range(168)]

    for i in range(168):
        for j in range(11):
            P[i][j] = max(min(max((lmbd[i] - b[j]) / (2 * c[j]), 0), pmax[j]), pmin[j])
            U[i][j] = 1 - min(max(cost_function(P[i][j], a[j], b[j], c[j]) - lmbd[i] * P[i][j], 0), 1)
            P[i][j] = U[i][j] * P[i][j]
            F[i][j] = cost_function(P[i][j], a[j], b[j], c[j])
            Fprime[i][j] = cost_function_derivative(P[i][j], b[j], c[j])
            PUsum[i] += P[i][j]
        if load[i] < PUsum[i] or (P[i][j] > 0 for j in range(11)):
            PED[i], _ = economic_dispatch(b, c, U[i], pmin, pmax, load[i])
#            print(PED[i])
            J_iter = J_iter + sum(cost_function(PED[i][j], a[j], b[j], c[j]) for j in range(11))
            # Update the generator penalty array
            for j in range(11):
                if PED[i][j] > 0:
                    gen_periods_up[j] += 1
        else:
            J_iter += 10000  # Add penalty for not meeting load demand, equal to 10000.
        q_iter = q_iter + sum(cost_function(P[i][j], a[j], b[j], c[j]) for j in range(11))
    q_iter = q_iter + sum((lmbd[i]) * (load[i] - PUsum[i]) for i in range(168))
    # Update the Lagrange multipliers (lmbd) inside this function
    # Add a soft constraint for generator up-time by penalizing the objective function
#    penalty_weight = 100000000  # Adjust the penalty weight to balance the constraint importance
#    J_iter += penalty_weight * sum(max(1 - gen_periods_up[j], 0) for j in range(11))
    opt_iter = (J_iter - q_iter) / q_iter
    for i in range(168):
        if load[i] > PUsum[i]:
            lmbd[i] += 0.1 * (load[i] - PUsum[i])
        else:
            lmbd[i] -= 0.02 * (load[i] - PUsum[i])
    return opt_iter, J_iter, q_iter, PED, lmbd

def economic_dispatch(b, c, U, Pmin, Pmax, load, tol=0.001):
    n_gens = 11
    lmbd_min = min([b[i] + 2 * c[i] * Pmin[i] for i in range(n_gens) if U[i]])  # Fixed the indexing issue
    lmbd_max = max([b[i] + 2 * c[i] * Pmax[i] for i in range(n_gens) if U[i]])  # Fixed the indexing issue
    lmbda = (lmbd_min + lmbd_max) / 2
    delta_lambda = (lmbd_max - lmbd_min) / 2

    while delta_lambda > tol:
        P = [max(min(((lmbda - b[i]) / (2 * c[i])), Pmax[i]), Pmin[i]) if U[i] else 0 for i in range(n_gens)]
        PUsum = sum(P)
        if PUsum < load:
            lmbda = lmbda + delta_lambda
        else:
            lmbda = lmbda - delta_lambda
        delta_lambda /= 2

    PED = [max(min(((lmbda - b[i]) / (2 * c[i])), Pmax[i]), Pmin[i]) if U[i] else 0 for i in range(n_gens)]
    return PED, lmbda

def cost_function(p, a, b, c):
    """
    Computes the total cost of power generation for a given set of power outputs.
    Inputs:
        P: the power output for each generator.
        a: the constant cost coefficient for each generator.
        b: the linear cost coefficient for each generator.
        c: the quadratic cost coefficient for each generator.
    Outputs:
        F: the total cost of power generation.
    """
    f = a + p*b + (p**2)*c
    return f


def cost_function_derivative(p, b, c):
    """
    Computes the derivative of the total cost of power generation with respect to the power output for each generator.
    Inputs:
        P: the power output for each generator.
        b: the linear cost coefficient for each generator.
        c: the quadratic cost coefficient for each generator.
    Outputs:
        Fprime: the derivative of the total cost of power generation.
    """
    fprime = b + 2*c*p
    return fprime


optimization = lagrange_relaxation_UC_ED()
