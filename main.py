import math
import pandas as pd
import numpy as np
from scipy.optimize import minimize

original_data = pd.read_excel('UC.xlsx')
sheet = original_data.get_sheet_by_name('gendata')
pmin = [sheet.cell(row=i + 2, column=4).value for i in range(11)]
pmax = [sheet.cell(row=i + 2, column=3).value for i in range(11)]
a = [sheet.cell(row=i + 2, column=5).value for i in range(11)]
b = [sheet.cell(row=i + 2, column=6).value for i in range(11)]
c = [sheet.cell(row=i + 2, column=7).value for i in range(11)]
sheet2 = original_data.get_sheet_by_name('load')
load = [sheet2.cell(row=i + 2, column=2).value for i in range(168)]
def lagrange_relaxation_UC_ED(max_iter=1000, tol=0.001):
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
    lmbd = [load[i]*0.01 for i in range(168)]

    # Initialize flag for economic dispatch needed to False.
    PED_needed = [False for i in range(168)]

    # Initialize generator power outputs to zero, unit commitment variables to zero (off), and
    # power output after economic dispatch to zero.
    P = [[0 for j in range(11)] for i in range(168)]
    U = [[0 for j in range(11)] for i in range(168)]
    PED = [[0 for j in range(11)] for i in range(168)]

    J = 0
    q = 0
    for i in range(168):
        PUsum = sum(P[i][j]*U[i][j] for j in range(11))
        if PUsum < load[i]:
            economic_dispatch(F, lmbd[i], load[i], P[i])
        else:
            J += 10000
        for j in range(11):
            P[i][j] = max(min(max((lmbd[i][j]-b[j])/(2 * c[j]), 0), pmax[j]), pmin[j])
            F = a[j] + b[j] * P[i][j] + c[j] * P[i][j] ** 2
            Fprime = b[j] + 2 * c[j] * P[i][j]
            U[i][j] = 1 - min(max((F - lmbd[i][j]*P[i][j]), 0), 1)
            PUsum += P[i][j] * U[i][j]
        q += PUsum

    # Iterate until convergence or maximum number of iterations is reached.
    for k in range(max_iter):
        # Update Lagrange multipliers.
        for i in range(168):
            PUsum = sum(P[i][j]*U[i][j] for j in range(11))
            if PED[i]:
                lmbd[i] += 0.01*(load[i] - PUsum)
            else:
                lmbd[i] -= 0.01*(PUsum - load[i])

        # Calculate updated generator power outputs

def calculate_Jq(P, F, lmbd, load, PED):
    # Calculate q* and J*
    q_star = 0
    for i in range(len(load)):
        if PED[i]:
            q_star += max(load[i] - sum([P[j] for j in range(len(P)) if PED[j]]), 0)
    J_star = sum(F) + sum([lmbd[i] * q_star for i in range(len(lmbd))])
    return J_star, q_star

def economic_dispatch(F, lmbd, load, P):
    """
    Solves the economic dispatch problem. There are 11 generators per each economic dispatch problem.
    Inputs:
        F: the cost function for each generator.
        lmbd: the Lagrange multiplier.
        load: the load value.
    """


