import math
import pandas as pd
import numpy as np
from scipy.optimize import minimize
"""
this program uses the lagrange relaxation method to solve the UC problem for a 168 hour period with 11 generators.
the data is taken from the excel file UC.xlsx, which contains the data for the 11 generators, the load at each time
interval, and the initial values for the lagrange multipliers the data is read from the excel file and stored in the
appropriate variables
"""

original_data = pd.read_excel('UC.xlsx')
sheet = original_data.get_sheet_by_name('gendata')
sheet2 = original_data.get_sheet_by_name('load')

# the lagrange multipliers are initialized to 0
lmbd = [0 for i in range(168)]
pmin = []
pmax = []
a = []
b = []
c = []
F = []
P = []
U = []
PED = []
Fprime = []
J = 0
q = 0

load = []
PUsum = 0

for i in range(168):
    load[i] = sheet2.cell(row=i + 2, column=2).value
    for j in range(11):
        pmin[j] = sheet.cell(row=j + 2, column=4).value
        pmax[j] = sheet.cell(row=j + 2, column=3).value
        a[j] = sheet.cell(row=j + 2, column=5).value
        b[j] = sheet.cell(row=j + 2, column=6).value
        c[j] = sheet.cell(row=j + 2, column=7).value
        P[j] = max(min(max((lmbd[j]-b[j])/(2 * c[j]), 0), pmax[j]), pmin[j])
        F[j] = a[j] + b[j] * P[j] + c[j] * P[j] ^ 2
        Fprime[j] = b[j] + 2 * c[j] * P[j]
        U[j] = 1 - min(max((F[j] - lmbd[j]*P[j]), 0), 1)
        PUsum += P[j] * U[j]
        if PUsum > 0:
            lmbd[j] += 0.01*PUsum
    PED[j] = True if load[j] < PUsum else False
    if PED[j]:
        # Economic dispatch is required if Pload < P*U for all generators
        # and the values of the generators' power, using economic dispatch.
        # use mixed integer linear programming to optimize PED
        # PED is the new power for each generator after economic dispatch

def objective(x):
    """
    Objective function to minimize the total cost of power generation
    """
    return sum([a[i] + b[i] * x[i] + c[i] * x[i] ** 2 for i in range(len(x))])


def constraint(x):
    """
    Constraint function that ensures the total power generated meets the load demand
    """
    return sum(x) - sum([load[i] for i in range(len(load))])


cons = {'type': 'eq', 'fun': constraint}
bnds = [(pmin[i], pmax[i]) for i in range(len(pmin))]

while True:
    # Solve the Economic Dispatch problem using a convex optimization solver
    result = minimize(objective, P, bounds=bnds, constraints=cons)
    PED = result.x

    # Check if the Economic Dispatch solution satisfies the load demand
    if sum(PED) >= sum(load):
        P = PED
        break

    # If not, update the Lagrange multipliers and try again
    for j in range(11):
        F[j] = a[j] + b[j] * PED[j] + c[j] * PED[j] ** 2
        Fprime[j] = b[j] + 2 * c[j] * PED[j]
        U[j] = 1 - min(max((F[j] - lmbd[j] * PED[j]), 0), 1)
        lmbd[j] += 0.01 * (sum(PED) - sum(load)) * U[j]

P = [PED[i] if PED[i] else P[i] for i in range(len(P))]
