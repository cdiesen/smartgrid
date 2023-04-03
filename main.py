import math
import pandas as pd
import numpy as np
"""
this program uses the lagrange relaxation method to solve the UC problem for a 168 hour period with 11 generators. the data is taken from the excel file UC.xlsx, which contains the data for the 11 generators, the load at each time interval, and the initial values for the lagrange multipliers the data is read from the excel file and stored in the appropriate variables
"""
original_data = pd.read_excel('UC.xlsx')
sheet = original_data.get_sheet_by_name('gendata')
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
sheet2 = original_data.get_sheet_by_name('load')
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
        # PED is the new power for each generator after economic dispatch. It is calculated using MILP

