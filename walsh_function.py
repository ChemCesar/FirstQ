#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 21 12:59:22 2024

@author: cesar
"""

import os
import numpy as np 
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from Walsh import WalshConstructor

n_qubits = 12
n_cpus = 1
n_operators = 100

def f(x):
    
    return np.cos(2 * np.pi * x)

start = timer()

#Additional setup, DO NOT TOUCH IF NOT MANDATORY
script_dir = os.path.dirname(__file__)
N = 2 ** n_qubits

#Generation of the results
walsh = WalshConstructor(n_qubits, n_operators, n_cpus, script_dir, f)
result = walsh.parallel_sparse_walsh_series()

x = np.array(range(N))/N
plt.plot(x, result)
plt.plot(x, f(x))
plt.savefig("Example.eps")

end = timer() - start 
result = np.array(result)/np.linalg.norm(result)
f = np.array(f(x))/np.linalg.norm(f(x))
fidelity = result.T @ f

print(f"Fidelity of the Walsh decomposition of the function f : {fidelity}")
print(f"Total Runtime for generating the Walsh decomposition of the function f : {end} s")