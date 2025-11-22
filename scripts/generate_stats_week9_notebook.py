#!/usr/bin/env python3
import json
def cm(c): return {"cell_type": "markdown", "metadata": {}, "source": c.split('\n')}
def cc(c): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.split('\n')}
nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.9.6"}}, "nbformat": 4, "nbformat_minor": 4}
nb["cells"].extend([
cm("""# Week 9: Discrete Distributions

**Course**: BSMA1002 - Statistics for Data Science I  
**Topic**: Binomial and Poisson Distributions

## Learning Objectives
- Master binomial distribution
- Apply Poisson distribution  
- Use SciPy for calculations
- Solve real-world problems"""),
cc("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
print("✓ Ready")"""),
cm("""## 1. Binomial Distribution

**n trials, probability p**

$$P(X = k) = \\binom{n}{k} p^k (1-p)^{n-k}$$

- E[X] = np
- Var(X) = np(1-p)"""),
cc("""from scipy.stats import binom
n, p = 10, 0.3
x = range(n+1)
probs = [binom.pmf(k, n, p) for k in x]

print(f"Binomial(n={n}, p={p})")
print(f"E[X] = {n*p:.1f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x, probs, edgecolor='black')
ax.set_title(f'Binomial PMF: n={n}, p={p}', fontweight='bold')
ax.set_xlabel('k')
ax.set_ylabel('P(X=k)')
plt.show()"""),
cm("""## 2. Poisson Distribution

**Count of events, rate λ**

$$P(X = k) = \\frac{\\lambda^k e^{-\\lambda}}{k!}$$

- E[X] = λ
- Var(X) = λ"""),
cc("""from scipy.stats import poisson
lam = 4
x = range(15)
probs = [poisson.pmf(k, lam) for k in x]

print(f"Poisson(λ={lam})")
print(f"E[X] = Var(X) = {lam}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x, probs, edgecolor='black', color='orange')
ax.set_title(f'Poisson PMF: λ={lam}', fontweight='bold')
ax.set_xlabel('k')
ax.set_ylabel('P(X=k)')
plt.show()"""),
cm("""## Summary

| Distribution | Parameters | Mean | Variance |
|--------------|------------|------|----------|
| **Binomial** | n, p | np | np(1-p) |
| **Poisson** | λ | λ | λ |

**Next**: Continuous distributions""")
])
with open("/Users/godsonkurishinkal/Projects/iitm-bs-learning/01-Foundation-Level/02-Statistics-I/notebooks/week-09-discrete-distributions.ipynb", 'w') as f:
    json.dump(nb, f, indent=2)
print(f"✓ Week 9: {len(nb['cells'])} cells")
