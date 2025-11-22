#!/usr/bin/env python3
import json
def cm(c): return {"cell_type": "markdown", "metadata": {}, "source": c.split('\n')}
def cc(c): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.split('\n')}
nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.9.6"}}, "nbformat": 4, "nbformat_minor": 4}
nb["cells"].extend([
cm("""# Week 10: Continuous Distributions

**Course**: BSMA1002 - Statistics for Data Science I  
**Topic**: Uniform, Exponential

## Learning Objectives
- Understand PDF vs PMF
- Apply uniform distribution
- Use exponential for timing
- Calculate probabilities with CDF"""),
cc("""import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
print("✓ Ready")"""),
cm("""## 1. Uniform Distribution

**Equally likely on [a, b]**

$$f(x) = \\frac{1}{b-a}, \\quad a \\leq x \\leq b$$

- E[X] = (a+b)/2
- Var(X) = (b-a)²/12"""),
cc("""from scipy.stats import uniform
a, b = 0, 10
x = np.linspace(a-2, b+2, 1000)
pdf = uniform.pdf(x, a, b-a)

print(f"Uniform({a}, {b})")
print(f"Mean = {(a+b)/2}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, pdf, linewidth=3)
ax.fill_between(x, pdf, alpha=0.3)
ax.set_title('Uniform Distribution PDF', fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.show()"""),
cm("""## 2. Exponential Distribution

**Time between events, rate λ**

$$f(x) = \\lambda e^{-\\lambda x}, \\quad x \\geq 0$$

- E[X] = 1/λ
- Var(X) = 1/λ²
- Memoryless property"""),
cc("""from scipy.stats import expon
lam = 0.5
mean = 1/lam
x = np.linspace(0, 10, 1000)
pdf = expon.pdf(x, scale=mean)

print(f"Exponential(λ={lam})")
print(f"Mean = {mean}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x, pdf, linewidth=3, color='green')
ax.fill_between(x, pdf, alpha=0.3)
ax.set_title(f'Exponential PDF: λ={lam}', fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
plt.show()"""),
cm("""## Summary

| Distribution | PDF | Mean | Use Case |
|--------------|-----|------|----------|
| **Uniform** | 1/(b-a) | (a+b)/2 | Equal likelihood |
| **Exponential** | λe^(-λx) | 1/λ | Time between events |

**Next**: Normal distribution""")
])
with open("/Users/godsonkurishinkal/Projects/iitm-bs-learning/01-Foundation-Level/02-Statistics-I/notebooks/week-10-continuous-distributions.ipynb", 'w') as f:
    json.dump(nb, f, indent=2)
print(f"✓ Week 10: {len(nb['cells'])} cells")
