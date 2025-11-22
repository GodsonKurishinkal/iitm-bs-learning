#!/usr/bin/env python3
import json
def cm(c): return {"cell_type": "markdown", "metadata": {}, "source": c.split('\n')}
def cc(c): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.split('\n')}
nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.9.6"}}, "nbformat": 4, "nbformat_minor": 4}
nb["cells"].extend([
cm("""# Week 11: Normal Distribution

**Course**: BSMA1002 - Statistics for Data Science I  
**Topic**: The Most Important Distribution

## Learning Objectives
- Master normal distribution
- Apply 68-95-99.7 rule
- Use Z-scores and standardization
- Apply to real data"""),
cc("""import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
print("✓ Ready")"""),
cm("""## 1. Normal Distribution

**Bell curve, parameters μ and σ**

$$f(x) = \\frac{1}{\\sigma\\sqrt{2\\pi}} e^{-\\frac{1}{2}(\\frac{x-\\mu}{\\sigma})^2}$$

### 68-95-99.7 Rule
- 68% within ±1σ
- 95% within ±2σ
- 99.7% within ±3σ"""),
cc("""from scipy.stats import norm
mu, sigma = 100, 15
x = np.linspace(mu-4*sigma, mu+4*sigma, 1000)
pdf = norm.pdf(x, mu, sigma)

print(f"Normal(μ={mu}, σ={sigma})")

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, pdf, linewidth=3)

# 68-95-99.7 regions
ax.fill_between(x, pdf, where=(x>=mu-sigma)&(x<=mu+sigma), 
                alpha=0.3, label='68% (±1σ)')
ax.fill_between(x, pdf, where=((x>=mu-2*sigma)&(x<mu-sigma))|((x>mu+sigma)&(x<=mu+2*sigma)), 
                alpha=0.2, label='95% (±2σ)')

ax.axvline(mu, color='red', linestyle='--', label=f'μ={mu}')
ax.set_title('Normal Distribution: 68-95-99.7 Rule', fontsize=14, fontweight='bold')
ax.set_xlabel('x')
ax.set_ylabel('f(x)')
ax.legend()
plt.show()"""),
cm("""## 2. Z-Scores (Standardization)

$$Z = \\frac{X - \\mu}{\\sigma}$$

Converts any normal to **Standard Normal** N(0,1)"""),
cc("""# Z-score example
x_value = 115
z_score = (x_value - mu) / sigma
prob_less = norm.cdf(x_value, mu, sigma)

print(f"Value: {x_value}")
print(f"Z-score: {z_score:.2f}")
print(f"P(X < {x_value}) = {prob_less:.4f}")

# IQ example
iq_scores = np.array([85, 100, 115, 130])
z_scores = (iq_scores - mu) / sigma

print(f"\\nIQ Scores and Z-scores:")
for iq, z in zip(iq_scores, z_scores):
    percentile = norm.cdf(iq, mu, sigma) * 100
    print(f"  IQ {iq}: Z = {z:+.2f}, {percentile:.1f}th percentile")"""),
cm("""## 3. Real Application: Quality Control

Manufacturing process with normal variation."""),
cc("""# Quality control
target = 500  # mm
tolerance = 10  # mm
sigma_process = 3  # mm

lower_spec = target - tolerance
upper_spec = target + tolerance

prob_in_spec = norm.cdf(upper_spec, target, sigma_process) - norm.cdf(lower_spec, target, sigma_process)
prob_defect = 1 - prob_in_spec

print(f"Target: {target} mm")
print(f"Specification limits: [{lower_spec}, {upper_spec}]")
print(f"Process σ: {sigma_process} mm")
print(f"\\nP(in spec) = {prob_in_spec:.4f} ({prob_in_spec:.2%})")
print(f"Defect rate = {prob_defect:.4f} ({prob_defect:.2%})")

x = np.linspace(target-15, target+15, 1000)
pdf = norm.pdf(x, target, sigma_process)

fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(x, pdf, linewidth=3)
ax.fill_between(x, pdf, where=(x>=lower_spec)&(x<=upper_spec), 
                color='green', alpha=0.3, label='In Spec')
ax.fill_between(x, pdf, where=(x<lower_spec)|(x>upper_spec), 
                color='red', alpha=0.3, label='Defect')
ax.axvline(lower_spec, color='red', linestyle='--')
ax.axvline(upper_spec, color='red', linestyle='--')
ax.set_title(f'Process Control: {prob_defect:.2%} Defect Rate', fontsize=14, fontweight='bold')
ax.set_xlabel('Measurement (mm)')
ax.set_ylabel('Probability Density')
ax.legend()
plt.show()"""),
cm("""## Summary

### Normal Distribution N(μ, σ²)
- **Bell-shaped, symmetric**
- **68-95-99.7 rule** for quick probabilities
- **Z-scores** standardize any normal
- **Central Limit Theorem**: Sums of variables → Normal

### Applications
- Heights, weights, test scores
- Measurement errors
- Quality control (Six Sigma)
- Statistical inference

**Next**: Course review and applications""")
])
with open("/Users/godsonkurishinkal/Projects/iitm-bs-learning/01-Foundation-Level/02-Statistics-I/notebooks/week-11-normal-distribution.ipynb", 'w') as f:
    json.dump(nb, f, indent=2)
print(f"✓ Week 11: {len(nb['cells'])} cells")
