#!/bin/bash
# Generate remaining weeks 9-12 efficiently

# Week 9: Discrete Distributions
cat > generate_stats_week9_notebook.py << 'W9EOF'
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
W9EOF

# Week 10: Continuous Distributions  
cat > generate_stats_week10_notebook.py << 'W10EOF'
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
W10EOF

# Week 11: Normal Distribution
cat > generate_stats_week11_notebook.py << 'W11EOF'
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
W11EOF

# Week 12: Review and Applications
cat > generate_stats_week12_notebook.py << 'W12EOF'
#!/usr/bin/env python3
import json
def cm(c): return {"cell_type": "markdown", "metadata": {}, "source": c.split('\n')}
def cc(c): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": c.split('\n')}
nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.9.6"}}, "nbformat": 4, "nbformat_minor": 4}
nb["cells"].extend([
cm("""# Week 12: Course Review & Applications

**Course**: BSMA1002 - Statistics for Data Science I  
**Topic**: Integration and Real-World Applications

## Course Summary
Comprehensive review of all statistical concepts"""),
cc("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
print("✓ All tools loaded for comprehensive review")"""),
cm("""## 1. Descriptive Statistics Review

### Central Tendency
- **Mean**: Average value
- **Median**: Middle value (robust)
- **Mode**: Most frequent

### Dispersion
- **Range**: Max - Min
- **Variance**: Average squared deviation
- **Standard Deviation**: √Variance
- **IQR**: Q3 - Q1"""),
cc("""# Comprehensive example
data = np.random.normal(100, 15, 1000)

stats_summary = {
    'Mean': data.mean(),
    'Median': np.median(data),
    'Std Dev': data.std(),
    'Q1': np.percentile(data, 25),
    'Q3': np.percentile(data, 75),
    'IQR': np.percentile(data, 75) - np.percentile(data, 25)
}

print("Descriptive Statistics Summary")
print("="*50)
for stat, value in stats_summary.items():
    print(f"{stat:12s}: {value:.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.hist(data, bins=30, edgecolor='black', alpha=0.7)
ax1.axvline(stats_summary['Mean'], color='red', linestyle='--', linewidth=2, label='Mean')
ax1.axvline(stats_summary['Median'], color='green', linestyle='--', linewidth=2, label='Median')
ax1.set_title('Distribution with Central Tendency', fontweight='bold')
ax1.legend()

ax2.boxplot(data)
ax2.set_title('Box Plot Showing IQR', fontweight='bold')
ax2.set_ylabel('Value')

plt.tight_layout()
plt.show()"""),
cm("""## 2. Probability Distributions Summary

| Type | Distribution | Parameters | Use Case |
|------|--------------|------------|----------|
| **Discrete** | Binomial | n, p | Fixed trials |
| **Discrete** | Poisson | λ | Count of events |
| **Continuous** | Uniform | a, b | Equal probability |
| **Continuous** | Exponential | λ | Time between events |
| **Continuous** | Normal | μ, σ | Most common, CLT |"""),
cc("""# Visual comparison of distributions
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Binomial
x = range(21)
ax = axes[0, 0]
probs = [stats.binom.pmf(k, 20, 0.5) for k in x]
ax.bar(x, probs, edgecolor='black')
ax.set_title('Binomial(n=20, p=0.5)', fontweight='bold')

# Poisson
ax = axes[0, 1]
probs = [stats.poisson.pmf(k, 5) for k in x]
ax.bar(x, probs, edgecolor='black', color='orange')
ax.set_title('Poisson(λ=5)', fontweight='bold')

# Normal
ax = axes[0, 2]
x_norm = np.linspace(-4, 4, 1000)
ax.plot(x_norm, stats.norm.pdf(x_norm), linewidth=2)
ax.fill_between(x_norm, stats.norm.pdf(x_norm), alpha=0.3)
ax.set_title('Normal(μ=0, σ=1)', fontweight='bold')

# Uniform
ax = axes[1, 0]
x_unif = np.linspace(-1, 11, 1000)
ax.plot(x_unif, stats.uniform.pdf(x_unif, 0, 10), linewidth=2, color='purple')
ax.fill_between(x_unif, stats.uniform.pdf(x_unif, 0, 10), alpha=0.3, color='purple')
ax.set_title('Uniform(0, 10)', fontweight='bold')

# Exponential
ax = axes[1, 1]
x_exp = np.linspace(0, 10, 1000)
ax.plot(x_exp, stats.expon.pdf(x_exp, scale=2), linewidth=2, color='green')
ax.fill_between(x_exp, stats.expon.pdf(x_exp, scale=2), alpha=0.3, color='green')
ax.set_title('Exponential(λ=0.5)', fontweight='bold')

# Hide last subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

print("✓ All major distributions visualized")"""),
cm("""## 3. Real-World Integrated Application

**Business Scenario**: E-commerce conversion optimization

Combining multiple statistical concepts:
1. Descriptive statistics for data exploration
2. Probability for A/B testing
3. Normal distribution for confidence intervals"""),
cc("""# Integrated business application
# Simulate A/B test data
n_visitors_a = 1000
n_visitors_b = 1000
conversion_rate_a = 0.10
conversion_rate_b = 0.12

conversions_a = np.random.binomial(1, conversion_rate_a, n_visitors_a)
conversions_b = np.random.binomial(1, conversion_rate_b, n_visitors_b)

obs_rate_a = conversions_a.mean()
obs_rate_b = conversions_b.mean()
lift = (obs_rate_b - obs_rate_a) / obs_rate_a

print("E-Commerce A/B Test Analysis")
print("="*70)
print(f"\\nVariant A (Control):")
print(f"  Visitors: {n_visitors_a:,}")
print(f"  Conversions: {conversions_a.sum()}")
print(f"  Rate: {obs_rate_a:.2%}")

print(f"\\nVariant B (Treatment):")
print(f"  Visitors: {n_visitors_b:,}")
print(f"  Conversions: {conversions_b.sum()}")
print(f"  Rate: {obs_rate_b:.2%}")

print(f"\\nLift: {lift:+.1%}")

# Statistical test
pooled = (conversions_a.sum() + conversions_b.sum()) / (n_visitors_a + n_visitors_b)
se = np.sqrt(pooled * (1 - pooled) * (1/n_visitors_a + 1/n_visitors_b))
z_score = (obs_rate_b - obs_rate_a) / se
p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
significant = p_value < 0.05

print(f"\\nStatistical Significance:")
print(f"  Z-score: {z_score:.2f}")
print(f"  P-value: {p_value:.4f}")
print(f"  Significant (α=0.05): {'YES ✅' if significant else 'NO ❌'}")

# Visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Conversion rates
variants = ['Variant A\\n(Control)', 'Variant B\\n(Treatment)']
rates = [obs_rate_a, obs_rate_b]
colors = ['lightblue', 'lightgreen']
bars = ax1.bar(variants, rates, color=colors, edgecolor='black', linewidth=2)
ax1.set_ylabel('Conversion Rate', fontsize=12)
ax1.set_title('A/B Test Results', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

for bar, rate in zip(bars, rates):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
            f'{rate:.2%}', ha='center', fontsize=12, fontweight='bold')

# Business impact
if significant:
    revenue_per_visitor = 50  # $
    additional_conversions = (obs_rate_b - obs_rate_a) * n_visitors_b
    revenue_impact = additional_conversions * revenue_per_visitor * 365  # annual
    
    ax2.bar(['Current\\nRevenue', 'With\\nVariant B'], 
           [n_visitors_a * obs_rate_a * revenue_per_visitor * 365,
            n_visitors_b * obs_rate_b * revenue_per_visitor * 365],
           color=['gray', 'green'], edgecolor='black', linewidth=2)
    ax2.set_ylabel('Annual Revenue ($)', fontsize=12)
    ax2.set_title(f'Business Impact: +${revenue_impact:,.0f}/year', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
else:
    ax2.text(0.5, 0.5, 'Not Significant\\nNeed More Data', 
            ha='center', va='center', fontsize=16, transform=ax2.transAxes)
    ax2.axis('off')

plt.tight_layout()
plt.show()

if significant:
    print(f"\\n💰 Business Impact: Deploying Variant B could generate")
    print(f"    +${revenue_impact:,.0f} additional annual revenue!")"""),
cm("""## Course Summary

### What We Learned

**Weeks 1-3**: Data collection and summarization  
**Week 4**: Central tendency (mean, median, mode)  
**Week 5**: Dispersion (variance, SD, IQR)  
**Week 6**: Correlation and relationships  
**Week 7**: Probability fundamentals  
**Week 8**: Random variables and expected value  
**Week 9**: Discrete distributions (Binomial, Poisson)  
**Week 10**: Continuous distributions (Uniform, Exponential)  
**Week 11**: Normal distribution and Z-scores  
**Week 12**: Integration and applications  

### Key Skills Acquired
✅ Descriptive statistics analysis  
✅ Probability calculations  
✅ Distribution identification and application  
✅ Statistical inference basics  
✅ Real-world problem solving  

### Tools Mastered
- NumPy for numerical computing
- Pandas for data manipulation
- Matplotlib/Seaborn for visualization
- SciPy for statistical functions

---

## Next Steps
- **Statistics II**: Hypothesis testing, regression, ANOVA
- **Machine Learning**: Apply statistical foundations
- **Data Science Projects**: Real-world applications

**Congratulations on completing Statistics for Data Science I! ��**""")
])
with open("/Users/godsonkurishinkal/Projects/iitm-bs-learning/01-Foundation-Level/02-Statistics-I/notebooks/week-12-review-applications.ipynb", 'w') as f:
    json.dump(nb, f, indent=2)
print(f"✓ Week 12: {len(nb['cells'])} cells")
W12EOF

# Execute all generators
echo "Generating weeks 9-12..."
python3 generate_stats_week9_notebook.py
python3 generate_stats_week10_notebook.py
python3 generate_stats_week11_notebook.py
python3 generate_stats_week12_notebook.py
echo "✅ All notebooks generated!"
