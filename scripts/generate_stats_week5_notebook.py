#!/usr/bin/env python3
"""
Generate Week 5 Statistics Notebook: Measures of Dispersion and Variability
Creates comprehensive interactive notebook covering range, variance, standard deviation, IQR.
"""

import json

def create_markdown_cell(content):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split('\n')
    }

def create_code_cell(code, outputs=None):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": code.split('\n')
    }

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.6"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

notebook["cells"].append(create_markdown_cell("""# Week 5: Measures of Dispersion and Variability

---
**Date**: 2025-11-22
**Course**: BSMA1002 - Statistics for Data Science I
**Week**: 5 of 12
**Topic**: Descriptive Statistics - Variability
---

## Learning Objectives

- Understand why measuring spread is as important as measuring center
- Calculate and interpret range, variance, and standard deviation
- Use IQR and detect outliers
- Compare datasets using coefficient of variation
- Visualize variability with box plots and error bars
- Apply dispersion measures to real-world data analysis

## Prerequisites

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```"""))

notebook["cells"].append(create_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Libraries imported successfully")
print(f"NumPy: {np.__version__}, Pandas: {pd.__version__}")"""))

notebook["cells"].append(create_markdown_cell("""## 1. Why Measure Variability?

**Central tendency tells us WHERE the data centers, but VARIABILITY tells us HOW SPREAD OUT it is.**

### Two Datasets with Same Mean

Consider these temperature readings from two cities:"""))

notebook["cells"].append(create_code_cell("""# Two cities with same mean but different variability
city_a = np.array([20, 21, 20, 19, 20, 21, 20, 19])  # Consistent
city_b = np.array([5, 10, 25, 30, 15, 22, 18, 35])   # Variable

mean_a, mean_b = np.mean(city_a), np.mean(city_b)

print("Temperature Data (°C):")
print("=" * 60)
print(f"City A: {city_a}")
print(f"  Mean: {mean_a:.1f}°C")
print(f"\\nCity B: {city_b}")
print(f"  Mean: {mean_b:.1f}°C")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(city_a, 'o-', linewidth=2, markersize=8, label='City A')
ax1.axhline(mean_a, color='r', linestyle='--', label=f'Mean = {mean_a:.1f}')
ax1.fill_between(range(len(city_a)), city_a.min(), city_a.max(), alpha=0.2)
ax1.set_title('City A: Low Variability', fontweight='bold')
ax1.set_xlabel('Day')
ax1.set_ylabel('Temperature (°C)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(city_b, 'o-', linewidth=2, markersize=8, label='City B', color='orange')
ax2.axhline(mean_b, color='r', linestyle='--', label=f'Mean = {mean_b:.1f}')
ax2.fill_between(range(len(city_b)), city_b.min(), city_b.max(), alpha=0.2)
ax2.set_title('City B: High Variability', fontweight='bold')
ax2.set_xlabel('Day')
ax2.set_ylabel('Temperature (°C)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("\\n🔍 Same mean, VERY different spread!")
print("    We need measures of VARIABILITY to distinguish them!")"""))

notebook["cells"].append(create_markdown_cell("""## 2. Range

**Definition**: Difference between maximum and minimum values

$$\\text{Range} = \\text{Max} - \\text{Min}$$

### Pros and Cons

✅ Simple to calculate and interpret  
✅ Shows full spread of data  
❌ Only uses two values (ignores all others)  
❌ Extremely sensitive to outliers"""))

notebook["cells"].append(create_code_cell("""# Calculate range for both cities
range_a = city_a.max() - city_a.min()
range_b = city_b.max() - city_b.min()

print("Range Calculation:")
print("=" * 60)
print(f"City A: {city_a.max()}°C - {city_a.min()}°C = {range_a}°C")
print(f"City B: {city_b.max()}°C - {city_b.min()}°C = {range_b}°C")

# Demonstrate outlier sensitivity
data_no_outlier = np.array([10, 12, 15, 13, 14, 11, 16, 12])
data_with_outlier = np.append(data_no_outlier, 100)

print(f"\\nOutlier Sensitivity:")
print(f"  Without outlier - Range: {data_no_outlier.ptp()}")
print(f"  With outlier    - Range: {data_with_outlier.ptp()}")
print(f"  Impact: Range increased {data_with_outlier.ptp() / data_no_outlier.ptp():.1f}x!")"""))

notebook["cells"].append(create_markdown_cell("""## 3. Variance

**Definition**: Average of squared deviations from the mean

**Population variance:**
$$\\sigma^2 = \\frac{\\sum_{i=1}^{n}(x_i - \\mu)^2}{n}$$

**Sample variance** (unbiased):
$$s^2 = \\frac{\\sum_{i=1}^{n}(x_i - \\bar{x})^2}{n-1}$$

### Why Square the Deviations?

1. Negative and positive deviations don't cancel out
2. Larger deviations get more weight
3. Mathematical properties useful for inference"""))

notebook["cells"].append(create_code_cell("""# Calculate variance step by step
data = np.array([4, 8, 6, 5, 3, 7])

print("Variance Calculation (Step by Step)")
print("=" * 60)
print(f"Data: {data}")

# Step 1: Mean
mean = data.mean()
print(f"\\n1. Calculate mean: {mean:.2f}")

# Step 2: Deviations
deviations = data - mean
print(f"\\n2. Deviations from mean:")
for i, (x, dev) in enumerate(zip(data, deviations)):
    print(f"   {x} - {mean:.2f} = {dev:+.2f}")

# Step 3: Squared deviations
squared_devs = deviations ** 2
print(f"\\n3. Squared deviations:")
for dev, sq_dev in zip(deviations, squared_devs):
    print(f"   ({dev:+.2f})² = {sq_dev:.2f}")

# Step 4: Average
print(f"\\n4. Sum of squared deviations: {squared_devs.sum():.2f}")
print(f"   n - 1 = {len(data) - 1}")
variance = squared_devs.sum() / (len(data) - 1)
print(f"   Variance (s²) = {squared_devs.sum():.2f} / {len(data) - 1} = {variance:.2f}")

# Verify with NumPy
np_variance = np.var(data, ddof=1)  # ddof=1 for sample variance
print(f"\\n✓ NumPy variance: {np_variance:.2f}")

# Visualize deviations
fig, ax = plt.subplots(figsize=(12, 6))
x_pos = np.arange(len(data))
bars = ax.bar(x_pos, data, alpha=0.6, edgecolor='black')
ax.axhline(mean, color='r', linestyle='--', linewidth=2, label=f'Mean = {mean:.2f}')

# Draw deviation lines
for i, (val, dev) in enumerate(zip(data, deviations)):
    ax.plot([i, i], [mean, val], 'g-', linewidth=2, alpha=0.7)
    ax.text(i, val + 0.3, f'{dev:+.2f}', ha='center', fontsize=9)

ax.set_xlabel('Data Point Index', fontsize=12)
ax.set_ylabel('Value', fontsize=12)
ax.set_title('Deviations from Mean (Variance Calculation)', fontsize=14, fontweight='bold')
ax.set_xticks(x_pos)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.show()"""))

notebook["cells"].append(create_markdown_cell("""## 4. Standard Deviation

**Definition**: Square root of variance

$$\\sigma = \\sqrt{\\sigma^2} \\quad \\text{or} \\quad s = \\sqrt{s^2}$$

### Why Use Standard Deviation?

✅ Same units as original data (variance is squared units)  
✅ More interpretable  
✅ Used in 68-95-99.7 rule for normal distributions  
✅ Standard measure for comparison"""))

notebook["cells"].append(create_code_cell("""# Compare variance and standard deviation
data_examples = {
    'Test Scores': np.array([75, 82, 88, 79, 91, 85, 77]),
    'Salaries ($1000s)': np.array([45, 52, 48, 55, 50, 47, 53]),
    'Heights (cm)': np.array([165, 172, 168, 170, 175, 169, 171])
}

print("Variance vs Standard Deviation")
print("=" * 70)

for name, data in data_examples.items():
    mean = data.mean()
    variance = data.var(ddof=1)
    std_dev = data.std(ddof=1)
    
    print(f"\\n{name}:")
    print(f"  Mean:     {mean:.2f}")
    print(f"  Variance: {variance:.2f} (squared units)")
    print(f"  Std Dev:  {std_dev:.2f} (same units as data)")
    print(f"  Range:    [{mean - std_dev:.2f}, {mean + std_dev:.2f}]")

# Visualize for test scores
scores = data_examples['Test Scores']
mean_scores = scores.mean()
std_scores = scores.std(ddof=1)

fig, ax = plt.subplots(figsize=(12, 6))
ax.scatter(range(len(scores)), scores, s=100, zorder=3, label='Scores')
ax.axhline(mean_scores, color='r', linestyle='-', linewidth=2, label=f'Mean = {mean_scores:.1f}')
ax.axhline(mean_scores + std_scores, color='g', linestyle='--', linewidth=2, 
           label=f'Mean ± 1 SD')
ax.axhline(mean_scores - std_scores, color='g', linestyle='--', linewidth=2)
ax.fill_between(range(len(scores)), mean_scores - std_scores, mean_scores + std_scores,
                alpha=0.2, color='green')

ax.text(len(scores)-0.5, mean_scores + std_scores + 1, f'+1σ = {mean_scores + std_scores:.1f}',
        fontsize=10)
ax.text(len(scores)-0.5, mean_scores - std_scores - 1, f'-1σ = {mean_scores - std_scores:.1f}',
        fontsize=10)

ax.set_xlabel('Student Index', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Standard Deviation: ±1σ Range', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)
plt.show()"""))

notebook["cells"].append(create_markdown_cell("""## 5. Quartiles and Interquartile Range (IQR)

**Quartiles** divide sorted data into four equal parts:
- Q1 (25th percentile): 25% of data below
- Q2 (50th percentile): Median
- Q3 (75th percentile): 75% of data below

**Interquartile Range (IQR)**:
$$\\text{IQR} = Q3 - Q1$$

### Advantages
✅ Resistant to outliers  
✅ Focuses on middle 50% of data  
✅ Used in box plots  
✅ Outlier detection method"""))

notebook["cells"].append(create_code_cell("""# Calculate quartiles and IQR
data = np.array([12, 15, 18, 20, 22, 25, 28, 30, 35, 40, 45, 50, 100])  # Note outlier

print("Quartiles and IQR Calculation")
print("=" * 60)
print(f"Data: {data}")
print(f"Sorted: {np.sort(data)}")

# Calculate quartiles
Q1 = np.percentile(data, 25)
Q2 = np.percentile(data, 50)  # median
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

print(f"\\nQuartiles:")
print(f"  Q1 (25th percentile): {Q1:.2f}")
print(f"  Q2 (50th percentile/Median): {Q2:.2f}")
print(f"  Q3 (75th percentile): {Q3:.2f}")
print(f"\\nIQR = Q3 - Q1 = {Q3:.2f} - {Q1:.2f} = {IQR:.2f}")

# Outlier detection using IQR method
lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR
outliers = data[(data < lower_fence) | (data > upper_fence)]

print(f"\\nOutlier Detection (1.5 × IQR rule):")
print(f"  Lower fence: Q1 - 1.5×IQR = {Q1:.2f} - {1.5*IQR:.2f} = {lower_fence:.2f}")
print(f"  Upper fence: Q3 + 1.5×IQR = {Q3:.2f} + {1.5*IQR:.2f} = {upper_fence:.2f}")
print(f"  Outliers: {outliers}")

# Box plot visualization
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Detailed box plot
bp = ax1.boxplot(data, vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
bp['boxes'][0].set_edgecolor('black')
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2)

# Add labels
ax1.text(1.15, Q1, f'Q1 = {Q1:.1f}', fontsize=10, va='center')
ax1.text(1.15, Q2, f'Q2 = {Q2:.1f}', fontsize=10, va='center')
ax1.text(1.15, Q3, f'Q3 = {Q3:.1f}', fontsize=10, va='center')
ax1.text(1.15, lower_fence, f'Lower fence', fontsize=9, va='center', style='italic')
ax1.text(1.15, upper_fence, f'Upper fence', fontsize=9, va='center', style='italic')

ax1.set_ylabel('Value', fontsize=12)
ax1.set_title('Box Plot with Outlier Detection', fontsize=13, fontweight='bold')
ax1.set_xticks([])
ax1.grid(True, alpha=0.3, axis='y')

# Show IQR range
ax2.hist(data, bins=15, edgecolor='black', alpha=0.7)
ax2.axvline(Q1, color='g', linestyle='--', linewidth=2, label='Q1')
ax2.axvline(Q3, color='r', linestyle='--', linewidth=2, label='Q3')
ax2.axvspan(Q1, Q3, alpha=0.2, color='yellow', label=f'IQR = {IQR:.1f}')
ax2.axvline(Q2, color='b', linestyle='-', linewidth=2, label='Median')

for outlier in outliers:
    ax2.axvline(outlier, color='red', linestyle=':', linewidth=1.5, alpha=0.7)
    ax2.text(outlier, ax2.get_ylim()[1]*0.9, 'Outlier', rotation=90, 
             fontsize=9, ha='right', color='red')

ax2.set_xlabel('Value', fontsize=12)
ax2.set_ylabel('Frequency', fontsize=12)
ax2.set_title('Histogram with Quartiles and IQR', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()"""))

notebook["cells"].append(create_markdown_cell("""## 6. Coefficient of Variation (CV)

**Definition**: Standardized measure of dispersion

$$CV = \\frac{s}{\\bar{x}} \\times 100\\%$$

### When to Use CV

✅ Comparing variability of datasets with different units  
✅ Comparing datasets with different scales  
✅ Assessing relative variability

**Example**: Compare variability of heights (cm) vs weights (kg)"""))

notebook["cells"].append(create_code_cell("""# Compare datasets using CV
datasets = {
    'Heights (cm)': np.array([165, 170, 175, 168, 172, 169, 171, 173]),
    'Weights (kg)': np.array([60, 65, 70, 62, 68, 64, 66, 69]),
    'Income ($1000s)': np.array([45, 50, 55, 48, 52, 49, 51, 53])
}

print("Coefficient of Variation Comparison")
print("=" * 70)

cv_results = []
for name, data in datasets.items():
    mean = data.mean()
    std = data.std(ddof=1)
    cv = (std / mean) * 100
    cv_results.append((name, mean, std, cv))
    
    print(f"\\n{name}:")
    print(f"  Mean: {mean:.2f}")
    print(f"  Std Dev: {std:.2f}")
    print(f"  CV: {cv:.2f}%")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Standard deviations (not comparable due to different scales)
names = [name.split()[0] for name in datasets.keys()]
stds = [result[2] for result in cv_results]
ax1.bar(names, stds, edgecolor='black', alpha=0.7)
ax1.set_ylabel('Standard Deviation', fontsize=12)
ax1.set_title('Standard Deviations\\n(NOT Comparable - Different Units)', 
             fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3, axis='y')

# CVs (comparable)
cvs = [result[3] for result in cv_results]
bars = ax2.bar(names, cvs, edgecolor='black', alpha=0.7, color='green')
ax2.set_ylabel('Coefficient of Variation (%)', fontsize=12)
ax2.set_title('Coefficients of Variation\\n(Comparable - Standardized)', 
             fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, cv in zip(bars, cvs):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.1,
            f'{cv:.1f}%', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

print(f"\\n📊 CV allows comparison across different scales!")
print(f"    Highest relative variability: {cv_results[np.argmax(cvs)][0]}")"""))

notebook["cells"].append(create_markdown_cell("""## 7. Real-World Application: Stock Portfolio Analysis

Analyze risk (variability) vs return for different investment options."""))

notebook["cells"].append(create_code_cell("""# Simulate monthly returns for different investments
np.random.seed(42)

# Different investment options
investments = {
    'Safe Bond': np.random.normal(2, 1, 36),      # Low return, low risk
    'Blue Chip Stock': np.random.normal(5, 3, 36),  # Medium return, medium risk
    'Tech Stock': np.random.normal(8, 8, 36),     # High return, high risk
    'Crypto': np.random.normal(10, 15, 36)        # Very high return, very high risk
}

print("Investment Risk-Return Analysis (36 months)")
print("=" * 70)

results = []
for name, returns in investments.items():
    mean_return = returns.mean()
    std_dev = returns.std(ddof=1)
    cv = (std_dev / abs(mean_return)) * 100 if mean_return != 0 else 0
    min_return = returns.min()
    max_return = returns.max()
    
    results.append({
        'Investment': name,
        'Mean Return (%)': mean_return,
        'Std Dev (%)': std_dev,
        'CV (%)': cv,
        'Min (%)': min_return,
        'Max (%)': max_return
    })
    
    print(f"\\n{name}:")
    print(f"  Average Monthly Return: {mean_return:.2f}%")
    print(f"  Risk (Std Dev): {std_dev:.2f}%")
    print(f"  Coefficient of Variation: {cv:.2f}%")
    print(f"  Range: [{min_return:.2f}%, {max_return:.2f}%]")

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Risk-Return Scatter
ax1 = fig.add_subplot(gs[0, :])
means = [r['Mean Return (%)'] for r in results]
stds = [r['Std Dev (%)'] for r in results]
names = [r['Investment'] for r in results]

ax1.scatter(stds, means, s=300, alpha=0.6, c=range(len(results)), cmap='viridis')
for i, name in enumerate(names):
    ax1.annotate(name, (stds[i], means[i]), fontsize=11, fontweight='bold',
                ha='center', va='center')

ax1.set_xlabel('Risk (Standard Deviation %)', fontsize=12)
ax1.set_ylabel('Return (Mean %)', fontsize=12)
ax1.set_title('Risk vs Return Profile', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 2-5. Individual time series
for idx, (name, returns) in enumerate(investments.items()):
    row = (idx // 2) + 1
    col = idx % 2
    ax = fig.add_subplot(gs[row, col])
    
    months = range(len(returns))
    ax.plot(months, returns, 'o-', linewidth=1.5, markersize=4, alpha=0.7)
    ax.axhline(returns.mean(), color='r', linestyle='--', linewidth=2,
               label=f'Mean = {returns.mean():.1f}%')
    ax.fill_between(months, 
                    returns.mean() - returns.std(),
                    returns.mean() + returns.std(),
                    alpha=0.2, color='green', label='±1 SD')
    
    ax.set_xlabel('Month', fontsize=10)
    ax.set_ylabel('Return (%)', fontsize=10)
    ax.set_title(name, fontsize=11, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.show()

# Summary table
df_results = pd.DataFrame(results)
print("\\n" + "=" * 70)
print("SUMMARY TABLE")
print("=" * 70)
print(df_results.to_string(index=False))

print("\\n💡 Key Insight: Higher returns come with higher risk (variability)!")"""))

notebook["cells"].append(create_markdown_cell("""## 8. Summary and Key Takeaways

### Measures of Dispersion

| Measure | Formula | Pros | Cons | Best For |
|---------|---------|------|------|----------|
| **Range** | Max - Min | Simple | Only 2 values, outlier-sensitive | Quick overview |
| **Variance** | $s^2 = \\frac{\\sum(x_i - \\bar{x})^2}{n-1}$ | Uses all data | Squared units | Calculations |
| **Std Dev** | $s = \\sqrt{s^2}$ | Same units as data | Outlier-sensitive | Most common |
| **IQR** | Q3 - Q1 | Outlier-resistant | Uses only middle 50% | Skewed data |
| **CV** | $\\frac{s}{\\bar{x}} \\times 100\\%$ | Standardized | Needs non-zero mean | Comparing datasets |

### Decision Guide

- **Quick overview** → Range
- **Symmetric data, no outliers** → Standard Deviation  
- **Outliers or skewed** → IQR
- **Compare different scales** → Coefficient of Variation
- **Risk/uncertainty** → Standard Deviation or Variance

### Key Insights

✓ Variability is as important as central tendency  
✓ Different measures for different situations  
✓ Low variability = Predictable, consistent  
✓ High variability = Unpredictable, diverse  

### Data Science Applications

- **Machine learning**: Feature scaling, variance thresholding
- **Finance**: Risk assessment (volatility = std dev)
- **Quality control**: Process consistency  
- **A/B testing**: Compare group variability
- **Outlier detection**: IQR method

---

**Next Week**: Correlation and Association between Variables"""))

# Save
output_path = "/Users/godsonkurishinkal/Projects/iitm-bs-learning/01-Foundation-Level/02-Statistics-I/notebooks/week-05-dispersion-variability.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"✓ Week 5 notebook created: {output_path}")
print(f"✓ Total cells: {len(notebook['cells'])}")
print(f"✓ Markdown: {sum(1 for c in notebook['cells'] if c['cell_type']=='markdown')}")
print(f"✓ Code: {sum(1 for c in notebook['cells'] if c['cell_type']=='code')}")
