#!/usr/bin/env python3
"""
Generate Week 4 Statistics Notebook: Measures of Central Tendency
Creates comprehensive interactive notebook with visualizations and applications.
"""

import json

def create_markdown_cell(content):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split('\n')
    }

def create_code_cell(code, outputs=None):
    """Create a code cell with optional outputs."""
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": code.split('\n')
    }
    return cell

# Build notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
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

# Cell 1: Title and metadata
notebook["cells"].append(create_markdown_cell("""# Week 4: Measures of Central Tendency - Practice Notebook

---
**Date**: 2025-11-22
**Course**: BSMA1002 - Statistics for Data Science I
**Level**: Foundation
**Week**: 4 of 12
**Topic Area**: Descriptive Statistics - Central Tendency
---

## Learning Objectives

This notebook provides hands-on practice with:
- Understanding mean, median, and mode as measures of center
- Calculating central tendency for different data types
- Choosing the appropriate measure based on data characteristics
- Identifying how outliers affect different measures
- Comparing distributions using central tendency
- Applying these concepts to real-world datasets
- Understanding weighted means and their applications

## Prerequisites

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```"""))

# Cell 2: Setup and imports
notebook["cells"].append(create_code_cell("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Libraries imported successfully")
print(f"NumPy version: {np.__version__}")
print(f"Pandas version: {pd.__version__}")"""))

# Cell 3: Introduction
notebook["cells"].append(create_markdown_cell("""## 1. Introduction to Central Tendency

**Central tendency** refers to the measure that represents the center or typical value of a dataset.

### Why It Matters

- **Summarize data**: Reduce thousands of values to a single representative number
- **Compare groups**: Compare different datasets using their centers
- **Detect patterns**: Identify where most data concentrates
- **Make decisions**: Use typical values for planning and prediction

### The Three Main Measures

| Measure | Definition | Best For |
|---------|------------|----------|
| **Mean** | Arithmetic average | Symmetric distributions, no outliers |
| **Median** | Middle value when sorted | Skewed data, presence of outliers |
| **Mode** | Most frequent value | Categorical data, finding peaks |

**Key Insight**: Different measures tell different stories about your data!"""))

# Cell 4: The Mean
notebook["cells"].append(create_markdown_cell("""## 2. The Mean (Arithmetic Average)

### Definition

The **mean** is the sum of all values divided by the count:

$$\\bar{x} = \\frac{\\sum_{i=1}^{n} x_i}{n} = \\frac{x_1 + x_2 + \\cdots + x_n}{n}$$

### Properties

- **Uses all data**: Every value contributes to the mean
- **Sensitive to outliers**: Extreme values pull the mean toward them
- **Balance point**: Sum of deviations from mean equals zero
- **Unique**: Only one mean per dataset

Let's calculate and visualize the mean:"""))

# Cell 5: Calculate mean
notebook["cells"].append(create_code_cell("""# Sample data: Test scores
scores = np.array([72, 85, 90, 78, 88, 92, 75, 82, 88, 95])

# Calculate mean
mean_score = np.mean(scores)
print(f"Test Scores: {scores}")
print(f"\\nMean (Average) Score: {mean_score:.2f}")

# Verify with manual calculation
manual_mean = scores.sum() / len(scores)
print(f"Manual calculation: {scores.sum()} / {len(scores)} = {manual_mean:.2f}")

# Using pandas
df_scores = pd.DataFrame({'scores': scores})
pandas_mean = df_scores['scores'].mean()
print(f"Pandas mean: {pandas_mean:.2f}")

# Visualize
fig, ax = plt.subplots(figsize=(12, 6))

# Plot individual scores
ax.scatter(range(len(scores)), scores, s=100, alpha=0.6, 
           label='Individual Scores', zorder=3)
ax.plot(range(len(scores)), scores, 'o-', alpha=0.3)

# Plot mean line
ax.axhline(y=mean_score, color='r', linestyle='--', linewidth=2,
           label=f'Mean = {mean_score:.2f}')

# Add text
ax.text(len(scores)-1, mean_score + 2, f'Mean: {mean_score:.2f}',
        fontsize=12, color='red', fontweight='bold')

ax.set_xlabel('Student Index', fontsize=12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Test Scores with Mean', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.show()

# Show deviations from mean
deviations = scores - mean_score
print(f"\\nDeviations from mean: {deviations}")
print(f"Sum of deviations: {deviations.sum():.10f} (should be ~0)")"""))

# Cell 6: Mean sensitivity to outliers
notebook["cells"].append(create_markdown_cell("""### 2.1 Mean's Sensitivity to Outliers

The mean is heavily influenced by extreme values. Let's see this in action:"""))

# Cell 7: Outlier demonstration
notebook["cells"].append(create_code_cell("""# Salaries in a small company
salaries_normal = np.array([45000, 48000, 50000, 52000, 47000, 49000, 51000, 46000, 48500, 50500])

# Add CEO salary (outlier)
salaries_with_ceo = np.append(salaries_normal, 500000)

# Calculate means
mean_normal = np.mean(salaries_normal)
mean_with_ceo = np.mean(salaries_with_ceo)

print("Company Salaries Analysis:")
print("=" * 60)
print(f"\\nWithout CEO:")
print(f"  Salaries: {salaries_normal}")
print(f"  Mean: ${mean_normal:,.2f}")

print(f"\\nWith CEO (outlier):")
print(f"  Salaries: {salaries_with_ceo}")
print(f"  Mean: ${mean_with_ceo:,.2f}")

print(f"\\nImpact: Mean increased by ${mean_with_ceo - mean_normal:,.2f}")
print(f"Percentage increase: {((mean_with_ceo - mean_normal) / mean_normal) * 100:.1f}%")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Without outlier
ax1.scatter(range(len(salaries_normal)), salaries_normal/1000, s=100, alpha=0.7)
ax1.axhline(y=mean_normal/1000, color='r', linestyle='--', linewidth=2,
            label=f'Mean = ${mean_normal/1000:.1f}K')
ax1.set_title('Salaries Without Outlier', fontsize=13, fontweight='bold')
ax1.set_xlabel('Employee Index')
ax1.set_ylabel('Salary ($1000s)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 550)

# With outlier
colors = ['blue'] * len(salaries_normal) + ['red']
ax2.scatter(range(len(salaries_with_ceo)), salaries_with_ceo/1000, 
            s=100, alpha=0.7, c=colors)
ax2.axhline(y=mean_with_ceo/1000, color='r', linestyle='--', linewidth=2,
            label=f'Mean = ${mean_with_ceo/1000:.1f}K')
ax2.text(len(salaries_with_ceo)-1, 520, 'CEO Salary\n(Outlier)', 
         fontsize=10, ha='center', color='red', fontweight='bold')
ax2.set_title('Salaries With Outlier', fontsize=13, fontweight='bold')
ax2.set_xlabel('Employee Index')
ax2.set_ylabel('Salary ($1000s)')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(0, 550)

plt.tight_layout()
plt.show()

print("\\n⚠️  The mean is NOT representative when outliers are present!")"""))

# Cell 8: The Median
notebook["cells"].append(create_markdown_cell("""## 3. The Median (Middle Value)

### Definition

The **median** is the middle value when data is sorted:

- **Odd n**: Median = middle value
- **Even n**: Median = average of two middle values

$$\\text{Median position} = \\frac{n + 1}{2}$$

### Properties

- **Resistant to outliers**: Extreme values don't affect it
- **Positional**: Only considers position, not actual values
- **50th percentile**: Half the data is below, half above
- **Better for skewed data**: More representative than mean

### Calculating Median"""))

# Cell 9: Calculate median
notebook["cells"].append(create_code_cell("""# Same salary data
salaries_normal = np.array([45000, 48000, 50000, 52000, 47000, 49000, 51000, 46000, 48500, 50500])
salaries_with_ceo = np.append(salaries_normal, 500000)

# Calculate medians
median_normal = np.median(salaries_normal)
median_with_ceo = np.median(salaries_with_ceo)

# Calculate means for comparison
mean_normal = np.mean(salaries_normal)
mean_with_ceo = np.mean(salaries_with_ceo)

print("Median vs Mean Comparison:")
print("=" * 60)

print(f"\\nWithout CEO (10 employees):")
sorted_normal = np.sort(salaries_normal)
print(f"  Sorted salaries: {sorted_normal}")
print(f"  Middle positions: {len(sorted_normal)//2 - 1} and {len(sorted_normal)//2}")
print(f"  Middle values: ${sorted_normal[4]:,} and ${sorted_normal[5]:,}")
print(f"  Median: ${median_normal:,.2f}")
print(f"  Mean: ${mean_normal:,.2f}")

print(f"\\nWith CEO (11 employees):")
sorted_with_ceo = np.sort(salaries_with_ceo)
print(f"  Sorted salaries: {sorted_with_ceo}")
print(f"  Middle position: {len(sorted_with_ceo)//2}")
print(f"  Median: ${median_with_ceo:,.2f}")
print(f"  Mean: ${mean_with_ceo:,.2f}")

print(f"\\nImpact of outlier:")
print(f"  Median change: ${median_with_ceo - median_normal:,.2f}")
print(f"  Mean change: ${mean_with_ceo - mean_normal:,.2f}")
print(f"\\n✓ Median is ROBUST to outliers!")

# Visualize comparison
fig, ax = plt.subplots(figsize=(10, 6))

# Box plot
positions = [1, 2]
bp = ax.boxplot([salaries_normal, salaries_with_ceo], positions=positions,
                 labels=['Without CEO', 'With CEO'], widths=0.6,
                 patch_artist=True)

# Color boxes
for patch, color in zip(bp['boxes'], ['lightblue', 'lightcoral']):
    patch.set_facecolor(color)

# Add mean markers
ax.plot(1, mean_normal, 'rs', markersize=12, label='Mean', zorder=3)
ax.plot(2, mean_with_ceo, 'rs', markersize=12, zorder=3)

# Add labels
ax.text(1, median_normal - 2000, f'Median: ${median_normal/1000:.1f}K',
        ha='center', fontsize=10, fontweight='bold')
ax.text(2, median_with_ceo - 2000, f'Median: ${median_with_ceo/1000:.1f}K',
        ha='center', fontsize=10, fontweight='bold')

ax.set_ylabel('Salary ($)', fontsize=12)
ax.set_title('Median vs Mean: Robustness to Outliers', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
plt.show()"""))

# Cell 10: The Mode
notebook["cells"].append(create_markdown_cell("""## 4. The Mode (Most Frequent Value)

### Definition

The **mode** is the value that appears most frequently in the dataset.

### Properties

- **For categorical data**: The only measure that works for nominal data
- **Multiple modes**: A dataset can be unimodal, bimodal, or multimodal
- **No mode**: When all values appear with equal frequency
- **Not unique**: Can have several modes

### Types of Distributions by Modality

- **Unimodal**: One peak
- **Bimodal**: Two peaks
- **Multimodal**: Multiple peaks
- **Uniform**: No clear mode"""))

# Cell 11: Calculate mode
notebook["cells"].append(create_code_cell("""# Different datasets demonstrating mode
print("=" * 60)
print("MODE EXAMPLES")
print("=" * 60)

# Example 1: Shoe sizes (discrete numerical)
shoe_sizes = np.array([7, 8, 8, 9, 7, 8, 10, 8, 9, 8, 7, 8])
mode_shoe = stats.mode(shoe_sizes, keepdims=True)
print(f"\\n1. Shoe Sizes: {shoe_sizes}")
print(f"   Mode: {mode_shoe.mode[0]} (appears {mode_shoe.count[0]} times)")

# Example 2: Categorical data (favorite color)
colors = np.array(['Red', 'Blue', 'Blue', 'Green', 'Blue', 'Red', 'Blue', 'Yellow'])
unique, counts = np.unique(colors, return_counts=True)
mode_idx = np.argmax(counts)
print(f"\\n2. Favorite Colors: {colors}")
print(f"   Frequency:")
for color, count in zip(unique, counts):
    print(f"     {color}: {count}")
print(f"   Mode: {unique[mode_idx]} (appears {counts[mode_idx]} times)")

# Example 3: Bimodal distribution
grades = np.array([65, 70, 68, 72, 70, 88, 92, 90, 88, 92, 70, 88])
print(f"\\n3. Test Grades: {grades}")
unique_grades, counts_grades = np.unique(grades, return_counts=True)
max_count = counts_grades.max()
modes = unique_grades[counts_grades == max_count]
print(f"   Bimodal: {modes} (each appears {max_count} times)")

# Visualize distributions
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Shoe sizes
ax1 = axes[0]
unique_sizes, counts_sizes = np.unique(shoe_sizes, return_counts=True)
ax1.bar(unique_sizes, counts_sizes, edgecolor='black', alpha=0.7)
ax1.axvline(x=mode_shoe.mode[0], color='r', linestyle='--', linewidth=2,
            label=f'Mode = {mode_shoe.mode[0]}')
ax1.set_xlabel('Shoe Size', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Unimodal: Shoe Sizes', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Colors
ax2 = axes[1]
ax2.bar(unique, counts, edgecolor='black', alpha=0.7, color=unique.lower())
ax2.set_xlabel('Color', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Categorical: Favorite Colors', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3, axis='y')

# Grades (bimodal)
ax3 = axes[2]
ax3.hist(grades, bins=range(60, 100, 5), edgecolor='black', alpha=0.7)
for mode in modes:
    ax3.axvline(x=mode, color='r', linestyle='--', linewidth=2)
ax3.text(modes[0], ax3.get_ylim()[1]*0.9, f'Mode 1: {modes[0]}',
         fontsize=10, ha='center', color='red')
ax3.text(modes[1], ax3.get_ylim()[1]*0.8, f'Mode 2: {modes[1]}',
         fontsize=10, ha='center', color='red')
ax3.set_xlabel('Grade', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title('Bimodal: Test Grades', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()"""))

# Cell 12: Choosing the right measure
notebook["cells"].append(create_markdown_cell("""## 5. Choosing the Right Measure

### Decision Framework

```
Is data categorical?
├─ YES → Use MODE
└─ NO (numerical) → Continue...
    │
    Are there outliers?
    ├─ YES → Use MEDIAN
    └─ NO → Continue...
        │
        Is distribution symmetric?
        ├─ YES → Use MEAN (or all three)
        └─ NO (skewed) → Use MEDIAN
```

### Comparison Table

| Characteristic | Mean | Median | Mode |
|----------------|------|--------|------|
| Data type | Numerical only | Numerical/Ordinal | Any type |
| Sensitive to outliers | YES ✗ | NO ✓ | NO ✓ |
| Uses all data | YES | NO | NO |
| Unique value | YES | Usually | May have multiple |
| Best for | Symmetric, no outliers | Skewed, outliers | Categorical, peaks |
| Mathematical properties | Many | Fewer | Fewest |

### Real-World Guidelines

1. **Income/Salary data**: Use MEDIAN (usually right-skewed with high earners)
2. **Test scores**: Use MEAN (usually symmetric)
3. **House prices**: Use MEDIAN (outliers common)
4. **Customer ratings**: Use MODE or MEDIAN
5. **Survey responses**: Use MODE for categories, MEDIAN for ordinal"""))

# Cell 13: Skewness and central tendency
notebook["cells"].append(create_markdown_cell("""## 6. Relationship with Skewness

The relationship between mean, median, and mode reveals the distribution's shape:

- **Symmetric**: Mean ≈ Median ≈ Mode
- **Right-skewed (positive)**: Mode < Median < Mean
- **Left-skewed (negative)**: Mean < Median < Mode

Let's create and visualize these distributions:"""))

# Cell 14: Skewness visualization
notebook["cells"].append(create_code_cell("""from scipy.stats import skewnorm

# Generate different distributions
np.random.seed(42)

# Symmetric (normal)
symmetric = np.random.normal(100, 15, 1000)

# Right-skewed (positive skew)
right_skewed = skewnorm.rvs(5, loc=100, scale=20, size=1000)

# Left-skewed (negative skew)
left_skewed = skewnorm.rvs(-5, loc=100, scale=20, size=1000)

# Calculate measures for each
def calculate_measures(data, name):
    mean = np.mean(data)
    median = np.median(data)
    mode_result = stats.mode(data.round(), keepdims=True)
    mode = mode_result.mode[0]
    skewness = stats.skew(data)
    
    print(f"\\n{name}:")
    print(f"  Mean:   {mean:.2f}")
    print(f"  Median: {median:.2f}")
    print(f"  Mode:   {mode:.2f}")
    print(f"  Skewness: {skewness:.2f}")
    
    return mean, median, mode

print("=" * 60)
print("SKEWNESS AND CENTRAL TENDENCY")
print("=" * 60)

# Calculate measures
sym_measures = calculate_measures(symmetric, "Symmetric Distribution")
right_measures = calculate_measures(right_skewed, "Right-Skewed Distribution")
left_measures = calculate_measures(left_skewed, "Left-Skewed Distribution")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

datasets = [
    (symmetric, sym_measures, "Symmetric\\n(Normal)", axes[0]),
    (right_skewed, right_measures, "Right-Skewed\\n(Positive)", axes[1]),
    (left_skewed, left_measures, "Left-Skewed\\n(Negative)", axes[2])
]

colors = ['blue', 'green', 'red']
labels = ['Mean', 'Median', 'Mode']

for data, measures, title, ax in datasets:
    # Histogram
    ax.hist(data, bins=40, alpha=0.6, edgecolor='black', density=True)
    
    # Overlay KDE
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(data)
    x_range = np.linspace(data.min(), data.max(), 100)
    ax.plot(x_range, kde(x_range), 'k-', linewidth=2, label='Density')
    
    # Mark mean, median, mode
    for measure, color, label in zip(measures, colors, labels):
        ax.axvline(x=measure, color=color, linestyle='--', linewidth=2, label=label)
    
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Value', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Summary
print("\\n" + "=" * 60)
print("KEY OBSERVATIONS:")
print("=" * 60)
print("• Symmetric: Mean ≈ Median ≈ Mode")
print("• Right-Skewed: Mode < Median < Mean (tail pulls mean right)")
print("• Left-Skewed: Mean < Median < Mode (tail pulls mean left)")"""))

# Cell 15: Weighted mean
notebook["cells"].append(create_markdown_cell("""## 7. Weighted Mean

### Definition

When values have different importance (weights), use the **weighted mean**:

$$\\bar{x}_w = \\frac{\\sum_{i=1}^{n} w_i x_i}{\\sum_{i=1}^{n} w_i}$$

### Applications

- **Grade calculation**: Different assignments have different weights
- **Stock portfolio**: Weighted by investment amount
- **Customer satisfaction**: Weighted by customer size
- **Voting systems**: Weighted by population or representation

### Example: Course Grade Calculation"""))

# Cell 16: Weighted mean calculation
notebook["cells"].append(create_code_cell("""# Course grade components
components = pd.DataFrame({
    'Component': ['Homework', 'Midterm', 'Project', 'Final Exam'],
    'Score': [85, 78, 92, 88],
    'Weight': [0.20, 0.25, 0.25, 0.30]
})

print("Course Grade Calculation")
print("=" * 60)
print(components.to_string(index=False))

# Calculate weighted mean
weighted_mean = np.average(components['Score'], weights=components['Weight'])

# Compare with simple mean
simple_mean = components['Score'].mean()

print(f"\\nSimple average (unweighted): {simple_mean:.2f}")
print(f"Weighted average (final grade): {weighted_mean:.2f}")
print(f"Difference: {abs(weighted_mean - simple_mean):.2f} points")

# Show calculation breakdown
components['Contribution'] = components['Score'] * components['Weight']
print(f"\\nBreakdown:")
print(components[['Component', 'Score', 'Weight', 'Contribution']].to_string(index=False))
print(f"\\nSum of weighted scores: {components['Contribution'].sum():.2f}")
print(f"Sum of weights: {components['Weight'].sum():.2f}")
print(f"Weighted mean: {components['Contribution'].sum() / components['Weight'].sum():.2f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Component scores
x = np.arange(len(components))
bars1 = ax1.bar(x, components['Score'], alpha=0.7, edgecolor='black')
ax1.axhline(y=simple_mean, color='r', linestyle='--', linewidth=2,
            label=f'Simple Mean = {simple_mean:.1f}')
ax1.axhline(y=weighted_mean, color='g', linestyle='--', linewidth=2,
            label=f'Weighted Mean = {weighted_mean:.1f}')
ax1.set_xlabel('Component', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Component Scores', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(components['Component'], rotation=45, ha='right')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Weighted contributions
bars2 = ax2.bar(x, components['Contribution'], alpha=0.7, edgecolor='black', color='green')
ax2.set_xlabel('Component', fontsize=12)
ax2.set_ylabel('Weighted Contribution', fontsize=12)
ax2.set_title('Contribution to Final Grade', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(components['Component'], rotation=45, ha='right')

# Add percentage labels
for i, (bar, weight) in enumerate(zip(bars2, components['Weight'])):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, height + 0.5,
             f'{weight*100:.0f}%', ha='center', fontsize=10, fontweight='bold')

ax2.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()"""))

# Cell 17: Real-world application
notebook["cells"].append(create_markdown_cell("""## 8. Real-World Application: Housing Data Analysis

Let's apply all three measures to a real-world scenario: analyzing house prices in a neighborhood."""))

# Cell 18: Housing data analysis
notebook["cells"].append(create_code_cell("""# Generate realistic housing price data
np.random.seed(42)

# Base prices (log-normal distribution - realistic for house prices)
base_prices = np.random.lognormal(mean=12.5, sigma=0.3, size=100) * 1000

# Add a few luxury homes (outliers)
luxury_homes = np.array([1500000, 1800000, 2200000])
all_prices = np.concatenate([base_prices, luxury_homes])

# Create DataFrame
housing_df = pd.DataFrame({
    'Price': all_prices,
    'Type': ['Standard']*100 + ['Luxury']*3
})

# Calculate all measures
mean_price = housing_df['Price'].mean()
median_price = housing_df['Price'].median()
mode_result = stats.mode(housing_df['Price'].round(-4), keepdims=True)  # Round to nearest 10k
mode_price = mode_result.mode[0]

print("=" * 70)
print("HOUSING MARKET ANALYSIS")
print("=" * 70)
print(f"\\nDataset: {len(housing_df)} houses")
print(f"  Standard homes: {100}")
print(f"  Luxury homes: {3}")

print(f"\\nCentral Tendency Measures:")
print(f"  Mean:   ${mean_price:,.0f}")
print(f"  Median: ${median_price:,.0f}")
print(f"  Mode:   ${mode_price:,.0f}")

print(f"\\nPrice Range:")
print(f"  Minimum: ${housing_df['Price'].min():,.0f}")
print(f"  Maximum: ${housing_df['Price'].max():,.0f}")

print(f"\\nWhich measure is most appropriate?")
print(f"  ✓ MEDIAN - Best represents typical home price")
print(f"  ✗ MEAN - Inflated by luxury homes")
print(f"  ✗ MODE - Not very informative for continuous data")

# Comprehensive visualization
fig = plt.figure(figsize=(15, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 1. Histogram with measures
ax1 = fig.add_subplot(gs[0, :])
ax1.hist(housing_df['Price']/1000, bins=30, edgecolor='black', alpha=0.7)
ax1.axvline(x=mean_price/1000, color='r', linestyle='--', linewidth=2,
            label=f'Mean = ${mean_price/1000:.0f}K')
ax1.axvline(x=median_price/1000, color='g', linestyle='--', linewidth=2,
            label=f'Median = ${median_price/1000:.0f}K')
ax1.set_xlabel('Price ($1000s)', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12)
ax1.set_title('Distribution of House Prices', fontsize=14, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3, axis='y')

# 2. Box plot
ax2 = fig.add_subplot(gs[1, 0])
bp = ax2.boxplot(housing_df['Price']/1000, vert=True, patch_artist=True)
bp['boxes'][0].set_facecolor('lightblue')
ax2.plot(1, mean_price/1000, 'rs', markersize=10, label='Mean')
ax2.set_ylabel('Price ($1000s)', fontsize=12)
ax2.set_title('Box Plot (shows outliers)', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 3. Scatter plot by type
ax3 = fig.add_subplot(gs[1, 1])
for home_type in housing_df['Type'].unique():
    subset = housing_df[housing_df['Type'] == home_type]
    color = 'red' if home_type == 'Luxury' else 'blue'
    marker = 's' if home_type == 'Luxury' else 'o'
    ax3.scatter(range(len(subset)), subset['Price']/1000, 
                label=home_type, alpha=0.6, s=60, color=color, marker=marker)
ax3.axhline(y=median_price/1000, color='g', linestyle='--', linewidth=2, alpha=0.7)
ax3.set_xlabel('House Index', fontsize=12)
ax3.set_ylabel('Price ($1000s)', fontsize=12)
ax3.set_title('Prices by Type', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Statistical summary
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

summary_text = f'''
STATISTICAL SUMMARY

Measures of Central Tendency:
  • Mean:   ${mean_price:,.0f}  (Average - pulled up by luxury homes)
  • Median: ${median_price:,.0f}  (Middle value - best representative)
  • Mode:   ${mode_price:,.0f}  (Most common price range)

Key Insights:
  • Mean > Median indicates RIGHT-SKEWED distribution
  • Presence of luxury homes creates outliers
  • Median better represents "typical" home price
  • Mean is ${(mean_price - median_price):,.0f} higher than median

Recommendation for Buyers:
  "Typical home price: ${median_price:,.0f}" is more accurate than
  "Average home price: ${mean_price:,.0f}"
'''

ax4.text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
         verticalalignment='center',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.show()

# Compare standard vs luxury
print("\\n" + "=" * 70)
print("COMPARISON BY TYPE")
print("=" * 70)
for home_type in ['Standard', 'Luxury']:
    subset = housing_df[housing_df['Type'] == home_type]
    print(f"\\n{home_type} Homes:")
    print(f"  Count:  {len(subset)}")
    print(f"  Mean:   ${subset['Price'].mean():,.0f}")
    print(f"  Median: ${subset['Price'].median():,.0f}")
    print(f"  Range:  ${subset['Price'].min():,.0f} - ${subset['Price'].max():,.0f}")"""))

# Cell 19: Practice problems
notebook["cells"].append(create_markdown_cell("""## 9. Practice Problems

Test your understanding with these exercises:

**Problem 1**: Calculate mean, median, and mode
**Problem 2**: Identify the most appropriate measure
**Problem 3**: Weighted mean calculation
**Problem 4**: Analyze impact of outliers
**Problem 5**: Real dataset analysis"""))

# Cell 20: Solutions
notebook["cells"].append(create_code_cell("""print("=" * 70)
print("PRACTICE PROBLEM SOLUTIONS")
print("=" * 70)

# Problem 1: Calculate mean, median, mode
print("\\nProblem 1: Dataset = [15, 18, 21, 21, 25, 27, 21, 30, 33]")
data1 = np.array([15, 18, 21, 21, 25, 27, 21, 30, 33])
print(f"  Mean:   {np.mean(data1):.2f}")
print(f"  Median: {np.median(data1):.2f}")
mode1 = stats.mode(data1, keepdims=True)
print(f"  Mode:   {mode1.mode[0]} (appears {mode1.count[0]} times)")

# Problem 2: Which measure for skewed salary data?
print("\\nProblem 2: Employee salaries: [40K, 42K, 45K, 48K, 50K, 200K]")
salaries2 = np.array([40, 42, 45, 48, 50, 200])
print(f"  Mean:   ${np.mean(salaries2):.1f}K")
print(f"  Median: ${np.median(salaries2):.1f}K")
print(f"  Best choice: MEDIAN (resistant to outlier)")

# Problem 3: Weighted mean for portfolio
print("\\nProblem 3: Investment portfolio returns")
returns = np.array([8, 12, -3, 15])  # percentages
amounts = np.array([10000, 25000, 5000, 15000])  # dollars
weighted_return = np.average(returns, weights=amounts)
print(f"  Returns: {returns}%")
print(f"  Amounts: ${amounts}")
print(f"  Weighted average return: {weighted_return:.2f}%")

# Problem 4: Add outlier and compare
print("\\nProblem 4: Impact of outlier")
data4_original = np.array([10, 12, 15, 18, 20])
data4_with_outlier = np.append(data4_original, 100)
print(f"  Original - Mean: {np.mean(data4_original):.1f}, Median: {np.median(data4_original):.1f}")
print(f"  With outlier - Mean: {np.mean(data4_with_outlier):.1f}, Median: {np.median(data4_with_outlier):.1f}")
print(f"  Median changed by: {np.median(data4_with_outlier) - np.median(data4_original):.1f}")
print(f"  Mean changed by: {np.mean(data4_with_outlier) - np.mean(data4_original):.1f}")

# Problem 5: Real dataset
print("\\nProblem 5: Analyze this dataset")
np.random.seed(123)
data5 = np.random.gamma(2, 2, 50)
print(f"  Dataset size: {len(data5)}")
print(f"  Mean:   {np.mean(data5):.2f}")
print(f"  Median: {np.median(data5):.2f}")
print(f"  Std Dev: {np.std(data5):.2f}")
print(f"  Skewness: {stats.skew(data5):.2f}")
print(f"  Distribution is: Right-skewed (mean > median)")"""))

# Cell 21: Summary
notebook["cells"].append(create_markdown_cell("""## 10. Summary and Key Takeaways

### Essential Concepts

✓ **Mean**: Average of all values
  - Best for symmetric distributions without outliers
  - Sensitive to extreme values
  - Uses all data points

✓ **Median**: Middle value when sorted
  - Best for skewed distributions or with outliers
  - Resistant to extreme values
  - 50th percentile

✓ **Mode**: Most frequent value
  - Only measure for categorical data
  - Can identify peaks in distribution
  - May have multiple modes or no mode

### Decision Rules

1. **Categorical data** → Use MODE
2. **Outliers present** → Use MEDIAN
3. **Symmetric, no outliers** → Use MEAN
4. **Skewed distribution** → Use MEDIAN
5. **Different importance** → Use WEIGHTED MEAN

### Skewness Relationships

- **Symmetric**: Mean ≈ Median ≈ Mode
- **Right-skewed**: Mode < Median < Mean
- **Left-skewed**: Mean < Median < Mode

### Data Science Applications

- **EDA**: Summarize datasets quickly
- **Outlier detection**: Compare mean and median
- **Feature engineering**: Create aggregated features
- **Model evaluation**: Average performance metrics
- **A/B testing**: Compare group centers

### Common Mistakes to Avoid

❌ Using mean for skewed data
❌ Ignoring outliers
❌ Using mode for continuous data without context
❌ Not considering data type
❌ Forgetting to check distribution shape

### Next Steps

**Week 5**: Measures of Dispersion (Range, Variance, Standard Deviation, IQR)

---

**Remember**: The "center" is only half the story - you also need to understand the spread!"""))

# Save notebook
output_path = "/Users/godsonkurishinkal/Projects/iitm-bs-learning/01-Foundation-Level/02-Statistics-I/notebooks/week-04-central-tendency-measures.ipynb"

with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"✓ Notebook created successfully")
print(f"✓ Output: {output_path}")
print(f"✓ Total cells: {len(notebook['cells'])}")
markdown_count = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
code_count = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'code')
print(f"✓ Markdown cells: {markdown_count}")
print(f"✓ Code cells: {code_count}")
