#!/usr/bin/env python3
"""Generate Week 6: Correlation and Association"""
import json
import sys

def cm(content): return {"cell_type": "markdown", "metadata": {}, "source": content.split('\n')}
def cc(code): return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.split('\n')}

nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python", "version": "3.9.6"}}, "nbformat": 4, "nbformat_minor": 4}

nb["cells"].extend([
cm("""# Week 6: Correlation and Association

**Course**: BSMA1002 - Statistics for Data Science I  
**Topic**: Measuring Relationships Between Variables

## Learning Objectives
- Understand correlation vs causation
- Calculate and interpret Pearson correlation
- Use Spearman rank correlation for non-linear relationships
- Visualize relationships with scatter plots
- Understand correlation matrix and heatmaps
- Apply correlation analysis to real datasets"""),

cc("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
plt.style.use('seaborn-v0_8-darkgrid')
print("✓ Libraries loaded")"""),

cm("""## 1. What is Correlation?

**Correlation**: Measures the strength and direction of LINEAR relationship between two variables

$$r = \\frac{\\sum(x_i - \\bar{x})(y_i - \\bar{y})}{\\sqrt{\\sum(x_i - \\bar{x})^2 \\sum(y_i - \\bar{y})^2}}$$

### Range and Interpretation
- $-1 \\leq r \\leq +1$
- $r = +1$: Perfect positive linear relationship
- $r = 0$: No linear relationship
- $r = -1$: Perfect negative linear relationship

### ⚠️ Correlation ≠ Causation!"""),

cc("""# Visualize different correlation strengths
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

correlations = [1.0, 0.8, 0.5, 0.0, -0.5, -0.8]
n = 50

for ax, r in zip(axes, correlations):
    # Generate correlated data
    x = np.random.randn(n)
    y = r * x + np.sqrt(1 - r**2) * np.random.randn(n)
    
    # Calculate actual correlation
    actual_r = np.corrcoef(x, y)[0, 1]
    
    # Plot
    ax.scatter(x, y, alpha=0.6, s=50)
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    ax.plot(np.sort(x), p(np.sort(x)), "r--", linewidth=2)
    
    ax.set_title(f'r = {actual_r:.2f}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Variable X')
    ax.set_ylabel('Variable Y')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("📊 Correlation strength interpretation:")
print("  |r| = 0.0-0.3: Weak")
print("  |r| = 0.3-0.7: Moderate") 
print("  |r| = 0.7-1.0: Strong")"""),

cm("""## 2. Pearson Correlation Coefficient

Calculate correlation step-by-step for understanding."""),

cc("""# Step-by-step correlation calculation
study_hours = np.array([2, 3, 4, 5, 6, 7, 8])
exam_scores = np.array([50, 55, 60, 70, 75, 85, 90])

print("Pearson Correlation Calculation")
print("="*60)
print(f"Study Hours (X): {study_hours}")
print(f"Exam Scores (Y): {exam_scores}")

# Step 1: Means
x_mean = study_hours.mean()
y_mean = exam_scores.mean()
print(f"\\nMeans: X̄ = {x_mean:.2f}, Ȳ = {y_mean:.2f}")

# Step 2: Deviations
x_dev = study_hours - x_mean
y_dev = exam_scores - y_mean
print(f"\\nDeviations from mean:")
for i, (x, y, xd, yd) in enumerate(zip(study_hours, exam_scores, x_dev, y_dev)):
    print(f"  Point {i+1}: ({x}, {y}) → ({xd:+.2f}, {yd:+.2f})")

# Step 3: Products and sums
products = x_dev * y_dev
sum_products = products.sum()
sum_x_sq = (x_dev ** 2).sum()
sum_y_sq = (y_dev ** 2).sum()

print(f"\\nProducts: {products}")
print(f"Sum of products: {sum_products:.2f}")
print(f"Sum of X² deviations: {sum_x_sq:.2f}")
print(f"Sum of Y² deviations: {sum_y_sq:.2f}")

# Step 4: Correlation
r = sum_products / np.sqrt(sum_x_sq * sum_y_sq)
print(f"\\nr = {sum_products:.2f} / √({sum_x_sq:.2f} × {sum_y_sq:.2f})")
print(f"r = {r:.4f}")

# Verify
r_numpy = np.corrcoef(study_hours, exam_scores)[0, 1]
print(f"\\n✓ NumPy verification: r = {r_numpy:.4f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.scatter(study_hours, exam_scores, s=100, alpha=0.6, edgecolors='black', linewidths=2)
z = np.polyfit(study_hours, exam_scores, 1)
p = np.poly1d(z)
ax1.plot(study_hours, p(study_hours), "r--", linewidth=2, label='Best fit line')
ax1.set_xlabel('Study Hours', fontsize=12)
ax1.set_ylabel('Exam Scores', fontsize=12)
ax1.set_title(f'Strong Positive Correlation (r = {r:.3f})', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Show deviations
ax2.scatter(x_dev, y_dev, s=100, alpha=0.6, edgecolors='black', linewidths=2)
ax2.axhline(0, color='k', linestyle='-', linewidth=1)
ax2.axvline(0, color='k', linestyle='-', linewidth=1)
for xd, yd in zip(x_dev, y_dev):
    ax2.plot([0, xd], [0, yd], 'g--', alpha=0.5)
ax2.set_xlabel('X deviation from mean', fontsize=12)
ax2.set_ylabel('Y deviation from mean', fontsize=12)
ax2.set_title('Deviations (both mostly same direction = positive r)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""),

cm("""## 3. Spearman Rank Correlation

**Non-parametric alternative** for:
- Non-linear monotonic relationships
- Ordinal data
- Outliers present

Calculated on **ranks** instead of raw values."""),

cc("""# Compare Pearson vs Spearman
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
y_linear = 2 * x + 5  # Perfect linear
y_exponential = np.exp(x / 3)  # Non-linear but monotonic
y_with_outlier = 2 * x + 5
y_with_outlier[-1] = 100  # Add outlier

print("Pearson vs Spearman Correlation")
print("="*70)

# Linear relationship
r_pearson_lin = stats.pearsonr(x, y_linear)[0]
r_spearman_lin = stats.spearmanr(x, y_linear)[0]
print(f"\\n1. Linear Relationship:")
print(f"   Pearson:  {r_pearson_lin:.4f}")
print(f"   Spearman: {r_spearman_lin:.4f}")

# Non-linear monotonic
r_pearson_exp = stats.pearsonr(x, y_exponential)[0]
r_spearman_exp = stats.spearmanr(x, y_exponential)[0]
print(f"\\n2. Exponential (Monotonic) Relationship:")
print(f"   Pearson:  {r_pearson_exp:.4f} (underestimates)")
print(f"   Spearman: {r_spearman_exp:.4f} (captures monotonicity)")

# With outlier
r_pearson_out = stats.pearsonr(x, y_with_outlier)[0]
r_spearman_out = stats.spearmanr(x, y_with_outlier)[0]
print(f"\\n3. Linear with Outlier:")
print(f"   Pearson:  {r_pearson_out:.4f} (affected)")
print(f"   Spearman: {r_spearman_out:.4f} (resistant)")

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

datasets = [
    (x, y_linear, "Linear", r_pearson_lin, r_spearman_lin),
    (x, y_exponential, "Exponential", r_pearson_exp, r_spearman_exp),
    (x, y_with_outlier, "With Outlier", r_pearson_out, r_spearman_out)
]

for ax, (x_data, y_data, title, rp, rs) in zip(axes, datasets):
    ax.scatter(x_data, y_data, s=80, alpha=0.6, edgecolors='black')
    ax.set_title(f'{title}\\nPearson={rp:.3f}, Spearman={rs:.3f}',
                fontsize=11, fontweight='bold')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""),

cm("""## 4. Correlation Matrix & Heatmap

Analyze multiple variable relationships simultaneously."""),

cc("""# Create synthetic dataset with multiple variables
np.random.seed(42)
n = 100

data = pd.DataFrame({
    'Height': np.random.normal(170, 10, n),
    'Weight': np.random.normal(70, 15, n),
    'Age': np.random.randint(20, 60, n),
    'Income': np.random.normal(50000, 15000, n)
})

# Add correlated variables
data['Weight'] = data['Height'] * 0.8 + np.random.normal(0, 5, n)  # Correlated with height
data['Shoe_Size'] = data['Height'] * 0.15 + np.random.normal(0, 1, n)  # Also correlated

print("Correlation Matrix")
print("="*70)
corr_matrix = data.corr()
print(corr_matrix.round(3))

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax1)
ax1.set_title('Correlation Matrix Heatmap', fontsize=13, fontweight='bold')

# Scatter matrix for key relationships
scatter_data = data[['Height', 'Weight', 'Shoe_Size']]
pd.plotting.scatter_matrix(scatter_data, alpha=0.6, figsize=(10, 10), 
                          diagonal='hist', ax=ax2)
ax2.remove()  # Remove the extra axis

plt.tight_layout()
plt.show()

# Find strongest correlations
corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j], 
                          corr_matrix.iloc[i, j]))

corr_pairs_sorted = sorted(corr_pairs, key=lambda x: abs(x[2]), reverse=True)
print(f"\\nStrongest Correlations:")
for var1, var2, r in corr_pairs_sorted[:3]:
    print(f"  {var1} ↔ {var2}: r = {r:.3f}")"""),

cm("""## 5. Real-World Application: Housing Prices

Analyze which features correlate most with house prices."""),

cc("""# Simulate realistic housing dataset
np.random.seed(42)
n_houses = 200

housing = pd.DataFrame({
    'Square_Feet': np.random.randint(800, 4000, n_houses),
    'Bedrooms': np.random.randint(1, 6, n_houses),
    'Bathrooms': np.random.randint(1, 4, n_houses),
    'Age_Years': np.random.randint(0, 50, n_houses),
    'Distance_CBD_km': np.random.uniform(1, 30, n_houses),
    'Crime_Rate': np.random.uniform(1, 10, n_houses)
})

# Generate price based on features (with realistic relationships)
housing['Price_k'] = (
    housing['Square_Feet'] * 0.2 +
    housing['Bedrooms'] * 20 +
    housing['Bathrooms'] * 15 -
    housing['Age_Years'] * 2 -
    housing['Distance_CBD_km'] * 5 -
    housing['Crime_Rate'] * 10 +
    np.random.normal(0, 50, n_houses)
)

print("Housing Price Correlation Analysis")
print("="*70)
print(f"Dataset: {len(housing)} houses")
print(f"\\nFeatures: {list(housing.columns)}")

# Calculate correlations with price
correlations = housing.corr()['Price_k'].sort_values(ascending=False)
print(f"\\nCorrelations with Price:")
for feature, corr in correlations.items():
    if feature != 'Price_k':
        strength = "Strong" if abs(corr) > 0.7 else "Moderate" if abs(corr) > 0.4 else "Weak"
        direction = "positive" if corr > 0 else "negative"
        print(f"  {feature:20s}: {corr:+.3f} ({strength} {direction})")

# Comprehensive visualization
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# Heatmap
ax1 = fig.add_subplot(gs[0, :])
sns.heatmap(housing.corr(), annot=True, fmt='.2f', cmap='RdYlGn', 
            center=0, square=True, linewidths=1, ax=ax1)
ax1.set_title('Full Correlation Matrix', fontsize=14, fontweight='bold')

# Top 6 scatter plots
features_to_plot = ['Square_Feet', 'Bedrooms', 'Bathrooms', 
                    'Age_Years', 'Distance_CBD_km', 'Crime_Rate']

for idx, feature in enumerate(features_to_plot):
    row = (idx // 3) + 1
    col = idx % 3
    ax = fig.add_subplot(gs[row, col])
    
    ax.scatter(housing[feature], housing['Price_k'], alpha=0.5, s=30)
    
    # Add regression line
    z = np.polyfit(housing[feature], housing['Price_k'], 1)
    p = np.poly1d(z)
    x_line = np.linspace(housing[feature].min(), housing[feature].max(), 100)
    ax.plot(x_line, p(x_line), "r--", linewidth=2)
    
    corr = correlations[feature]
    ax.set_xlabel(feature.replace('_', ' '), fontsize=10)
    ax.set_ylabel('Price ($k)', fontsize=10)
    ax.set_title(f'r = {corr:+.3f}', fontsize=11, fontweight='bold')
    ax.grid(True, alpha=0.3)

plt.show()

print(f"\\n💡 Key Insights:")
print(f"   • Square footage has strongest positive correlation")
print(f"   • Crime rate and distance from CBD negatively correlate")
print(f"   • Age of house reduces price")"""),

cm("""## Summary & Key Takeaways

### Correlation Types

| Type | Use Case | Range | Sensitive to Outliers |
|------|----------|-------|----------------------|
| **Pearson** | Linear relationships | [-1, 1] | Yes |
| **Spearman** | Monotonic, ordinal data | [-1, 1] | No |

### Critical Reminders

⚠️ **Correlation ≠ Causation**
- Ice cream sales and drownings correlate (both high in summer)
- Doesn't mean ice cream causes drowning!

✓ **Always visualize** - correlation alone can miss patterns  
✓ **Check for outliers** - can distort Pearson correlation  
✓ **Consider non-linear relationships** - use Spearman or transformations

### Next Steps
- **Week 7**: Probability fundamentals
- **Future**: Regression analysis (predict Y from X)

---
**Practice**: Apply correlation analysis to your own datasets!""")
])

output = "/Users/godsonkurishinkal/Projects/iitm-bs-learning/01-Foundation-Level/02-Statistics-I/notebooks/week-06-correlation-association.ipynb"
with open(output, 'w') as f: json.dump(nb, f, indent=2)
print(f"✓ Week 6 created: {len(nb['cells'])} cells")
