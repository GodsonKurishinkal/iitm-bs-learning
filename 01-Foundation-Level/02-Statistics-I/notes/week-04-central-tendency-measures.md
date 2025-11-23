# Week 4: Association Between Two Variables - Correlation and Relationships

---
Date: 2025-11-22
Course: BSMA1002 - Statistics for Data Science I
Level: Foundation
Week: 4 of 12
Source: IIT Madras Statistics I Week 4
Topic Area: Bivariate Analysis - Association
Tags: #BSMA1002 #Correlation #Covariance #ScatterPlot #Association #Week4 #Foundation
---

## Overview (BLUF)

**One-sentence definition**: Association analysis examines how two variables relate to each other—whether they tend to increase together (positive correlation), move in opposite directions (negative correlation), or show no systematic relationship.

**Why it matters**: Understanding relationships between variables is fundamental to prediction, feature selection in ML, and causal inference. Before building models, you must know which features are related to your target variable and which features are redundant (highly correlated with each other). Correlation ≠ causation, but it's the starting point for investigation.

**When to use**: Feature selection (which variables predict target?), multicollinearity detection (are predictors too similar?), exploratory data analysis (what patterns exist?), validating hypotheses (does exercise correlate with health?), portfolio diversification (find uncorrelated assets).

**Prerequisites**: Descriptive statistics ([week-03](week-03-numerical-data-visualization.md)), variance and standard deviation, cross-tabulation for categorical data ([week-02](week-02-categorical-data-analysis.md)).

---

## Core Theory

### 1. Types of Variable Relationships

| Variable 1 Type | Variable 2 Type | Analysis Method |
|-----------------|-----------------|-----------------|
| Numerical | Numerical | Correlation coefficient, scatter plot |
| Categorical | Categorical | Contingency table, chi-square |
| Numerical | Categorical | Point-biserial correlation, box plots |
| Ordinal | Ordinal | Spearman rank correlation |

---

### 2. Association Between Numerical Variables

#### 2.1 Scatter Plot - Visual Exploration

**Definition**: Graph with one variable on x-axis, another on y-axis. Each point represents one observation.

**Patterns to look for**:
1. **Positive linear**: Points trend upward-right
2. **Negative linear**: Points trend downward-right
3. **Non-linear**: Curved pattern (quadratic, exponential, etc.)
4. **No relationship**: Random cloud
5. **Outliers**: Points far from main pattern

#### Example 1: Scatter Plot Patterns

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Generate data with different patterns
x = np.linspace(0, 10, 100)

# 1. Strong positive linear
y_pos = 2*x + np.random.normal(0, 2, 100)

# 2. Strong negative linear
y_neg = -1.5*x + 15 + np.random.normal(0, 2, 100)

# 3. Non-linear (quadratic)
y_quad = (x - 5)**2 + np.random.normal(0, 3, 100)

# 4. No relationship
y_none = np.random.normal(5, 5, 100)

# Plot all patterns
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0, 0].scatter(x, y_pos, alpha=0.6, color='blue')
axes[0, 0].set_title('Positive Linear Relationship', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')
axes[0, 0].grid(alpha=0.3)

axes[0, 1].scatter(x, y_neg, alpha=0.6, color='red')
axes[0, 1].set_title('Negative Linear Relationship', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('X')
axes[0, 1].set_ylabel('Y')
axes[0, 1].grid(alpha=0.3)

axes[1, 0].scatter(x, y_quad, alpha=0.6, color='green')
axes[1, 0].set_title('Non-Linear Relationship (Quadratic)', fontsize=14, fontweight='bold')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('Y')
axes[1, 0].grid(alpha=0.3)

axes[1, 1].scatter(x, y_none, alpha=0.6, color='gray')
axes[1, 1].set_title('No Relationship', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('X')
axes[1, 1].set_ylabel('Y')
axes[1, 1].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 2.2 Covariance - Measuring Direction of Relationship

**Definition**: Measure of how two variables vary together.

**Population covariance**:
$$\sigma_{XY} = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu_X)(y_i - \mu_Y)$$

**Sample covariance**:
$$s_{XY} = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})$$

**Interpretation**:
- $s_{XY} > 0$: Positive relationship (when X increases, Y tends to increase)
- $s_{XY} < 0$: Negative relationship (when X increases, Y tends to decrease)
- $s_{XY} \approx 0$: Little or no linear relationship

**Problem with covariance**: Units depend on scales of X and Y. Can't compare covariances across different variable pairs.

#### Example 2: Computing Covariance

**Data**: Study hours (X) and Exam scores (Y) for 6 students

| Student | Hours (X) | Score (Y) |
|---------|-----------|-----------|
| 1       | 2         | 65        |
| 2       | 4         | 75        |
| 3       | 6         | 80        |
| 4       | 8         | 85        |
| 5       | 10        | 95        |
| 6       | 12        | 100       |

**Step 1**: Compute means
$$\bar{x} = \frac{2+4+6+8+10+12}{6} = 7$$
$$\bar{y} = \frac{65+75+80+85+95+100}{6} = 83.33$$

**Step 2**: Compute deviations

| Student | $x_i - \bar{x}$ | $y_i - \bar{y}$ | $(x_i - \bar{x})(y_i - \bar{y})$ |
|---------|-----------------|-----------------|----------------------------------|
| 1       | -5              | -18.33          | 91.65                            |
| 2       | -3              | -8.33           | 24.99                            |
| 3       | -1              | -3.33           | 3.33                             |
| 4       | 1               | 1.67            | 1.67                             |
| 5       | 3               | 11.67           | 35.01                            |
| 6       | 5               | 16.67           | 83.35                            |

**Step 3**: Sum and divide
$$s_{XY} = \frac{91.65 + 24.99 + 3.33 + 1.67 + 35.01 + 83.35}{6-1} = \frac{240}{5} = 48$$

**Interpretation**: Covariance is positive (48), indicating positive relationship between study hours and exam scores.

```python
hours = np.array([2, 4, 6, 8, 10, 12])
scores = np.array([65, 75, 80, 85, 95, 100])

# Manual calculation
mean_hours = np.mean(hours)
mean_scores = np.mean(scores)
dev_hours = hours - mean_hours
dev_scores = scores - mean_scores
cov_manual = np.sum(dev_hours * dev_scores) / (len(hours) - 1)

print(f"Manual covariance: {cov_manual:.2f}")

# Using NumPy
cov_matrix = np.cov(hours, scores, ddof=1)
cov_numpy = cov_matrix[0, 1]
print(f"NumPy covariance: {cov_numpy:.2f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(hours, scores, s=100, alpha=0.7, color='purple', edgecolor='black')
plt.plot(hours, scores, 'r--', alpha=0.5, label='Trend')
plt.xlabel('Study Hours', fontsize=12)
plt.ylabel('Exam Score', fontsize=12)
plt.title(f'Study Hours vs Exam Score (Cov = {cov_manual:.2f})', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)
plt.legend()
plt.show()
```

#### 2.3 Pearson Correlation Coefficient - Standardized Covariance

**Definition**: Standardized measure of linear relationship, ranges from -1 to +1.

**Formula**:
$$r = \frac{s_{XY}}{s_X s_Y} = \frac{\sum(x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum(x_i - \bar{x})^2}\sqrt{\sum(y_i - \bar{y})^2}}$$

where $s_X$ and $s_Y$ are standard deviations of X and Y.

**Interpretation**:
- $r = +1$: Perfect positive linear relationship
- $r = -1$: Perfect negative linear relationship
- $r = 0$: No linear relationship
- $0.7 < |r| < 1$: Strong correlation
- $0.3 < |r| < 0.7$: Moderate correlation
- $0 < |r| < 0.3$: Weak correlation

**Properties**:
- ✅ Dimensionless (no units)
- ✅ Always between -1 and +1
- ✅ Symmetric: $r_{XY} = r_{YX}$
- ❌ Only measures **linear** relationships (can miss non-linear patterns)
- ❌ Sensitive to outliers

#### Example 3: Pearson Correlation

**Continuing Example 2**:

**Step 1**: Compute standard deviations
$$s_X = \sqrt{\frac{\sum(x_i - \bar{x})^2}{n-1}} = \sqrt{\frac{(-5)^2 + (-3)^2 + (-1)^2 + 1^2 + 3^2 + 5^2}{5}} = \sqrt{\frac{70}{5}} = 3.74$$

$$s_Y = \sqrt{\frac{\sum(y_i - \bar{y})^2}{n-1}} = \sqrt{\frac{1680.02}{5}} = 18.33$$

**Step 2**: Compute correlation
$$r = \frac{s_{XY}}{s_X s_Y} = \frac{48}{3.74 \times 18.33} = \frac{48}{68.55} = 0.70$$

**Interpretation**: Correlation of 0.70 indicates **strong positive** linear relationship. As study hours increase, exam scores tend to increase.

```python
# Compute correlation
std_hours = np.std(hours, ddof=1)
std_scores = np.std(scores, ddof=1)
corr_manual = cov_numpy / (std_hours * std_scores)

print(f"Standard deviation (hours): {std_hours:.2f}")
print(f"Standard deviation (scores): {std_scores:.2f}")
print(f"Correlation coefficient (manual): {corr_manual:.2f}")

# Using NumPy
corr_numpy = np.corrcoef(hours, scores)[0, 1]
print(f"Correlation coefficient (NumPy): {corr_numpy:.2f}")

# Using Pandas
import pandas as pd
df = pd.DataFrame({'Hours': hours, 'Scores': scores})
corr_pandas = df['Hours'].corr(df['Scores'])
print(f"Correlation coefficient (Pandas): {corr_pandas:.2f}")
```

#### Example 4: Visualizing Correlation Strength

```python
# Generate datasets with different correlations
np.random.seed(42)
x = np.random.randn(100)

# Create y variables with different correlations
y_perfect = x  # r = 1
y_strong = x + 0.3*np.random.randn(100)  # r ≈ 0.96
y_moderate = x + np.random.randn(100)  # r ≈ 0.71
y_weak = x + 3*np.random.randn(100)  # r ≈ 0.32
y_none = np.random.randn(100)  # r ≈ 0

# Plot all
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

datasets = [
    (x, y_perfect, 'Perfect (r=1.00)'),
    (x, y_strong, f'Strong (r={np.corrcoef(x, y_strong)[0,1]:.2f})'),
    (x, y_moderate, f'Moderate (r={np.corrcoef(x, y_moderate)[0,1]:.2f})'),
    (x, y_weak, f'Weak (r={np.corrcoef(x, y_weak)[0,1]:.2f})'),
    (x, y_none, f'None (r={np.corrcoef(x, y_none)[0,1]:.2f})'),
    (x, -x, 'Perfect Negative (r=-1.00)')
]

for idx, (x_data, y_data, title) in enumerate(datasets):
    row = idx // 3
    col = idx % 3
    axes[row, col].scatter(x_data, y_data, alpha=0.6)
    axes[row, col].set_title(title, fontsize=14, fontweight='bold')
    axes[row, col].set_xlabel('X')
    axes[row, col].set_ylabel('Y')
    axes[row, col].grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 2.4 Spearman Rank Correlation

**When to use**:
- Ordinal data (rankings)
- Non-linear monotonic relationships
- Presence of outliers

**Method**: Rank the data, then compute Pearson correlation on ranks.

**Formula**:
$$r_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

where $d_i$ is difference in ranks.

**Advantage**: More robust than Pearson to outliers and works for monotonic (but not necessarily linear) relationships.

#### Example 5: Spearman vs Pearson

```python
from scipy.stats import spearmanr, pearsonr

# Data with outlier
x_outlier = np.array([1, 2, 3, 4, 5, 100])  # Outlier: 100
y_outlier = np.array([2, 4, 6, 8, 10, 12])

# Pearson (sensitive to outlier)
pearson_r, _ = pearsonr(x_outlier, y_outlier)
print(f"Pearson correlation: {pearson_r:.3f}")

# Spearman (robust to outlier)
spearman_r, _ = spearmanr(x_outlier, y_outlier)
print(f"Spearman correlation: {spearman_r:.3f}")

# Visualize
plt.figure(figsize=(12, 5))

plt.subplot(121)
plt.scatter(x_outlier, y_outlier, s=100, alpha=0.7, color='red', edgecolor='black')
plt.xlabel('X')
plt.ylabel('Y')
plt.title(f'Original Data\nPearson r = {pearson_r:.3f}', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

plt.subplot(122)
ranks_x = np.argsort(np.argsort(x_outlier)) + 1
ranks_y = np.argsort(np.argsort(y_outlier)) + 1
plt.scatter(ranks_x, ranks_y, s=100, alpha=0.7, color='green', edgecolor='black')
plt.xlabel('Rank of X')
plt.ylabel('Rank of Y')
plt.title(f'Ranked Data\nSpearman r = {spearman_r:.3f}', fontsize=14, fontweight='bold')
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### 3. Association Between Categorical Variables

#### 3.1 Contingency Tables with Relative Frequencies

**Review from Week 2**: Cross-tabulation shows frequency of each combination.

**New**: Compute **relative frequencies** to identify patterns.

**Types**:
1. **Row percentages**: Within each row, what proportion is in each column?
2. **Column percentages**: Within each column, what proportion is in each row?
3. **Cell percentages**: What proportion of total is each cell?

#### Example 6: Marketing Campaign Effectiveness

**Scenario**: Testing two ad campaigns (A and B) across two age groups.

**Frequency table**:

| Age Group / Campaign | A   | B   | Total |
|----------------------|-----|-----|-------|
| 18-35                | 60  | 40  | 100   |
| 36-55                | 30  | 70  | 100   |
| **Total**            | 90  | 110 | 200   |

**Row percentages** (within age group):

| Age Group / Campaign | A    | B    | Total |
|----------------------|------|------|-------|
| 18-35                | 60%  | 40%  | 100%  |
| 36-55                | 30%  | 70%  | 100%  |

**Interpretation**:
- Younger group (18-35) prefers Campaign A (60% vs 40%)
- Older group (36-55) prefers Campaign B (70% vs 30%)
- **Strong association** between age and campaign preference!

```python
# Create data
data = pd.DataFrame({
    'Age_Group': ['18-35']*100 + ['36-55']*100,
    'Campaign': ['A']*60 + ['B']*40 + ['A']*30 + ['B']*70
})

# Contingency table
cont_table = pd.crosstab(data['Age_Group'], data['Campaign'], margins=True)
print("Frequency Table:")
print(cont_table)
print()

# Row percentages
row_pct = pd.crosstab(data['Age_Group'], data['Campaign'], normalize='index') * 100
print("Row Percentages:")
print(row_pct.round(1))
print()

# Visualize
row_pct.plot(kind='bar', stacked=False, figsize=(10, 6), color=['skyblue', 'salmon'],
            edgecolor='black')
plt.title('Campaign Preference by Age Group', fontsize=16, fontweight='bold')
plt.xlabel('Age Group', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Campaign', fontsize=11)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

---

### 4. Association Between Numerical and Categorical Variables

#### 4.1 Visual Exploration - Box Plots by Group

**Method**: Create box plots for numerical variable, split by categorical variable.

#### Example 7: Exam Scores by Study Method

**Data**: Exam scores for students using two study methods (self-study vs tutoring).

```python
# Generate sample data
np.random.seed(42)
self_study = np.random.normal(75, 10, 50)
tutoring = np.random.normal(85, 8, 50)

scores_method = pd.DataFrame({
    'Score': np.concatenate([self_study, tutoring]),
    'Method': ['Self-Study']*50 + ['Tutoring']*50
})

# Summary statistics
print("Summary Statistics by Method:")
print(scores_method.groupby('Method')['Score'].describe())

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot
axes[0].boxplot([self_study, tutoring], labels=['Self-Study', 'Tutoring'],
                patch_artist=True,
                boxprops=dict(facecolor='lightblue'),
                medianprops=dict(color='red', linewidth=2))
axes[0].set_ylabel('Exam Score', fontsize=12)
axes[0].set_title('Scores by Study Method', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Histogram
axes[1].hist(self_study, bins=15, alpha=0.7, label='Self-Study', edgecolor='black', color='skyblue')
axes[1].hist(tutoring, bins=15, alpha=0.7, label='Tutoring', edgecolor='black', color='salmon')
axes[1].set_xlabel('Exam Score', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Score Distributions', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 4.2 Point-Biserial Correlation

**When to use**: Correlation between:
- Continuous variable (numerical)
- Binary variable (0/1 categorical)

**Formula**: Same as Pearson correlation, treating binary as 0/1.

$$r_{pb} = \frac{\bar{Y}_1 - \bar{Y}_0}{s_Y}\sqrt{\frac{n_0 n_1}{n(n-1)}}$$

where:
- $\bar{Y}_1$: mean of continuous variable for group 1
- $\bar{Y}_0$: mean for group 0
- $s_Y$: standard deviation of continuous variable
- $n_0, n_1$: sample sizes

#### Example 8: Point-Biserial Correlation

```python
# Convert method to binary (0=Self-Study, 1=Tutoring)
scores_method['Method_Binary'] = (scores_method['Method'] == 'Tutoring').astype(int)

# Compute point-biserial correlation
from scipy.stats import pointbiserialr

r_pb, p_value = pointbiserialr(scores_method['Method_Binary'], scores_method['Score'])
print(f"Point-biserial correlation: {r_pb:.3f}")
print(f"P-value: {p_value:.4f}")

# Also works with regular Pearson
r_pearson = scores_method['Method_Binary'].corr(scores_method['Score'])
print(f"Pearson correlation (same result): {r_pearson:.3f}")

# Interpretation
if r_pb > 0:
    print(f"\nPositive correlation: Tutoring group has higher scores on average")
else:
    print(f"\nNegative correlation: Self-study group has higher scores on average")
```

---

## Data Science Applications

### 1. Feature Selection - Correlation with Target

**Problem**: Which features are most predictive of target variable?

```python
from sklearn.datasets import load_boston

# Load Boston housing data
boston = load_boston()
df_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
df_boston['PRICE'] = boston.target

# Compute correlations with target
correlations = df_boston.corr()['PRICE'].sort_values(ascending=False)
print("Correlations with House Price:")
print(correlations)

# Visualize
plt.figure(figsize=(12, 8))
correlations[1:].plot(kind='barh', color=['green' if x > 0 else 'red' for x in correlations[1:]])
plt.xlabel('Correlation with Price', fontsize=12)
plt.title('Feature Correlations with House Price', fontsize=16, fontweight='bold')
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.show()

# Select top features
top_features = correlations[1:].abs().sort_values(ascending=False).head(5).index.tolist()
print(f"\nTop 5 predictive features: {top_features}")
```

### 2. Multicollinearity Detection

**Problem**: Highly correlated features cause problems in regression.

```python
# Correlation matrix
corr_matrix = df_boston.corr()

# Heatmap
plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5)
plt.title('Correlation Matrix Heatmap', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Find highly correlated pairs (|r| > 0.8)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((corr_matrix.columns[i],
                                   corr_matrix.columns[j],
                                   corr_matrix.iloc[i, j]))

print("\nHighly Correlated Feature Pairs (|r| > 0.8):")
for feat1, feat2, corr in high_corr_pairs:
    print(f"  {feat1} <-> {feat2}: r = {corr:.3f}")
```

### 3. A/B Test Analysis

**Problem**: Did new website design increase conversion rate?

```python
# Simulated A/B test data
np.random.seed(42)
ab_test = pd.DataFrame({
    'Variant': ['A']*1000 + ['B']*1000,
    'Time_on_Site': np.concatenate([
        np.random.normal(180, 40, 1000),  # Control: mean=180s
        np.random.normal(210, 45, 1000)   # Treatment: mean=210s
    ]),
    'Converted': np.concatenate([
        np.random.binomial(1, 0.10, 1000),  # Control: 10% conversion
        np.random.binomial(1, 0.15, 1000)   # Treatment: 15% conversion
    ])
})

# Conversion rate by variant
conversion_rates = ab_test.groupby('Variant')['Converted'].mean() * 100
print("Conversion Rates:")
print(conversion_rates)

# Time on site by variant
print("\nAverage Time on Site (seconds):")
print(ab_test.groupby('Variant')['Time_on_Site'].mean())

# Point-biserial: time vs conversion
ab_test['Variant_Binary'] = (ab_test['Variant'] == 'B').astype(int)
r_variant, _ = pointbiserialr(ab_test['Variant_Binary'], ab_test['Time_on_Site'])
print(f"\nCorrelation between Variant B and Time: r = {r_variant:.3f}")
```

---

## Common Pitfalls and Misconceptions

### Pitfall 1: Correlation ≠ Causation

❌ **Wrong**: "Ice cream sales correlate with drowning deaths, so ice cream causes drowning."

✅ **Right**: Both caused by third variable (hot weather). Correlation alone doesn't prove causation.

### Pitfall 2: Assuming Linearity

❌ **Wrong**: "r = 0, so no relationship."

✅ **Right**: Pearson r only measures **linear** relationships. Could be strong non-linear relationship!

**Example**:
```python
x = np.linspace(-3, 3, 100)
y = x**2  # Perfect quadratic relationship

r = np.corrcoef(x, y)[0, 1]
print(f"Correlation: {r:.3f}")  # Close to 0!

plt.scatter(x, y)
plt.title(f'Perfect Quadratic Relationship, but r={r:.2f}')
plt.show()
```

### Pitfall 3: Ignoring Outliers

❌ **Wrong**: Compute correlation without checking for outliers.

✅ **Right**: Always visualize with scatter plot first.

### Pitfall 4: Extrapolation

❌ **Wrong**: "Study hours (1-10) correlate with scores. So 100 hours → 100% score!"

✅ **Right**: Correlation valid only within observed range. Don't extrapolate beyond data.

### Pitfall 5: Confusing Association with Independence

❌ **Wrong**: "Zero correlation means variables are independent."

✅ **Right**: Independence → zero correlation, but zero correlation doesn't → independence (for non-linear relationships).

---

## Self-Assessment and Active Recall

### Concept Check Questions

1. **Explain**: Why is correlation coefficient better than covariance for comparing relationships?

2. **Interpret**: If r = -0.85, what does this tell you about the relationship?

3. **Choose**: When should you use Spearman instead of Pearson correlation?

4. **True/False**: "If r = 0.3, then 30% of Y is explained by X." Explain.

5. **Identify**: Can two variables have r = 0 but still be strongly related? Give example.

### Practice Problems

#### Basic Level

1. Compute covariance and correlation for: X = {1, 2, 3, 4, 5}, Y = {2, 4, 6, 8, 10}

2. Given r = 0.6 between height and weight, interpret the relationship.

3. Create scatter plot for X = {1, 2, 3}, Y = {3, 2, 1}. Predict sign of r.

#### Intermediate Level

4. Dataset has r = 0.4 between X and Y. Add outlier (100, 100). How does r change?

5. Given contingency table, compute row percentages and interpret association.

6. Compute point-biserial correlation between gender (M/F) and salary.

#### Advanced Level

7. Prove that $-1 \leq r \leq 1$ for any dataset.

8. Show that adding constant to X or Y doesn't change r.

9. Create dataset where Pearson r ≈ 0 but Spearman r ≈ 1.

---

## Quick Reference Summary

### Key Formulas

**Covariance**:
$$s_{XY} = \frac{1}{n-1}\sum(x_i - \bar{x})(y_i - \bar{y})$$

**Pearson Correlation**:
$$r = \frac{s_{XY}}{s_X s_Y}$$

**Spearman Correlation**:
$$r_s = 1 - \frac{6\sum d_i^2}{n(n^2-1)}$$

### Correlation Interpretation Guide

| |r| Value | Interpretation |
|-----------|----------------|
| 0.9 - 1.0 | Very strong |
| 0.7 - 0.9 | Strong |
| 0.5 - 0.7 | Moderate |
| 0.3 - 0.5 | Weak |
| 0.0 - 0.3 | Very weak or none |

### Python Templates

```python
# Scatter plot
plt.scatter(x, y)

# Correlation
np.corrcoef(x, y)[0, 1]
df['x'].corr(df['y'])

# Covariance
np.cov(x, y, ddof=1)[0, 1]

# Spearman
from scipy.stats import spearmanr
spearmanr(x, y)

# Correlation matrix
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Point-biserial
from scipy.stats import pointbiserialr
pointbiserialr(binary_var, continuous_var)
```

### Top 3 Things to Remember

1. **Correlation ≠ Causation**: Association doesn't prove cause-and-effect
2. **Visualize first**: Always plot before computing correlations (catches non-linearity, outliers)
3. **Pearson for linear only**: Use Spearman for ordinal data or non-linear monotonic relationships

---

## Further Resources

### Documentation
- Pandas: `.corr()` method
- NumPy: `corrcoef()`, `cov()`
- SciPy: `pearsonr()`, `spearmanr()`, `pointbiserialr()`
- Seaborn: Correlation heatmaps

### Books
- Freedman, Pisani, Purves, "Statistics" - Chapter 5
- Moore & McCabe, "Introduction to the Practice of Statistics"

### Practice
- Kaggle: Titanic (correlation between features and survival)
- Real datasets: UCI ML Repository

### Review Schedule
- **After 1 day**: Compute correlations by hand for small dataset
- **After 3 days**: Create scatter plots and correlation matrices
- **After 1 week**: Analyze real dataset, identify top correlated features
- **After 2 weeks**: Build feature selection pipeline using correlations

---

**Related Notes**:
- Previous: [week-03-numerical-data-visualization.md](week-03-numerical-data-visualization.md)
- Next: [week-05-dispersion-variability.md](week-05-dispersion-variability.md)
- Application: Feature selection, multicollinearity detection, EDA

**Last Updated**: 2025-11-22
**Status**: Complete and comprehensive ✅
