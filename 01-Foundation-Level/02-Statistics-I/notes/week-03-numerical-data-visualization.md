# Week 3: Describing Numerical Data - Central Tendency and Dispersion

---
Date: 2025-11-22
Course: BSMA1002 - Statistics for Data Science I
Level: Foundation
Week: 3 of 12
Source: IIT Madras Statistics I Week 3
Topic Area: Descriptive Statistics - Numerical Data
Tags: #BSMA1002 #NumericalData #CentralTendency #Dispersion #Mean #Median #StandardDeviation #IQR #Week3 #Foundation
---

## Overview (BLUF)

**One-sentence definition**: Numerical data analysis summarizes datasets using measures of central tendency (where the "center" is) and measures of dispersion (how spread out values are), providing complete statistical snapshots of data distributions.

**Why it matters**: Three numbers—mean, standard deviation, and median—tell you more about a dataset than looking at thousands of individual values. These statistics are the foundation of hypothesis testing, quality control, risk assessment, and virtually every data science technique. Before training ML models, you must understand your data's center and spread.

**When to use**: Comparing datasets (which has higher average?), detecting outliers (values >3 standard deviations), quality control (is process within tolerance?), feature engineering (standardizing features), understanding distributions (normal? skewed?), summarizing survey results.

**Prerequisites**: Basic arithmetic, understanding of categorical data ([week-02](week-02-categorical-data-analysis.md)), knowledge of data types ([week-01](week-01-data-types-scales.md)).

---

## Core Theory

### 1. Measures of Central Tendency

**Purpose**: Answer "What's a typical value?"

Three main measures: Mean, Median, Mode

#### 1.1 Mean (Arithmetic Average)

**Definition**: Sum of all values divided by count.

$$\bar{x} = \frac{1}{n}\sum_{i=1}^{n}x_i = \frac{x_1 + x_2 + \cdots + x_n}{n}$$

**Properties**:
- ✅ Uses all data points
- ✅ Unique (only one mean)
- ✅ Sum of deviations from mean equals zero: $\sum(x_i - \bar{x}) = 0$
- ❌ Sensitive to outliers (extreme values pull mean)
- ❌ Not resistant (a single outlier can drastically change it)

**When to use**: When data is symmetric without extreme outliers.

#### Example 1: Computing Mean

**Scenario**: Exam scores for 10 students

Data: $\{85, 78, 92, 88, 76, 95, 82, 90, 87, 84\}$

$$\bar{x} = \frac{85 + 78 + 92 + 88 + 76 + 95 + 82 + 90 + 87 + 84}{10} = \frac{857}{10} = 85.7$$

**Interpretation**: Average score is 85.7 points.

**With outlier**: Add one more score: 20 (someone who didn't study)

Data: $\{85, 78, 92, 88, 76, 95, 82, 90, 87, 84, 20\}$

$$\bar{x} = \frac{857 + 20}{11} = \frac{877}{11} = 79.7$$

**Impact**: Single outlier reduced mean by 6 points! This shows mean's sensitivity.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Original data
scores = np.array([85, 78, 92, 88, 76, 95, 82, 90, 87, 84])
mean_original = np.mean(scores)
print(f"Original mean: {mean_original:.1f}")

# With outlier
scores_outlier = np.append(scores, 20)
mean_outlier = np.mean(scores_outlier)
print(f"Mean with outlier: {mean_outlier:.1f}")
print(f"Difference: {mean_original - mean_outlier:.1f} points")

# Visualize
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(scores, bins=8, edgecolor='black', alpha=0.7, color='skyblue')
axes[0].axvline(mean_original, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_original:.1f}')
axes[0].set_title('Original Scores', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Score')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

axes[1].hist(scores_outlier, bins=10, edgecolor='black', alpha=0.7, color='salmon')
axes[1].axvline(mean_outlier, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_outlier:.1f}')
axes[1].set_title('Scores with Outlier (20)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Score')
axes[1].set_ylabel('Frequency')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

#### 1.2 Median

**Definition**: Middle value when data is sorted.

**Algorithm**:
1. Sort data in ascending order
2. If $n$ is odd: median = middle value
3. If $n$ is even: median = average of two middle values

**Formula** (for sorted data $x_1 \leq x_2 \leq \cdots \leq x_n$):
$$\text{Median} = \begin{cases}
x_{(n+1)/2} & \text{if } n \text{ is odd} \\[8pt]
\frac{x_{n/2} + x_{(n/2)+1}}{2} & \text{if } n \text{ is even}
\end{cases}$$

**Properties**:
- ✅ **Robust** to outliers (not affected by extreme values)
- ✅ 50% of data below, 50% above (50th percentile)
- ✅ Better for skewed distributions
- ❌ Doesn't use all data (only positions matter, not exact values)

#### Example 2: Computing Median

**Original scores** (sorted): $\{76, 78, 82, 84, 85, 87, 88, 90, 92, 95\}$

$n = 10$ (even), so median = average of 5th and 6th values:
$$\text{Median} = \frac{85 + 87}{2} = 86$$

**With outlier** (sorted): $\{20, 76, 78, 82, 84, 85, 87, 88, 90, 92, 95\}$

$n = 11$ (odd), so median = 6th value:
$$\text{Median} = 85$$

**Key insight**: Median barely changed (86 → 85) despite outlier! This is robustness.

```python
# Median calculation
median_original = np.median(scores)
median_outlier = np.median(scores_outlier)

print(f"Original median: {median_original:.1f}")
print(f"Median with outlier: {median_outlier:.1f}")
print(f"Difference: {abs(median_original - median_outlier):.1f} points")

# Compare mean vs median
comparison = pd.DataFrame({
    'Statistic': ['Mean', 'Median'],
    'Original': [mean_original, median_original],
    'With Outlier': [mean_outlier, median_outlier],
    'Change': [mean_original - mean_outlier, median_original - median_outlier]
})
print("\nComparison:")
print(comparison)
```

#### 1.3 Mode

**Definition**: Most frequently occurring value.

**Properties**:
- Can have multiple modes (bimodal, multimodal)
- May not exist (all values unique)
- Works for both numerical and categorical data

**Less common** for numerical data (unless discrete with repeated values).

#### 1.4 Mean vs Median - When to Use Which?

| Condition | Use Mean | Use Median |
|-----------|----------|------------|
| Symmetric distribution, no outliers | ✅ | Either |
| Skewed distribution | ❌ | ✅ |
| Contains outliers | ❌ | ✅ |
| Need mathematical properties | ✅ | ❌ |
| Reporting "typical" value to public | ❌ | ✅ |

**Rule of thumb**:
- If mean ≈ median → symmetric distribution → use mean
- If mean > median → right-skewed → use median
- If mean < median → left-skewed → use median

---

### 2. Measures of Dispersion (Spread)

**Purpose**: Answer "How spread out are values?"

Two datasets can have same mean but very different spreads:
- Dataset A: {98, 99, 100, 101, 102} → Mean = 100, tightly clustered
- Dataset B: {0, 50, 100, 150, 200} → Mean = 100, widely spread

We need measures of **variability**.

#### 2.1 Range

**Definition**: Difference between maximum and minimum.

$$\text{Range} = x_{\max} - x_{\min}$$

**Properties**:
- ✅ Easy to compute and understand
- ❌ Uses only 2 values (ignores everything in between)
- ❌ Extremely sensitive to outliers

#### Example 3: Range Limitations

**Data**: $\{10, 11, 12, 13, 14, 15, 16, 17, 18, 100\}$

Range = $100 - 10 = 90$

But most values are between 10-18! Range doesn't reflect typical spread.

#### 2.2 Variance

**Definition**: Average squared deviation from the mean.

**Population variance** ($\sigma^2$):
$$\sigma^2 = \frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2$$

**Sample variance** ($s^2$):
$$s^2 = \frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2$$

**Why $n-1$ instead of $n$?** (Bessel's correction)
- Makes sample variance an **unbiased estimator** of population variance
- Corrects for fact that sample mean $\bar{x}$ is itself estimated from data
- For large $n$, difference is negligible

**Properties**:
- ✅ Uses all data points
- ✅ Mathematical properties (basis for many statistical tests)
- ❌ Units are squared (hard to interpret)
- ❌ Sensitive to outliers (squaring amplifies extreme values)

#### Example 4: Computing Variance

**Data**: $\{4, 8, 6, 5, 3, 7\}$

**Step 1**: Compute mean
$$\bar{x} = \frac{4 + 8 + 6 + 5 + 3 + 7}{6} = \frac{33}{6} = 5.5$$

**Step 2**: Compute deviations
$$\begin{align*}
x_1 - \bar{x} &= 4 - 5.5 = -1.5 \\
x_2 - \bar{x} &= 8 - 5.5 = 2.5 \\
x_3 - \bar{x} &= 6 - 5.5 = 0.5 \\
x_4 - \bar{x} &= 5 - 5.5 = -0.5 \\
x_5 - \bar{x} &= 3 - 5.5 = -2.5 \\
x_6 - \bar{x} &= 7 - 5.5 = 1.5
\end{align*}$$

**Step 3**: Square deviations
$$(-1.5)^2 = 2.25, \quad (2.5)^2 = 6.25, \quad (0.5)^2 = 0.25$$
$$(0.5)^2 = 0.25, \quad (-2.5)^2 = 6.25, \quad (1.5)^2 = 2.25$$

**Step 4**: Sum squared deviations
$$\sum(x_i - \bar{x})^2 = 2.25 + 6.25 + 0.25 + 0.25 + 6.25 + 2.25 = 17.5$$

**Step 5**: Divide by $n-1$
$$s^2 = \frac{17.5}{6-1} = \frac{17.5}{5} = 3.5$$

```python
data_var = np.array([4, 8, 6, 5, 3, 7])
mean_var = np.mean(data_var)
deviations = data_var - mean_var
squared_devs = deviations**2

print(f"Data: {data_var}")
print(f"Mean: {mean_var:.2f}")
print(f"Deviations: {deviations}")
print(f"Squared deviations: {squared_devs}")
print(f"Sum of squared deviations: {squared_devs.sum():.2f}")
print(f"Variance (sample): {np.var(data_var, ddof=1):.2f}")  # ddof=1 for sample variance
```

#### 2.3 Standard Deviation

**Definition**: Square root of variance (brings units back to original scale).

**Population standard deviation** ($\sigma$):
$$\sigma = \sqrt{\sigma^2} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(x_i - \mu)^2}$$

**Sample standard deviation** ($s$):
$$s = \sqrt{s^2} = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(x_i - \bar{x})^2}$$

**Properties**:
- ✅ Same units as original data (interpretable!)
- ✅ Average distance of data points from mean
- ✅ Most commonly used measure of spread
- ❌ Still sensitive to outliers

**Rule of thumb** (for approximately normal data):
- About **68%** of data within 1 standard deviation of mean ($\bar{x} \pm s$)
- About **95%** within 2 standard deviations ($\bar{x} \pm 2s$)
- About **99.7%** within 3 standard deviations ($\bar{x} \pm 3s$)

#### Example 5: Standard Deviation

From Example 4: $s^2 = 3.5$

$$s = \sqrt{3.5} \approx 1.87$$

**Interpretation**: On average, values deviate from mean (5.5) by about 1.87 units.

**Outlier detection**: Values beyond $\bar{x} \pm 3s$ are potential outliers.
$$\text{Lower bound} = 5.5 - 3(1.87) = -0.11$$
$$\text{Upper bound} = 5.5 + 3(1.87) = 11.11$$

All our values {4, 8, 6, 5, 3, 7} are within this range → no outliers.

```python
std_dev = np.std(data_var, ddof=1)
print(f"Standard deviation: {std_dev:.2f}")

# 68-95-99.7 rule
lower_1sd = mean_var - std_dev
upper_1sd = mean_var + std_dev
lower_2sd = mean_var - 2*std_dev
upper_2sd = mean_var + 2*std_dev
lower_3sd = mean_var - 3*std_dev
upper_3sd = mean_var + 3*std_dev

print(f"\n68% range: [{lower_1sd:.2f}, {upper_1sd:.2f}]")
print(f"95% range: [{lower_2sd:.2f}, {upper_2sd:.2f}]")
print(f"99.7% range: [{lower_3sd:.2f}, {upper_3sd:.2f}]")

# Count values in each range
within_1sd = np.sum((data_var >= lower_1sd) & (data_var <= upper_1sd))
within_2sd = np.sum((data_var >= lower_2sd) & (data_var <= upper_2sd))
within_3sd = np.sum((data_var >= lower_3sd) & (data_var <= upper_3sd))

print(f"\nActual percentages:")
print(f"Within 1 SD: {100*within_1sd/len(data_var):.1f}%")
print(f"Within 2 SD: {100*within_2sd/len(data_var):.1f}%")
print(f"Within 3 SD: {100*within_3sd/len(data_var):.1f}%")
```

#### 2.4 Quartiles and Percentiles

**Percentile**: Value below which a given percentage of data falls.

**Quartiles**: Special percentiles that divide data into four equal parts.

- **Q1** (First quartile, 25th percentile): 25% of data below this
- **Q2** (Second quartile, 50th percentile): **Median**
- **Q3** (Third quartile, 75th percentile): 75% of data below this

**Interquartile Range (IQR)**: Range of middle 50% of data.
$$\text{IQR} = Q3 - Q1$$

**Properties of IQR**:
- ✅ **Robust** to outliers (uses middle 50%, ignores extremes)
- ✅ Better than range for skewed data
- ✅ Basis for outlier detection

**Outlier detection rule** (Tukey's fences):
- Lower fence: $Q1 - 1.5 \times \text{IQR}$
- Upper fence: $Q3 + 1.5 \times \text{IQR}$
- Values outside fences = potential outliers

#### Example 6: Quartiles and IQR

**Data** (sorted): $\{10, 15, 18, 20, 22, 25, 28, 30, 35, 40, 50, 60\}$

$n = 12$

**Q1** (25th percentile): Position = $0.25 \times (12 + 1) = 3.25$ → between 3rd and 4th values
$$Q1 = 18 + 0.25(20 - 18) = 18 + 0.5 = 18.5$$

**Q2** (Median): Average of 6th and 7th values
$$Q2 = \frac{25 + 28}{2} = 26.5$$

**Q3** (75th percentile): Position = $0.75 \times (12 + 1) = 9.75$ → between 9th and 10th
$$Q3 = 35 + 0.75(40 - 35) = 35 + 3.75 = 38.75$$

**IQR**:
$$\text{IQR} = Q3 - Q1 = 38.75 - 18.5 = 20.25$$

**Outlier fences**:
$$\text{Lower fence} = 18.5 - 1.5(20.25) = 18.5 - 30.375 = -11.875$$
$$\text{Upper fence} = 38.75 + 1.5(20.25) = 38.75 + 30.375 = 69.125$$

Value 60 is within fences → not an outlier by this rule.

```python
data_iqr = np.array([10, 15, 18, 20, 22, 25, 28, 30, 35, 40, 50, 60])

Q1 = np.percentile(data_iqr, 25)
Q2 = np.percentile(data_iqr, 50)  # Median
Q3 = np.percentile(data_iqr, 75)
IQR = Q3 - Q1

lower_fence = Q1 - 1.5 * IQR
upper_fence = Q3 + 1.5 * IQR

print(f"Q1 (25th percentile): {Q1}")
print(f"Q2 (Median): {Q2}")
print(f"Q3 (75th percentile): {Q3}")
print(f"IQR: {IQR}")
print(f"\nOutlier fences:")
print(f"Lower: {lower_fence:.2f}")
print(f"Upper: {upper_fence:.2f}")

# Identify outliers
outliers_iqr = data_iqr[(data_iqr < lower_fence) | (data_iqr > upper_fence)]
print(f"\nOutliers: {outliers_iqr}")
```

#### 2.5 Five-Number Summary and Box Plot

**Five-number summary**: {Min, Q1, Median, Q3, Max}

**Box plot** (box-and-whisker plot): Visual representation of five-number summary.

**Components**:
- **Box**: From Q1 to Q3 (contains middle 50% of data)
- **Line in box**: Median
- **Whiskers**: Extend to smallest/largest values within 1.5×IQR from box
- **Points beyond whiskers**: Outliers

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Box plot
axes[0].boxplot(data_iqr, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', edgecolor='black'),
                medianprops=dict(color='red', linewidth=2),
                whiskerprops=dict(color='black', linewidth=1.5),
                capprops=dict(color='black', linewidth=1.5))
axes[0].set_ylabel('Value', fontsize=12)
axes[0].set_title('Box Plot', fontsize=14, fontweight='bold')
axes[0].grid(axis='y', alpha=0.3)

# Annotate five-number summary
axes[0].text(1.15, np.min(data_iqr), f'Min: {np.min(data_iqr)}', fontsize=10)
axes[0].text(1.15, Q1, f'Q1: {Q1}', fontsize=10)
axes[0].text(1.15, Q2, f'Median: {Q2}', fontsize=10, color='red', fontweight='bold')
axes[0].text(1.15, Q3, f'Q3: {Q3}', fontsize=10)
axes[0].text(1.15, np.max(data_iqr), f'Max: {np.max(data_iqr)}', fontsize=10)

# Histogram with quartiles marked
axes[1].hist(data_iqr, bins=8, edgecolor='black', alpha=0.7, color='skyblue')
axes[1].axvline(Q1, color='orange', linestyle='--', linewidth=2, label=f'Q1: {Q1}')
axes[1].axvline(Q2, color='red', linestyle='--', linewidth=2, label=f'Median: {Q2}')
axes[1].axvline(Q3, color='green', linestyle='--', linewidth=2, label=f'Q3: {Q3}')
axes[1].set_xlabel('Value', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].set_title('Histogram with Quartiles', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Five-number summary
five_num_summary = {
    'Min': np.min(data_iqr),
    'Q1': Q1,
    'Median': Q2,
    'Q3': Q3,
    'Max': np.max(data_iqr)
}
print("\nFive-Number Summary:")
for key, value in five_num_summary.items():
    print(f"{key}: {value}")
```

---

### 3. Comparing Distributions

#### Example 7: Comprehensive Comparison

**Scenario**: Comparing test scores from two classes.

**Class A**: {70, 72, 75, 78, 80, 82, 85, 88, 90, 92}
**Class B**: {50, 60, 70, 75, 80, 85, 90, 95, 100, 100}

```python
class_A = np.array([70, 72, 75, 78, 80, 82, 85, 88, 90, 92])
class_B = np.array([50, 60, 70, 75, 80, 85, 90, 95, 100, 100])

# Compute statistics
stats_comparison = pd.DataFrame({
    'Class A': [
        np.mean(class_A),
        np.median(class_A),
        np.std(class_A, ddof=1),
        np.percentile(class_A, 25),
        np.percentile(class_A, 75),
        np.percentile(class_A, 75) - np.percentile(class_A, 25),
        np.min(class_A),
        np.max(class_A)
    ],
    'Class B': [
        np.mean(class_B),
        np.median(class_B),
        np.std(class_B, ddof=1),
        np.percentile(class_B, 25),
        np.percentile(class_B, 75),
        np.percentile(class_B, 75) - np.percentile(class_B, 25),
        np.min(class_B),
        np.max(class_B)
    ]
}, index=['Mean', 'Median', 'Std Dev', 'Q1', 'Q3', 'IQR', 'Min', 'Max'])

print("Statistical Comparison:")
print(stats_comparison.round(2))

# Visualizations
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histograms
axes[0, 0].hist(class_A, bins=8, alpha=0.7, label='Class A', edgecolor='black', color='skyblue')
axes[0, 0].hist(class_B, bins=8, alpha=0.7, label='Class B', edgecolor='black', color='salmon')
axes[0, 0].axvline(np.mean(class_A), color='blue', linestyle='--', linewidth=2)
axes[0, 0].axvline(np.mean(class_B), color='red', linestyle='--', linewidth=2)
axes[0, 0].set_xlabel('Score')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Score Distributions', fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(axis='y', alpha=0.3)

# Box plots
axes[0, 1].boxplot([class_A, class_B], labels=['Class A', 'Class B'], patch_artist=True,
                   boxprops=dict(facecolor='lightblue'),
                   medianprops=dict(color='red', linewidth=2))
axes[0, 1].set_ylabel('Score')
axes[0, 1].set_title('Box Plot Comparison', fontweight='bold')
axes[0, 1].grid(axis='y', alpha=0.3)

# Mean comparison
axes[1, 0].bar(['Class A', 'Class B'], [np.mean(class_A), np.mean(class_B)],
              color=['skyblue', 'salmon'], edgecolor='black')
axes[1, 0].set_ylabel('Mean Score')
axes[1, 0].set_title('Mean Comparison', fontweight='bold')
axes[1, 0].grid(axis='y', alpha=0.3)

# Std Dev comparison
axes[1, 1].bar(['Class A', 'Class B'],
              [np.std(class_A, ddof=1), np.std(class_B, ddof=1)],
              color=['lightgreen', 'gold'], edgecolor='black')
axes[1, 1].set_ylabel('Standard Deviation')
axes[1, 1].set_title('Variability Comparison', fontweight='bold')
axes[1, 1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()

# Interpretation
print("\nInterpretation:")
print(f"- Both classes have same mean ({np.mean(class_A):.1f})")
print(f"- Class B has higher variability (SD: {np.std(class_B, ddof=1):.1f} vs {np.std(class_A, ddof=1):.1f})")
print(f"- Class B has wider range ({np.max(class_B) - np.min(class_B)} vs {np.max(class_A) - np.min(class_A)})")
print(f"- Class A more consistent performance")
```

---

## Data Science Applications

### 1. Feature Scaling and Standardization

**Problem**: ML algorithms perform better when features have similar scales.

**Z-score standardization**:
$$z = \frac{x - \bar{x}}{s}$$

Transforms data to have mean = 0, standard deviation = 1.

```python
from sklearn.preprocessing import StandardScaler

# Sample data
heights = np.array([150, 160, 170, 180, 190])  # cm
weights = np.array([50, 60, 70, 80, 90])  # kg

# Original scales very different
print(f"Heights: mean={np.mean(heights):.1f}, std={np.std(heights, ddof=1):.1f}")
print(f"Weights: mean={np.mean(weights):.1f}, std={np.std(weights, ddof=1):.1f}")

# Standardize
scaler = StandardScaler()
data = np.column_stack([heights, weights])
data_standardized = scaler.fit_transform(data)

print("\nAfter standardization:")
print(f"Heights: mean={np.mean(data_standardized[:, 0]):.1e}, std={np.std(data_standardized[:, 0], ddof=1):.1f}")
print(f"Weights: mean={np.mean(data_standardized[:, 1]):.1e}, std={np.std(data_standardized[:, 1], ddof=1):.1f}")
```

### 2. Outlier Detection in Quality Control

**Problem**: Detect defective products based on measurements.

```python
# Manufacturing data (product weights in grams)
product_weights = np.array([100.2, 100.5, 100.1, 100.4, 100.3, 100.6, 100.2, 105.0, 100.4, 100.1])

mean_weight = np.mean(product_weights)
std_weight = np.std(product_weights, ddof=1)

# 3-sigma rule
lower_limit = mean_weight - 3*std_weight
upper_limit = mean_weight + 3*std_weight

outliers_3sigma = product_weights[(product_weights < lower_limit) | (product_weights > upper_limit)]

print(f"Mean: {mean_weight:.2f} g")
print(f"Std Dev: {std_weight:.2f} g")
print(f"Control limits: [{lower_limit:.2f}, {upper_limit:.2f}]")
print(f"Outliers (3-sigma): {outliers_3sigma}")

# IQR method
Q1 = np.percentile(product_weights, 25)
Q3 = np.percentile(product_weights, 75)
IQR = Q3 - Q1
lower_fence = Q1 - 1.5*IQR
upper_fence = Q3 + 1.5*IQR

outliers_iqr = product_weights[(product_weights < lower_fence) | (product_weights > upper_fence)]
print(f"\nOutliers (IQR method): {outliers_iqr}")
```

### 3. Comparing Model Performance

**Problem**: Which model has better predictions?

```python
# Prediction errors from two models
model_A_errors = np.abs(np.random.normal(0, 2, 100))  # Lower variance
model_B_errors = np.abs(np.random.normal(0, 5, 100))  # Higher variance

print("Model A:")
print(f"  Mean error: {np.mean(model_A_errors):.2f}")
print(f"  Std error: {np.std(model_A_errors, ddof=1):.2f}")
print(f"  Median error: {np.median(model_A_errors):.2f}")

print("\nModel B:")
print(f"  Mean error: {np.mean(model_B_errors):.2f}")
print(f"  Std error: {np.std(model_B_errors, ddof=1):.2f}")
print(f"  Median error: {np.median(model_B_errors):.2f}")

print("\n→ Model A has lower error and more consistent predictions")
```

---

## Common Pitfalls and Misconceptions

### Pitfall 1: Using Mean for Skewed Data

❌ **Wrong**: "Median income is $50k but mean is $100k, so average is $100k"

✅ **Right**: Use median for skewed data (income is right-skewed due to high earners).

### Pitfall 2: Forgetting Units

❌ **Wrong**: "Variance is 25"

✅ **Right**: "Variance is 25 cm²" (or use standard deviation: 5 cm)

### Pitfall 3: Sample vs Population Formulas

❌ **Wrong**: Use $n$ when computing sample variance

✅ **Right**: Use $n-1$ for sample variance (Bessel's correction)

### Pitfall 4: Assuming Normal Distribution

❌ **Wrong**: "68% of data is within 1 SD" (for any distribution)

✅ **Right**: 68-95-99.7 rule applies only to approximately normal distributions.

### Pitfall 5: Outliers Always Bad

❌ **Wrong**: Remove all outliers automatically

✅ **Right**: Investigate outliers—they might be data errors OR important insights (e.g., fraud detection).

---

## Self-Assessment and Active Recall

### Concept Check Questions

1. **Explain**: Why is median more robust than mean to outliers?

2. **Calculate**: For data {2, 4, 6, 8, 10}, what is the variance and standard deviation?

3. **Interpret**: Dataset has mean = 50, median = 40. What does this tell you about distribution shape?

4. **Choose**: When should you use IQR instead of standard deviation for measuring spread?

5. **True/False**: "For any dataset, about 95% of values are within 2 standard deviations of the mean." Explain.

### Practice Problems

#### Basic Level

1. Compute mean, median, and mode for: {5, 7, 3, 7, 9, 7, 5, 8}

2. Find range, variance, and standard deviation for: {10, 20, 30, 40, 50}

3. Given Q1=25, Q3=75, identify outliers for value 150.

#### Intermediate Level

4. Dataset A: {10, 20, 30}, Dataset B: {15, 20, 25}. Both have mean=20. Compare standard deviations. Which is more variable?

5. Compute five-number summary for: {12, 15, 18, 20, 22, 25, 30, 35, 40, 50}

6. Given mean=100, SD=15, what percentage of data (assuming normality) is between 85 and 115?

#### Advanced Level

7. Prove that $\sum(x_i - \bar{x}) = 0$ (deviations from mean sum to zero).

8. Show that variance can also be computed as: $s^2 = \frac{\sum x_i^2}{n-1} - \frac{n\bar{x}^2}{n-1}$

9. Create synthetic dataset where mean ≠ median, identify which measure better represents "typical" value.

---

## Quick Reference Summary

### Key Formulas

**Mean**: $\bar{x} = \frac{1}{n}\sum x_i$

**Median**: Middle value (or average of two middle values)

**Variance**: $s^2 = \frac{1}{n-1}\sum(x_i - \bar{x})^2$

**Standard Deviation**: $s = \sqrt{s^2}$

**IQR**: $IQR = Q3 - Q1$

**Outlier fences**: $Q1 - 1.5 \times IQR$ and $Q3 + 1.5 \times IQR$

### Decision Tree: Which Statistic to Use?

```
Is data skewed or has outliers?
    ├── YES → Use Median and IQR
    └── NO  → Use Mean and Standard Deviation
```

### Python Templates

```python
# Central tendency
mean = np.mean(data)
median = np.median(data)
mode = data.mode()[0]  # For pandas Series

# Dispersion
range_val = np.max(data) - np.min(data)
variance = np.var(data, ddof=1)  # Sample variance
std_dev = np.std(data, ddof=1)   # Sample std dev

# Quartiles
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

# Five-number summary
np.min(data), Q1, np.median(data), Q3, np.max(data)

# Box plot
plt.boxplot(data)

# Outlier detection (IQR method)
lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR
outliers = data[(data < lower) | (data > upper)]
```

### Top 3 Things to Remember

1. **Mean for symmetric, median for skewed**: Choose measure based on distribution shape and outliers
2. **Standard deviation is interpretable**: Unlike variance, SD has same units as data
3. **IQR is robust**: Best measure of spread when data has outliers or is skewed

---

## Further Resources

### Documentation
- NumPy: Statistical functions (`mean`, `median`, `std`, `percentile`)
- Pandas: `describe()` method for quick summary
- SciPy: `scipy.stats` for advanced statistics

### Books
- Freedman, Pisani, Purves, "Statistics" - Chapters 3-5
- DeGroot & Schervish, "Probability and Statistics"

### Practice
- Kaggle: Exploratory Data Analysis notebooks
- Real datasets: UCI ML Repository, Titanic, Boston Housing

### Review Schedule
- **After 1 day**: Compute mean, median, SD by hand for small dataset
- **After 3 days**: Create box plots and interpret five-number summaries
- **After 1 week**: Analyze real dataset, compare distributions
- **After 2 weeks**: Implement outlier detection and feature scaling

---

**Related Notes**:
- Previous: [week-02-categorical-data-analysis.md](week-02-categorical-data-analysis.md)
- Next: [week-04-central-tendency-measures.md](week-04-central-tendency-measures.md)
- Application: Feature engineering, outlier detection, model evaluation

**Last Updated**: 2025-11-22
**Status**: Complete and comprehensive ✅
