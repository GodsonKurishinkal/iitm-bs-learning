# Week 2: Describing Categorical Data - Frequencies and Visualizations

---
Date: 2025-11-22
Course: BSMA1002 - Statistics for Data Science I
Level: Foundation
Week: 2 of 12
Source: IIT Madras Statistics I Week 2
Topic Area: Descriptive Statistics - Categorical Data
Tags: #BSMA1002 #CategoricalData #FrequencyDistribution #DataVisualization #Week2 #Foundation
---

## Overview (BLUF)

**One-sentence definition**: Categorical data describes qualities or categories without numeric meaning (like colors, genders, product types), and we analyze it using frequency counts, proportions, and visual tools like bar charts and pie charts.

**Why it matters**: Most real-world data includes categorical variables—customer demographics, product categories, survey responses, medical diagnoses. Understanding how to summarize and visualize categorical data is the first step in exploratory data analysis (EDA). Before building ML models, you need to understand what categories exist, which are common, and how they relate to each other.

**When to use**: Customer segmentation (which market segment has most customers?), quality control (what defect types are most common?), survey analysis (what's the most popular response?), A/B testing (which variant performed better?), medical research (disease prevalence by group).

**Prerequisites**: Basic understanding of data types from [week-01-data-types-scales.md](week-01-data-types-scales.md) (nominal vs ordinal scales), familiarity with Python data structures (lists, dictionaries).

---

## Core Theory

### 1. Categorical vs Numerical Data (Quick Review)

**Categorical (Qualitative)** data: Values are categories/labels
- **Nominal**: No natural order (colors, names, product types)
  - Example: {Red, Blue, Green}, {Male, Female, Other}
- **Ordinal**: Categories with meaningful order (ratings, education level)
  - Example: {Low, Medium, High}, {Disagree, Neutral, Agree}

**Numerical (Quantitative)** data: Values are numbers with meaning
- **Discrete**: Countable values (number of purchases)
- **Continuous**: Measurable values (height, temperature)

**Key insight**: You can't compute mean of categorical data! "Average color" makes no sense. Instead, we count **frequencies**.

---

### 2. Frequency Distribution

**Definition**: A **frequency distribution** shows how often each category appears in a dataset.

**Components**:
1. **Frequency** ($f_i$): Count of occurrences for category $i$
2. **Relative Frequency** ($RF_i$): Proportion of total
   $$RF_i = \frac{f_i}{n}$$
   where $n = \sum f_i$ is total count

3. **Percentage**: Relative frequency × 100%
   $$\text{Percentage}_i = RF_i \times 100\% = \frac{f_i}{n} \times 100\%$$

4. **Cumulative Frequency**: Running total (mainly for ordinal data)

**Properties**:
- $\sum_{i=1}^{k} f_i = n$ (frequencies sum to total)
- $\sum_{i=1}^{k} RF_i = 1$ (relative frequencies sum to 1)
- $\sum_{i=1}^{k} \text{Percentage}_i = 100\%$

#### Example 1: Customer Purchase Analysis

**Scenario**: An e-commerce store wants to analyze customer orders by product category.

**Raw Data** (50 orders):
```
Electronics, Clothing, Electronics, Home, Electronics, Clothing, Books, Electronics,
Clothing, Home, Electronics, Books, Clothing, Electronics, Home, Clothing, Electronics,
Books, Clothing, Electronics, Home, Books, Electronics, Clothing, Home, Electronics,
Books, Clothing, Electronics, Books, Clothing, Home, Electronics, Books, Clothing,
Electronics, Home, Books, Clothing, Electronics, Books, Clothing, Electronics, Books,
Clothing, Electronics, Books, Clothing, Home, Electronics
```

**Step 1**: Count frequencies

| Category    | Frequency ($f_i$) |
|-------------|-------------------|
| Electronics | 18                |
| Clothing    | 15                |
| Books       | 9                 |
| Home        | 8                 |
| **Total**   | **50**            |

**Step 2**: Compute relative frequencies

| Category    | Frequency | Relative Frequency | Percentage |
|-------------|-----------|-------------------|------------|
| Electronics | 18        | 18/50 = 0.36      | 36%        |
| Clothing    | 15        | 15/50 = 0.30      | 30%        |
| Books       | 9         | 9/50 = 0.18       | 18%        |
| Home        | 8         | 8/50 = 0.16       | 16%        |
| **Total**   | **50**    | **1.00**          | **100%**   |

**Interpretation**:
- Electronics is most popular (36% of orders)
- Clothing second (30%)
- Books and Home are less popular (18% and 16%)

```python
import pandas as pd
import numpy as np

# Create data
data = ['Electronics', 'Clothing', 'Electronics', 'Home', 'Electronics',
        'Clothing', 'Books', 'Electronics', 'Clothing', 'Home',
        'Electronics', 'Books', 'Clothing', 'Electronics', 'Home',
        'Clothing', 'Electronics', 'Books', 'Clothing', 'Electronics',
        'Home', 'Books', 'Electronics', 'Clothing', 'Home',
        'Electronics', 'Books', 'Clothing', 'Electronics', 'Books',
        'Clothing', 'Home', 'Electronics', 'Books', 'Clothing',
        'Electronics', 'Home', 'Books', 'Clothing', 'Electronics',
        'Books', 'Clothing', 'Electronics', 'Books', 'Clothing',
        'Electronics', 'Books', 'Clothing', 'Home', 'Electronics']

# Create DataFrame
df = pd.DataFrame({'Category': data})

# Frequency distribution
freq_table = df['Category'].value_counts().sort_values(ascending=False)
print("Frequency Distribution:")
print(freq_table)
print()

# Relative frequency
rel_freq = df['Category'].value_counts(normalize=True).sort_values(ascending=False)
print("Relative Frequency:")
print(rel_freq)
print()

# Percentage
percentage = (rel_freq * 100).round(2)
print("Percentage:")
print(percentage)

# Combined table
summary = pd.DataFrame({
    'Frequency': freq_table,
    'Relative Frequency': rel_freq.round(4),
    'Percentage': percentage
})
print("\nSummary Table:")
print(summary)
```

---

### 3. Visualizing Categorical Data

**Purpose**: Humans understand pictures better than tables. Visualizations reveal patterns instantly.

#### 3.1 Bar Chart

**Definition**: Bars represent frequencies, with height proportional to count.

**When to use**:
- Comparing frequencies across categories
- Works for any categorical data (nominal or ordinal)
- Preferred for many categories (>5)

**Best practices**:
- ✅ Order bars by frequency (descending) for nominal data
- ✅ Use horizontal bars if category names are long
- ✅ Include data labels on bars
- ✅ Space between bars (not touching)
- ❌ Don't use 3D effects (distorts perception)
- ❌ Don't start y-axis above zero (misleading)

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style('whitegrid')
plt.figure(figsize=(10, 6))

# Bar chart
freq_table.plot(kind='bar', color='skyblue', edgecolor='black')
plt.title('Product Category Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Category', fontsize=12)
plt.ylabel('Frequency (Count)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)

# Add data labels on bars
for i, v in enumerate(freq_table):
    plt.text(i, v + 0.5, str(v), ha='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

#### 3.2 Horizontal Bar Chart (Better for Long Labels)

```python
plt.figure(figsize=(10, 6))

# Horizontal bar chart
freq_table.plot(kind='barh', color='lightcoral', edgecolor='black')
plt.title('Product Category Distribution', fontsize=16, fontweight='bold')
plt.xlabel('Frequency (Count)', fontsize=12)
plt.ylabel('Category', fontsize=12)
plt.grid(axis='x', alpha=0.3)

# Add data labels
for i, v in enumerate(freq_table):
    plt.text(v + 0.5, i, str(v), va='center', fontweight='bold')

plt.tight_layout()
plt.show()
```

#### 3.3 Pie Chart

**Definition**: Circle divided into slices, with area proportional to frequency.

**When to use**:
- Showing parts of a whole
- Small number of categories (<6)
- Emphasizing proportions/percentages

**Best practices**:
- ✅ Start largest slice at 12 o'clock
- ✅ Order slices by size (clockwise descending)
- ✅ Include percentages in labels
- ✅ Use distinct colors
- ❌ Don't use for many categories (becomes cluttered)
- ❌ Don't use 3D (distorts areas)
- ❌ Don't "explode" multiple slices (confusing)

```python
plt.figure(figsize=(10, 8))

# Pie chart with custom colors
colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
explode = (0.1, 0, 0, 0)  # Explode largest slice

plt.pie(freq_table, labels=freq_table.index, autopct='%1.1f%%',
        colors=colors, explode=explode, startangle=90,
        textprops={'fontsize': 12, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})

plt.title('Product Category Distribution', fontsize=16, fontweight='bold', pad=20)
plt.axis('equal')  # Equal aspect ratio ensures circular pie
plt.tight_layout()
plt.show()
```

#### 3.4 Pareto Chart

**Definition**: Bar chart + cumulative line, ordered by frequency (descending).

**Purpose**: Identify the "vital few" (80/20 rule). Often, 80% of effects come from 20% of causes.

**When to use**:
- Quality control (which defects are most common?)
- Business analysis (which customers generate most revenue?)
- Resource allocation (focus on high-impact categories)

```python
fig, ax1 = plt.figure(figsize=(12, 6)), plt.gca()

# Sort by frequency
sorted_freq = freq_table.sort_values(ascending=False)
cumulative_pct = (sorted_freq.cumsum() / sorted_freq.sum() * 100)

# Bar chart
ax1.bar(range(len(sorted_freq)), sorted_freq, color='skyblue', edgecolor='black', label='Frequency')
ax1.set_xlabel('Category', fontsize=12)
ax1.set_ylabel('Frequency', fontsize=12, color='blue')
ax1.set_xticks(range(len(sorted_freq)))
ax1.set_xticklabels(sorted_freq.index, rotation=45, ha='right')
ax1.tick_params(axis='y', labelcolor='blue')

# Cumulative percentage line
ax2 = ax1.twinx()
ax2.plot(range(len(sorted_freq)), cumulative_pct, color='red', marker='o',
         linewidth=2, markersize=8, label='Cumulative %')
ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.axhline(y=80, color='green', linestyle='--', linewidth=1.5, label='80% threshold')
ax2.set_ylim(0, 105)

plt.title('Pareto Chart: Product Categories', fontsize=16, fontweight='bold')
fig.legend(loc='upper left', bbox_to_anchor=(0.15, 0.88))
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Identify 80% threshold
threshold_80 = cumulative_pct[cumulative_pct <= 80].index[-1]
print(f"Top categories covering 80% of orders: {list(sorted_freq[sorted_freq.index <= threshold_80].index)}")
```

**Interpretation**: Electronics + Clothing account for 66% of orders. These two categories should get most marketing/inventory focus.

---

### 4. Mode - Measure of Central Tendency for Categorical Data

**Definition**: The **mode** is the most frequently occurring value.

**Properties**:
- Only measure of central tendency for nominal data
- Can have multiple modes (bimodal, multimodal)
- Not affected by extreme values
- May not exist if all frequencies are equal

**For ordinal data**: Can also use **median** (middle rank when ordered).

#### Example 2: Survey Analysis

**Scenario**: 100 customers rated product satisfaction.

| Rating          | Frequency |
|-----------------|-----------|
| Very Dissatisfied | 5       |
| Dissatisfied    | 15        |
| Neutral         | 20        |
| Satisfied       | 40        |
| Very Satisfied  | 20        |

**Mode**: "Satisfied" (highest frequency = 40)

**Median**: Since data is ordinal and we have 100 responses:
- Cumulative frequencies: 5, 20, 40, 80, 100
- 50th observation falls in "Satisfied" category
- **Median** = "Satisfied"

**Interpretation**: Typical customer is satisfied with the product.

```python
# Survey data
ratings = ['Very Dissatisfied']*5 + ['Dissatisfied']*15 + ['Neutral']*20 + \
          ['Satisfied']*40 + ['Very Satisfied']*20

df_survey = pd.DataFrame({'Rating': ratings})

# Mode
mode_rating = df_survey['Rating'].mode()[0]
print(f"Mode (most common rating): {mode_rating}")

# For ordinal data, we can find median position
# Define order
order = ['Very Dissatisfied', 'Dissatisfied', 'Neutral', 'Satisfied', 'Very Satisfied']
df_survey['Rating_Code'] = pd.Categorical(df_survey['Rating'], categories=order, ordered=True)

# Frequency table
freq_survey = df_survey['Rating'].value_counts()[order]
print("\nFrequency Distribution:")
print(freq_survey)

# Cumulative frequency
cumfreq = freq_survey.cumsum()
print("\nCumulative Frequency:")
print(cumfreq)

# Median (50th percentile position)
median_position = len(df_survey) / 2
median_category = cumfreq[cumfreq >= median_position].index[0]
print(f"\nMedian (50th percentile): {median_category}")
```

---

### 5. Cross-Tabulation (Contingency Tables)

**Definition**: A **cross-tabulation** (crosstab) shows the frequency distribution of two categorical variables simultaneously.

**Purpose**: Examine relationship between two categorical variables.

**Structure**:
- Rows: Categories of first variable
- Columns: Categories of second variable
- Cells: Frequency of each combination
- Margins: Row/column totals

#### Example 3: Marketing Campaign Analysis

**Scenario**: Analyzing customer response by age group and gender.

**Data**: 200 customers

| Gender / Age Group | 18-25 | 26-35 | 36-45 | 46+ | **Total** |
|--------------------|-------|-------|-------|-----|-----------|
| Male               | 20    | 30    | 25    | 15  | **90**    |
| Female             | 25    | 35    | 30    | 20  | **110**   |
| **Total**          | **45**| **65**| **55**| **35** | **200** |

**Observations**:
- More female respondents (110 vs 90)
- 26-35 age group most represented (65 total)
- Female 26-35 is largest segment (35)

**Row percentages** (within each gender):

| Gender / Age Group | 18-25  | 26-35  | 36-45  | 46+   | **Total** |
|--------------------|--------|--------|--------|-------|-----------|
| Male               | 22.2%  | 33.3%  | 27.8%  | 16.7% | **100%**  |
| Female             | 22.7%  | 31.8%  | 27.3%  | 18.2% | **100%**  |

**Column percentages** (within each age group):

| Gender / Age Group | 18-25  | 26-35  | 36-45  | 46+   |
|--------------------|--------|--------|--------|-------|
| Male               | 44.4%  | 46.2%  | 45.5%  | 42.9% |
| Female             | 55.6%  | 53.8%  | 54.5%  | 57.1% |
| **Total**          | **100%**| **100%**| **100%** | **100%** |

```python
# Create sample data
np.random.seed(42)
gender_data = ['Male']*90 + ['Female']*110
age_data = ['18-25']*20 + ['26-35']*30 + ['36-45']*25 + ['46+']*15 + \
           ['18-25']*25 + ['26-35']*35 + ['36-45']*30 + ['46+']*20

df_marketing = pd.DataFrame({
    'Gender': gender_data,
    'Age_Group': age_data
})

# Cross-tabulation
crosstab = pd.crosstab(df_marketing['Gender'], df_marketing['Age_Group'], margins=True)
print("Cross-Tabulation (Frequency):")
print(crosstab)
print()

# Row percentages
row_pct = pd.crosstab(df_marketing['Gender'], df_marketing['Age_Group'], normalize='index') * 100
print("Row Percentages (within gender):")
print(row_pct.round(1))
print()

# Column percentages
col_pct = pd.crosstab(df_marketing['Gender'], df_marketing['Age_Group'], normalize='columns') * 100
print("Column Percentages (within age group):")
print(col_pct.round(1))
```

**Visualization** - Grouped Bar Chart:

```python
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Grouped bar chart
crosstab_no_margin = pd.crosstab(df_marketing['Gender'], df_marketing['Age_Group'])
crosstab_no_margin.plot(kind='bar', ax=axes[0], color=['skyblue', 'lightcoral', 'lightgreen', 'gold'],
                        edgecolor='black')
axes[0].set_title('Customer Distribution by Gender and Age Group', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Gender', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].legend(title='Age Group', fontsize=10)
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
axes[0].grid(axis='y', alpha=0.3)

# Stacked bar chart
crosstab_no_margin.T.plot(kind='bar', stacked=True, ax=axes[1],
                          color=['steelblue', 'salmon'], edgecolor='black')
axes[1].set_title('Age Group Distribution by Gender', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Age Group', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].legend(title='Gender', fontsize=10)
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## Data Science Applications

### 1. Customer Segmentation

**Problem**: E-commerce company wants to understand customer base by region and product interest.

```python
# Sample customer data
customers = pd.DataFrame({
    'Region': ['North', 'South', 'East', 'West', 'North', 'South', 'East', 'West'] * 125,
    'Interest': ['Tech', 'Fashion', 'Home', 'Books', 'Fashion', 'Tech', 'Books', 'Home'] * 125
})

# Cross-tabulation
segment_crosstab = pd.crosstab(customers['Region'], customers['Interest'])
print(segment_crosstab)

# Heatmap visualization
plt.figure(figsize=(10, 6))
sns.heatmap(segment_crosstab, annot=True, fmt='d', cmap='YlGnBu', linewidths=0.5)
plt.title('Customer Segmentation: Region vs Interest', fontsize=16, fontweight='bold')
plt.xlabel('Interest Category', fontsize=12)
plt.ylabel('Region', fontsize=12)
plt.tight_layout()
plt.show()
```

### 2. A/B Testing Results

**Problem**: Which website variant (A or B) has better conversion rate?

```python
# A/B test data
ab_test = pd.DataFrame({
    'Variant': ['A']*500 + ['B']*500,
    'Converted': ['Yes']*80 + ['No']*420 + ['Yes']*120 + ['No']*380
})

# Crosstab
conversion_table = pd.crosstab(ab_test['Variant'], ab_test['Converted'], normalize='index') * 100
print("Conversion Rates:")
print(conversion_table.round(2))

# Visualization
conversion_table.plot(kind='bar', stacked=False, color=['salmon', 'lightgreen'],
                     figsize=(8, 6), edgecolor='black')
plt.title('A/B Test: Conversion Rate Comparison', fontsize=16, fontweight='bold')
plt.xlabel('Variant', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(rotation=0)
plt.legend(title='Converted', fontsize=10)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

**Result**: Variant B has 24% conversion vs 16% for A → Deploy Variant B!

### 3. Survey Analysis Dashboard

**Complete workflow** from raw data to insights:

```python
# Simulate survey data
np.random.seed(42)
survey_responses = pd.DataFrame({
    'Satisfaction': np.random.choice(['Very Satisfied', 'Satisfied', 'Neutral', 'Dissatisfied', 'Very Dissatisfied'],
                                    size=300, p=[0.25, 0.35, 0.20, 0.15, 0.05]),
    'Product': np.random.choice(['Product A', 'Product B', 'Product C'], size=300),
    'Age_Group': np.random.choice(['18-30', '31-50', '51+'], size=300)
})

# Create dashboard
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. Overall satisfaction distribution
satisfaction_counts = survey_responses['Satisfaction'].value_counts()
axes[0, 0].bar(satisfaction_counts.index, satisfaction_counts.values, color='skyblue', edgecolor='black')
axes[0, 0].set_title('Overall Satisfaction Distribution', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Satisfaction Level')
axes[0, 0].set_ylabel('Count')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].grid(axis='y', alpha=0.3)

# 2. Satisfaction by product
product_satisfaction = pd.crosstab(survey_responses['Product'], survey_responses['Satisfaction'])
product_satisfaction.plot(kind='barh', stacked=True, ax=axes[0, 1], color=sns.color_palette('Set2'))
axes[0, 1].set_title('Satisfaction by Product', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Count')
axes[0, 1].legend(title='Satisfaction', bbox_to_anchor=(1.05, 1), loc='upper left')

# 3. Pie chart - Product distribution
product_counts = survey_responses['Product'].value_counts()
axes[1, 0].pie(product_counts, labels=product_counts.index, autopct='%1.1f%%', startangle=90,
              colors=['#ff9999', '#66b3ff', '#99ff99'])
axes[1, 0].set_title('Product Response Distribution', fontsize=14, fontweight='bold')

# 4. Heatmap - Age group vs Product
age_product = pd.crosstab(survey_responses['Age_Group'], survey_responses['Product'])
sns.heatmap(age_product, annot=True, fmt='d', cmap='YlOrRd', ax=axes[1, 1], linewidths=0.5)
axes[1, 1].set_title('Responses by Age Group and Product', fontsize=14, fontweight='bold')
axes[1, 1].set_xlabel('Product')
axes[1, 1].set_ylabel('Age Group')

plt.tight_layout()
plt.show()

# Statistical summary
print("\nKey Insights:")
print(f"Total responses: {len(survey_responses)}")
print(f"Mode (most common satisfaction): {survey_responses['Satisfaction'].mode()[0]}")
print(f"\nSatisfaction rate: {100 * (survey_responses['Satisfaction'].isin(['Very Satisfied', 'Satisfied']).sum() / len(survey_responses)):.1f}%")
```

---

## Common Pitfalls and Misconceptions

### Pitfall 1: Computing Mean of Categorical Data

❌ **Wrong**: "Average product category is 2.5"

✅ **Right**: Use mode, not mean. Categories have no numeric meaning.

### Pitfall 2: Using Pie Charts for Many Categories

❌ **Wrong**: Pie chart with 15 slices (impossible to read)

✅ **Right**: Use bar chart for >5 categories.

### Pitfall 3: Not Ordering Bars

❌ **Wrong**: Random bar order (hard to compare)

✅ **Right**: Order bars by frequency (descending) for nominal data.

### Pitfall 4: 3D Charts

❌ **Wrong**: 3D pie/bar charts (perspective distorts perception)

✅ **Right**: Stick to 2D visualizations.

### Pitfall 5: Ignoring Missing Data

❌ **Wrong**: Exclude missing categories without noting

✅ **Right**: Report missing data frequency, decide how to handle (exclude, impute, separate category).

```python
# Handling missing data
data_with_missing = pd.Series(['A', 'B', 'A', np.nan, 'C', 'B', np.nan, 'A'])

# Option 1: Count including NaN
print("With missing:")
print(data_with_missing.value_counts(dropna=False))

# Option 2: Count excluding NaN
print("\nWithout missing:")
print(data_with_missing.value_counts(dropna=True))

# Option 3: Fill missing with 'Unknown'
data_filled = data_with_missing.fillna('Unknown')
print("\nWith 'Unknown' category:")
print(data_filled.value_counts())
```

---

## Self-Assessment and Active Recall

### Concept Check Questions

1. **Explain**: Why can't you compute the mean of nominal categorical data?

2. **Choose**: When should you use a pie chart vs a bar chart?

3. **Interpret**: If mode = "Satisfied" and median = "Neutral" for ordinal data, what does this tell you?

4. **Calculate**: Given frequencies {A:20, B:30, C:10}, what are relative frequencies?

5. **Identify**: In a Pareto chart, which categories should you focus on and why?

### Practice Problems

#### Basic Level

1. Create a frequency table for: {Red, Blue, Red, Green, Blue, Red, Red, Blue}

2. Given frequencies {Small:15, Medium:25, Large:10}, compute percentages.

3. Which visualization (bar/pie) is better for 8 categories? Why?

#### Intermediate Level

4. Create cross-tabulation for:
   - Gender: {M, F, M, F, M, M, F, F}
   - Response: {Yes, No, Yes, Yes, No, Yes, No, Yes}

5. Interpret this crosstab - what's the relationship?

| City / Product | A   | B   |
|----------------|-----|-----|
| NYC            | 50  | 30  |
| LA             | 20  | 60  |

6. Given survey with ratings {1:5, 2:10, 3:20, 4:15, 5:10}, find mode and median.

#### Advanced Level

7. Design a Pareto analysis for defect types: {Scratch:40, Dent:25, Paint:15, Crack:10, Other:10}

8. Compare two cross-tabulations - which product has better female appeal?

9. Create comprehensive visualization dashboard for customer data with 3 categorical variables.

---

## Quick Reference Summary

### Key Formulas

**Relative Frequency**:
$$RF_i = \frac{f_i}{n}$$

**Percentage**:
$$\text{Percentage}_i = \frac{f_i}{n} \times 100\%$$

**Mode**: Most frequent category

### Chart Selection Guide

| Data Characteristics | Best Chart |
|---------------------|------------|
| Few categories (<6) | Pie chart or bar chart |
| Many categories (>5) | Bar chart (ordered) |
| Show proportions | Pie chart |
| Compare categories | Bar chart |
| Identify vital few | Pareto chart |
| Two variables | Grouped/stacked bar or heatmap |
| Long category names | Horizontal bar chart |

### Python Code Templates

```python
# Frequency table
freq = df['column'].value_counts()

# Relative frequency
rel_freq = df['column'].value_counts(normalize=True)

# Bar chart
freq.plot(kind='bar')

# Pie chart
plt.pie(freq, labels=freq.index, autopct='%1.1f%%')

# Cross-tabulation
pd.crosstab(df['var1'], df['var2'], margins=True)

# Mode
mode = df['column'].mode()[0]
```

### Top 3 Things to Remember

1. **Frequencies, not means**: Use counts and proportions for categorical data
2. **Visualize effectively**: Choose right chart type, order bars, avoid 3D
3. **Mode for center**: Only valid measure of central tendency for nominal data

---

## Further Resources

### Documentation
- Pandas: `value_counts()`, `crosstab()`
- Matplotlib: Bar plots, pie charts
- Seaborn: `countplot()`, `heatmap()`

### Books
- Freedman, Pisani, Purves, "Statistics" - Chapter 3
- Moore & McCabe, "Introduction to the Practice of Statistics"

### Practice
- Kaggle Datasets: Titanic (survival by class/gender), House Prices (neighborhood)
- Real surveys: Customer satisfaction, political polls

### Review Schedule
- **After 1 day**: Recreate frequency tables from memory
- **After 3 days**: Practice choosing appropriate visualizations
- **After 1 week**: Analyze real dataset with categorical variables
- **After 2 weeks**: Create comprehensive EDA dashboard

---

**Related Notes**:
- Previous: [week-01-data-types-scales.md](week-01-data-types-scales.md)
- Next: [week-03-numerical-data-summaries.md](week-03-numerical-data-summaries.md)
- Application: Exploratory Data Analysis, Customer Segmentation

**Last Updated**: 2025-11-22
**Status**: Complete and comprehensive ✅
