# BSMA1002: Statistics for Data Science I - Study Guide

**Course ID:** BSMA1002  
**Credits:** 4  
**Duration:** 12 weeks  
**Instructor:** Usha Mohan  
**Prerequisites:** None

## 📚 Course Overview

This course provides a comprehensive introduction to descriptive statistics and probability, essential foundations for data science. You'll learn to analyze datasets, visualize data patterns, understand probability concepts, and work with common probability distributions.

## 🎯 Learning Objectives

By the end of this course, you will be able to:
- Create and analyze datasets using appropriate statistical measures
- Describe data using measures of central tendency and dispersion
- Visualize data effectively using various chart types
- Understand and apply probability concepts
- Work with binomial and normal distributions
- Estimate probabilities from data
- Apply statistical thinking to real-world problems

## 📖 Reference Materials

**Required Books (Available for Download):**
- **Descriptive Statistics (Vol 1)** - [Download from course page](https://study.iitm.ac.in/ds/course_pages/BSMA1002.html)
- **Probability and Probability Distributions (Vol 2)** - [Download from course page](https://study.iitm.ac.in/ds/course_pages/BSMA1002.html)

**Video Lectures:** [YouTube Playlist](https://www.youtube.com/playlist?list=PLZ2ps__7DhBYN3kpiF5ZqSHx8d_rIkwqK)

---

## 📅 Week-by-Week Breakdown

### Week 1: Introduction to Statistics and Data Types

**Topics Covered:**
- What is statistics and why it matters for data science
- Types of data (Categorical vs Numerical)
- Nominal vs Ordinal data
- Discrete vs Continuous data
- Scales of measurement (Nominal, Ordinal, Interval, Ratio)
- Data collection methods
- Populations vs Samples

**Learning Activities:**
1. **Read:** Descriptive Statistics Vol 1, Chapter 1
2. **Watch:** Week 1 video lectures
3. **Practice:** Classify different types of data
4. **Code:** Load and explore datasets using Pandas

**Key Concepts:**
- **Categorical Data:** Data that can be divided into groups (e.g., colors, gender, categories)
- **Numerical Data:** Data that represents quantities (e.g., height, weight, temperature)
- **Scale of Measurement:** Determines what mathematical operations are valid

**Practice Problems:**
- Identify data types in real datasets
- Determine appropriate scales of measurement
- Design data collection strategies
- Distinguish between population and sample

**Python Applications:**
```python
import pandas as pd
import numpy as np

# Load a dataset
df = pd.read_csv('data.csv')

# Inspect data types
print(df.dtypes)
print(df.info())

# Identify categorical vs numerical columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Basic statistics
print(df.describe())  # For numerical columns
print(df.describe(include=['object']))  # For categorical columns
```

**Data Science Connection:** Understanding data types is crucial for choosing appropriate analysis methods and visualizations.

**Weekly Notebook:** `week-01-data-types.ipynb`

---

### Week 2: Describing Categorical Data

**Topics Covered:**
- Frequency tables and distributions
- Relative frequencies and percentages
- Bar charts and pie charts
- Pareto charts
- Mode as a measure of central tendency
- Cross-tabulation (contingency tables)
- Visualizing categorical data effectively

**Learning Activities:**
1. **Read:** Descriptive Statistics Vol 1, Chapter 2
2. **Watch:** Week 2 video lectures
3. **Practice:** Create frequency tables and charts
4. **Code:** Visualize categorical data using matplotlib and seaborn

**Key Concepts:**
- **Frequency Distribution:** Count of occurrences for each category
- **Relative Frequency:** Proportion or percentage of total
- **Mode:** Most frequently occurring value
- **Cross-tabulation:** Relationship between two categorical variables

**Practice Problems:**
- Create frequency distributions for different datasets
- Choose appropriate visualizations
- Interpret pie charts and bar charts
- Analyze cross-tabulations

**Python Applications:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Frequency table
frequency = df['category'].value_counts()
relative_freq = df['category'].value_counts(normalize=True)

# Bar chart
plt.figure(figsize=(10, 6))
frequency.plot(kind='bar')
plt.title('Frequency Distribution')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()

# Pie chart
plt.figure(figsize=(8, 8))
plt.pie(frequency, labels=frequency.index, autopct='%1.1f%%')
plt.title('Distribution of Categories')
plt.show()

# Cross-tabulation
pd.crosstab(df['category1'], df['category2'], margins=True)

# Seaborn countplot
sns.countplot(data=df, x='category', palette='viridis')
plt.title('Count Plot')
plt.show()
```

**Data Science Connection:** Categorical data analysis is essential for exploratory data analysis (EDA) and understanding customer segments, product categories, etc.

**Weekly Notebook:** `week-02-categorical-data.ipynb`

---

### Week 3: Describing Numerical Data

**Topics Covered:**
- Measures of central tendency (Mean, Median, Mode)
- When to use each measure
- Measures of dispersion (Range, Variance, Standard Deviation)
- Quartiles and Interquartile Range (IQR)
- Outlier detection
- Box plots and histograms
- Skewness and kurtosis

**Learning Activities:**
1. **Read:** Descriptive Statistics Vol 1, Chapter 3
2. **Watch:** Week 3 video lectures
3. **Practice:** Calculate descriptive statistics
4. **Code:** Create comprehensive statistical summaries

**Key Concepts:**
- **Mean:** Average value, sensitive to outliers
- **Median:** Middle value, robust to outliers
- **Standard Deviation:** Average distance from the mean
- **IQR:** Range of the middle 50% of data
- **Outliers:** Data points far from the bulk of data

**Practice Problems:**
- Calculate mean, median, mode for various datasets
- Compute variance and standard deviation
- Identify and handle outliers
- Interpret box plots and histograms

**Python Applications:**
```python
# Measures of central tendency
data = df['numerical_column']

mean = data.mean()
median = data.median()
mode = data.mode()[0]

print(f"Mean: {mean:.2f}")
print(f"Median: {median:.2f}")
print(f"Mode: {mode:.2f}")

# Measures of dispersion
range_val = data.max() - data.min()
variance = data.var()
std_dev = data.std()
iqr = data.quantile(0.75) - data.quantile(0.25)

print(f"Range: {range_val:.2f}")
print(f"Variance: {variance:.2f}")
print(f"Std Dev: {std_dev:.2f}")
print(f"IQR: {iqr:.2f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram
axes[0].hist(data, bins=30, edgecolor='black', alpha=0.7)
axes[0].axvline(mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
axes[0].axvline(median, color='g', linestyle='--', label=f'Median: {median:.2f}')
axes[0].set_title('Histogram with Mean and Median')
axes[0].legend()

# Box plot
axes[1].boxplot(data)
axes[1].set_title('Box Plot')
axes[1].set_ylabel('Value')

plt.tight_layout()
plt.show()

# Outlier detection using IQR method
Q1 = data.quantile(0.25)
Q3 = data.quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers = data[(data < lower_bound) | (data > upper_bound)]
print(f"Number of outliers: {len(outliers)}")
```

**Data Science Connection:** These fundamental statistics appear in almost every data analysis - from feature engineering to model evaluation.

**Weekly Notebook:** `week-03-numerical-data.ipynb`

---

### Week 4: Association Between Variables

**Topics Covered:**
- Scatter plots
- Covariance
- Correlation coefficient (Pearson's r)
- Interpretation of correlation
- Correlation vs causation
- Spearman's rank correlation
- Point-biserial correlation
- Correlation matrices

**Learning Activities:**
1. **Read:** Descriptive Statistics Vol 1, Chapter 4
2. **Watch:** Week 4 video lectures
3. **Practice:** Calculate correlations
4. **Code:** Create correlation matrices and visualizations

**Key Concepts:**
- **Covariance:** Measure of how two variables change together
- **Correlation:** Standardized covariance (-1 to +1)
- **Positive Correlation:** Variables increase together
- **Negative Correlation:** One increases as other decreases
- **Zero Correlation:** No linear relationship

**Practice Problems:**
- Calculate correlation coefficients
- Interpret scatter plots
- Identify spurious correlations
- Compare Pearson and Spearman correlations

**Python Applications:**
```python
# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(df['var1'], df['var2'], alpha=0.6)
plt.xlabel('Variable 1')
plt.ylabel('Variable 2')
plt.title('Scatter Plot')
plt.show()

# Correlation coefficient
correlation = df['var1'].corr(df['var2'])
print(f"Pearson correlation: {correlation:.3f}")

# Correlation matrix
correlation_matrix = df.corr()
print(correlation_matrix)

# Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix Heatmap')
plt.show()

# Pairplot for multiple variables
sns.pairplot(df[['var1', 'var2', 'var3', 'var4']])
plt.suptitle('Pairplot of Variables', y=1.02)
plt.show()

# Spearman correlation (for ordinal data)
from scipy.stats import spearmanr
spearman_corr, p_value = spearmanr(df['var1'], df['var2'])
print(f"Spearman correlation: {spearman_corr:.3f}")
```

**Data Science Connection:** Understanding correlations is crucial for feature selection, dimensionality reduction, and identifying multicollinearity in regression models.

**Weekly Notebook:** `week-04-correlation.ipynb`

---

### Week 5: Introduction to Probability

**Topics Covered:**
- Sample spaces and events
- Classical, empirical, and subjective probability
- Basic probability rules
- Addition rule
- Complement rule
- Probability of compound events
- Conditional probability basics

**Learning Activities:**
1. **Read:** Probability and Probability Distributions Vol 2, Chapter 1
2. **Watch:** Week 5 video lectures
3. **Practice:** Solve probability problems
4. **Code:** Simulate probability experiments

**Key Concepts:**
- **Sample Space:** Set of all possible outcomes
- **Event:** Subset of sample space
- **Probability:** P(A) = Number of favorable outcomes / Total outcomes
- **Complement:** P(A') = 1 - P(A)

**Practice Problems:**
- Calculate probabilities using counting techniques
- Apply probability rules
- Solve problems involving multiple events
- Use Venn diagrams

**Python Applications:**
```python
import random

# Simulate coin flips
def simulate_coin_flips(n):
    flips = [random.choice(['H', 'T']) for _ in range(n)]
    heads = flips.count('H')
    return heads / n

# Run simulation
n_flips = 10000
prob_heads = simulate_coin_flips(n_flips)
print(f"Probability of heads (n={n_flips}): {prob_heads:.4f}")

# Simulate dice rolls
def simulate_dice_rolls(n):
    rolls = [random.randint(1, 6) for _ in range(n)]
    return np.bincount(rolls)[1:] / n

n_rolls = 10000
probabilities = simulate_dice_rolls(n_rolls)
print("Probabilities for each face:")
for face, prob in enumerate(probabilities, 1):
    print(f"Face {face}: {prob:.4f}")

# Visualize
plt.bar(range(1, 7), probabilities)
plt.axhline(y=1/6, color='r', linestyle='--', label='Theoretical (1/6)')
plt.xlabel('Dice Face')
plt.ylabel('Probability')
plt.title('Simulated Dice Roll Probabilities')
plt.legend()
plt.show()
```

**Data Science Connection:** Probability theory is the foundation of statistical inference, hypothesis testing, and machine learning algorithms.

**Weekly Notebook:** `week-05-probability-basics.ipynb`

---

### Week 6: Conditional Probability and Independence

**Topics Covered:**
- Conditional probability
- Multiplication rule
- Independence of events
- Bayes' theorem
- Law of total probability
- Tree diagrams
- Applications to real problems

**Learning Activities:**
1. **Read:** Probability and Probability Distributions Vol 2, Chapter 2
2. **Watch:** Week 6 video lectures
3. **Practice:** Apply Bayes' theorem
4. **Code:** Implement Bayesian calculations

**Key Concepts:**
- **Conditional Probability:** P(A|B) = P(A and B) / P(B)
- **Independence:** P(A|B) = P(A) when A and B are independent
- **Bayes' Theorem:** P(A|B) = P(B|A) × P(A) / P(B)

**Practice Problems:**
- Calculate conditional probabilities
- Determine if events are independent
- Apply Bayes' theorem to medical testing, spam filtering
- Draw and analyze tree diagrams

**Python Applications:**
```python
# Bayes' theorem implementation
def bayes_theorem(prior, likelihood, evidence):
    """
    Calculate posterior probability using Bayes' theorem
    P(A|B) = P(B|A) * P(A) / P(B)
    """
    posterior = (likelihood * prior) / evidence
    return posterior

# Example: Medical test
# P(Disease) = 0.01 (1% prevalence)
# P(Positive|Disease) = 0.95 (95% sensitivity)
# P(Positive|No Disease) = 0.05 (5% false positive rate)

prior_disease = 0.01
prior_no_disease = 0.99
likelihood_positive_given_disease = 0.95
likelihood_positive_given_no_disease = 0.05

# Calculate P(Positive) using law of total probability
prob_positive = (likelihood_positive_given_disease * prior_disease + 
                 likelihood_positive_given_no_disease * prior_no_disease)

# Calculate P(Disease|Positive) using Bayes' theorem
posterior_disease_given_positive = bayes_theorem(
    prior_disease, 
    likelihood_positive_given_disease, 
    prob_positive
)

print(f"P(Disease|Positive Test) = {posterior_disease_given_positive:.4f}")
print(f"That's {posterior_disease_given_positive*100:.2f}%")

# Simulation to verify
def simulate_medical_test(n_people=100000):
    people = np.random.choice(['Disease', 'Healthy'], 
                              size=n_people, 
                              p=[0.01, 0.99])
    
    tests = []
    for person in people:
        if person == 'Disease':
            test = np.random.choice(['Positive', 'Negative'], p=[0.95, 0.05])
        else:
            test = np.random.choice(['Positive', 'Negative'], p=[0.05, 0.95])
        tests.append(test)
    
    # Calculate P(Disease|Positive)
    positive_tests = [i for i, test in enumerate(tests) if test == 'Positive']
    disease_and_positive = sum([people[i] == 'Disease' for i in positive_tests])
    
    return disease_and_positive / len(positive_tests)

simulated_prob = simulate_medical_test()
print(f"Simulated P(Disease|Positive) = {simulated_prob:.4f}")
```

**Data Science Connection:** Bayes' theorem is fundamental to Bayesian statistics, spam filters, recommendation systems, and naive Bayes classifiers.

**Weekly Notebook:** `week-06-conditional-probability.ipynb`

---

### Week 7-8: Probability Distributions - Discrete

**Topics Covered:**
- Random variables (discrete vs continuous)
- Probability mass functions (PMF)
- Expected value and variance
- Bernoulli distribution
- Binomial distribution
- Poisson distribution
- Geometric distribution
- Applications and when to use each

**Learning Activities:**
1. **Read:** Probability and Probability Distributions Vol 2, Chapters 3-4
2. **Watch:** Weeks 7-8 video lectures
3. **Practice:** Work with different distributions
4. **Code:** Implement and visualize distributions

**Key Concepts:**
- **Random Variable:** Variable whose value depends on chance
- **PMF:** P(X = x) for discrete random variables
- **Expected Value:** E(X) = Σ x·P(X=x)
- **Binomial:** Number of successes in n trials
- **Poisson:** Number of events in fixed interval

**Python Applications:**
```python
from scipy import stats

# Bernoulli distribution (coin flip)
p = 0.6  # Probability of success
bernoulli_rv = stats.bernoulli(p)
print(f"P(X=1) = {bernoulli_rv.pmf(1)}")

# Binomial distribution
n, p = 10, 0.5  # 10 trials, 50% success probability
binomial_rv = stats.binom(n, p)

# PMF
x = np.arange(0, n+1)
pmf = binomial_rv.pmf(x)

plt.figure(figsize=(10, 6))
plt.bar(x, pmf, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Successes')
plt.ylabel('Probability')
plt.title(f'Binomial Distribution (n={n}, p={p})')
plt.show()

# Calculate probabilities
print(f"P(X = 5) = {binomial_rv.pmf(5):.4f}")
print(f"P(X <= 5) = {binomial_rv.cdf(5):.4f}")
print(f"P(X > 5) = {1 - binomial_rv.cdf(5):.4f}")

# Expected value and variance
print(f"Expected value: {binomial_rv.mean():.2f}")
print(f"Variance: {binomial_rv.var():.2f}")

# Poisson distribution
lambda_param = 3  # Average rate
poisson_rv = stats.poisson(lambda_param)

x = np.arange(0, 15)
pmf = poisson_rv.pmf(x)

plt.figure(figsize=(10, 6))
plt.bar(x, pmf, edgecolor='black', alpha=0.7)
plt.xlabel('Number of Events')
plt.ylabel('Probability')
plt.title(f'Poisson Distribution (λ={lambda_param})')
plt.show()

# Simulate real scenario: Website visits per hour
np.random.seed(42)
visits_per_hour = np.random.poisson(lambda_param, 24)  # 24 hours
print(f"Simulated visits per hour (24 hours):\n{visits_per_hour}")
print(f"Total visits: {visits_per_hour.sum()}")
```

**Data Science Connection:** Binomial and Poisson distributions model many real phenomena - A/B testing, click-through rates, customer arrivals, etc.

**Weekly Notebook:** `week-07-08-discrete-distributions.ipynb`

---

### Week 9-10: Probability Distributions - Continuous

**Topics Covered:**
- Probability density functions (PDF)
- Cumulative distribution functions (CDF)
- Uniform distribution
- Normal (Gaussian) distribution
- Standard normal distribution
- Z-scores and standardization
- Normal approximation to binomial
- Applications of normal distribution

**Learning Activities:**
1. **Read:** Probability and Probability Distributions Vol 2, Chapters 5-6
2. **Watch:** Weeks 9-10 video lectures
3. **Practice:** Work with normal distribution
4. **Code:** Apply normalization and calculate probabilities

**Key Concepts:**
- **PDF:** f(x) ≥ 0 and integral = 1 (not a probability!)
- **Normal Distribution:** Bell curve, characterized by μ (mean) and σ (std dev)
- **68-95-99.7 Rule:** Empirical rule for normal distribution
- **Z-score:** (x - μ) / σ, standardizes values

**Python Applications:**
```python
# Normal distribution
mu, sigma = 100, 15  # Mean and standard deviation
normal_rv = stats.norm(mu, sigma)

# PDF
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
pdf = normal_rv.pdf(x)

plt.figure(figsize=(12, 6))
plt.plot(x, pdf, 'b-', linewidth=2)
plt.fill_between(x, pdf, where=(x >= mu-sigma) & (x <= mu+sigma), 
                 alpha=0.3, label='±1σ (68%)')
plt.fill_between(x, pdf, where=(x >= mu-2*sigma) & (x <= mu+2*sigma), 
                 alpha=0.2, label='±2σ (95%)')
plt.axvline(mu, color='r', linestyle='--', label=f'Mean ({mu})')
plt.xlabel('Value')
plt.ylabel('Density')
plt.title(f'Normal Distribution (μ={mu}, σ={sigma})')
plt.legend()
plt.show()

# Calculate probabilities
print(f"P(X ≤ 100) = {normal_rv.cdf(100):.4f}")
print(f"P(85 ≤ X ≤ 115) = {normal_rv.cdf(115) - normal_rv.cdf(85):.4f}")
print(f"P(X > 130) = {1 - normal_rv.cdf(130):.4f}")

# Z-scores
def z_score(x, mu, sigma):
    return (x - mu) / sigma

value = 115
z = z_score(value, mu, sigma)
print(f"Z-score for {value}: {z:.2f}")

# Standard normal distribution
standard_normal = stats.norm(0, 1)
print(f"P(Z ≤ {z:.2f}) = {standard_normal.cdf(z):.4f}")

# Inverse: Find value for given probability
percentile_95 = normal_rv.ppf(0.95)
print(f"95th percentile: {percentile_95:.2f}")

# Real application: Test scores
test_scores = np.random.normal(75, 10, 1000)  # 1000 students
plt.figure(figsize=(10, 6))
plt.hist(test_scores, bins=30, density=True, alpha=0.7, edgecolor='black')

# Overlay theoretical normal distribution
x = np.linspace(test_scores.min(), test_scores.max(), 1000)
plt.plot(x, stats.norm(75, 10).pdf(x), 'r-', linewidth=2, label='Theoretical')
plt.xlabel('Test Score')
plt.ylabel('Density')
plt.title('Distribution of Test Scores')
plt.legend()
plt.show()

# What percentage of students scored above 85?
percentage_above_85 = (test_scores > 85).sum() / len(test_scores) * 100
theoretical = (1 - stats.norm(75, 10).cdf(85)) * 100
print(f"Simulated: {percentage_above_85:.1f}% scored above 85")
print(f"Theoretical: {theoretical:.1f}% scored above 85")
```

**Data Science Connection:** The normal distribution is everywhere - central limit theorem ensures many phenomena are normally distributed. Essential for hypothesis testing and confidence intervals.

**Weekly Notebook:** `week-09-10-continuous-distributions.ipynb`

---

### Week 11: Sampling and Estimation

**Topics Covered:**
- Sampling methods (random, stratified, systematic)
- Sampling distributions
- Central Limit Theorem
- Point estimation
- Properties of estimators (bias, consistency, efficiency)
- Standard error
- Confidence intervals (introduction)

**Learning Activities:**
1. **Read:** Supplementary materials on sampling
2. **Watch:** Week 11 video lectures
3. **Practice:** Simulate sampling distributions
4. **Code:** Demonstrate Central Limit Theorem

**Key Concepts:**
- **Central Limit Theorem:** Sample means approach normal distribution as n increases
- **Standard Error:** Standard deviation of sampling distribution
- **Confidence Interval:** Range likely to contain parameter

**Python Applications:**
```python
# Demonstrate Central Limit Theorem
population = np.random.exponential(scale=2, size=100000)  # Non-normal population

# Take many samples and compute means
sample_sizes = [5, 10, 30, 100]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, n in enumerate(sample_sizes):
    sample_means = []
    for _ in range(1000):
        sample = np.random.choice(population, size=n, replace=False)
        sample_means.append(sample.mean())
    
    axes[idx].hist(sample_means, bins=30, density=True, alpha=0.7, edgecolor='black')
    
    # Overlay theoretical normal
    mu = np.mean(sample_means)
    sigma = np.std(sample_means)
    x = np.linspace(min(sample_means), max(sample_means), 100)
    axes[idx].plot(x, stats.norm(mu, sigma).pdf(x), 'r-', linewidth=2)
    axes[idx].set_title(f'Sample Size n={n}')
    axes[idx].set_xlabel('Sample Mean')
    axes[idx].set_ylabel('Density')

plt.tight_layout()
plt.show()

# Calculate confidence intervals
data = np.random.normal(100, 15, 100)
mean = np.mean(data)
std_error = stats.sem(data)  # Standard error of mean

# 95% confidence interval
confidence_level = 0.95
ci = stats.t.interval(confidence_level, len(data)-1, mean, std_error)
print(f"95% Confidence Interval: ({ci[0]:.2f}, {ci[1]:.2f})")
```

**Data Science Connection:** Sampling and estimation are crucial for A/B testing, survey analysis, and understanding model uncertainty.

**Weekly Notebook:** `week-11-sampling-estimation.ipynb`

---

### Week 12: Review and Applications

**Topics Covered:**
- Comprehensive review of all topics
- Integration of descriptive statistics and probability
- Real-world case studies
- Data analysis projects
- Exam preparation
- Best practices in statistical analysis

**Project Ideas:**
1. **Complete EDA:** Analyze a real dataset with comprehensive visualizations and statistics
2. **Probability Simulation:** Build simulations for complex probability problems
3. **A/B Test Analysis:** Design and analyze a simulated A/B test
4. **Distribution Fitting:** Fit different distributions to real data and compare

**Python Applications:**
```python
# Comprehensive data analysis template
def comprehensive_analysis(df, numerical_cols, categorical_cols):
    """Perform complete statistical analysis"""
    
    print("=" * 50)
    print("COMPREHENSIVE STATISTICAL ANALYSIS")
    print("=" * 50)
    
    # 1. Dataset Overview
    print("\n1. DATASET OVERVIEW")
    print(f"Shape: {df.shape}")
    print(f"Missing values:\n{df.isnull().sum()}")
    
    # 2. Numerical Analysis
    print("\n2. NUMERICAL VARIABLES ANALYSIS")
    print(df[numerical_cols].describe())
    
    # 3. Categorical Analysis
    print("\n3. CATEGORICAL VARIABLES ANALYSIS")
    for col in categorical_cols:
        print(f"\n{col}:")
        print(df[col].value_counts())
    
    # 4. Visualizations
    n_numerical = len(numerical_cols)
    fig, axes = plt.subplots(n_numerical, 2, figsize=(14, 5*n_numerical))
    
    for idx, col in enumerate(numerical_cols):
        # Histogram
        axes[idx, 0].hist(df[col].dropna(), bins=30, edgecolor='black', alpha=0.7)
        axes[idx, 0].set_title(f'{col} - Histogram')
        axes[idx, 0].set_xlabel(col)
        axes[idx, 0].set_ylabel('Frequency')
        
        # Box plot
        axes[idx, 1].boxplot(df[col].dropna())
        axes[idx, 1].set_title(f'{col} - Box Plot')
        axes[idx, 1].set_ylabel(col)
    
    plt.tight_layout()
    plt.show()
    
    # 5. Correlation analysis
    if len(numerical_cols) > 1:
        print("\n5. CORRELATION ANALYSIS")
        corr_matrix = df[numerical_cols].corr()
        print(corr_matrix)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.show()

# Usage example
# comprehensive_analysis(df, 
#                       numerical_cols=['age', 'income', 'score'],
#                       categorical_cols=['gender', 'category'])
```

**Review Checklist:**
- [ ] Understand all data types and scales
- [ ] Master descriptive statistics
- [ ] Create effective visualizations
- [ ] Calculate and interpret correlations
- [ ] Apply probability rules
- [ ] Work with conditional probability
- [ ] Use discrete and continuous distributions
- [ ] Apply Central Limit Theorem
- [ ] Calculate confidence intervals

**Weekly Notebook:** `week-12-comprehensive-review.ipynb`

---

## 🎯 Assessment Structure

- **Weekly Online Assignments:** 10-20%
- **Quiz 1 (In-person):** 15-20%
- **Quiz 2 (In-person):** 15-20%
- **End Term Exam (In-person):** 50-60%

**Passing Grade:** 40% overall with at least 40% in end-term exam

---

## 💡 Study Tips

1. **Work with Real Data:** Use Kaggle datasets to practice
2. **Visualize Everything:** Understanding comes from seeing patterns
3. **Practice Daily:** Solve at least 2-3 problems every day
4. **Learn by Coding:** Implement every concept in Python
5. **Connect to Applications:** Always think "Where is this used?"
6. **Use Simulations:** Verify probability calculations with simulations
7. **Form Study Groups:** Discuss interpretations with peers
8. **Master Pandas and Matplotlib:** Essential tools for data analysis

---

## 🔗 Important Links

- **Course Page:** https://study.iitm.ac.in/ds/course_pages/BSMA1002.html
- **Video Lectures:** [YouTube Playlist](https://www.youtube.com/playlist?list=PLZ2ps__7DhBYN3kpiF5ZqSHx8d_rIkwqK)
- **Kaggle Datasets:** https://www.kaggle.com/datasets

---

## 📚 Additional Resources

- **Think Stats** by Allen Downey (Free online)
- **OpenIntro Statistics** (Free textbook)
- **StatQuest YouTube Channel** - Excellent visual explanations
- **Khan Academy Statistics** - Practice problems

---

**Remember:** Statistics is about making sense of data. Every technique you learn has practical applications in understanding the world through data!
