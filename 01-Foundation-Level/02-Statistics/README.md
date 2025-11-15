# Statistics - IIT Madras Foundation

## 📖 Course Overview

Foundation level Statistics covering descriptive statistics, probability, and inferential statistics basics.

## 📚 Topics Covered

### Module 1: Introduction & Descriptive Statistics (Weeks 1-3)
- Data types and scales
- Measures of central tendency (mean, median, mode)
- Measures of dispersion (variance, SD, IQR)
- Data visualization basics

### Module 2: Probability (Weeks 4-6)
- Basic probability concepts
- Conditional probability
- Bayes' theorem
- Random variables
- Probability distributions

### Module 3: Discrete Distributions (Weeks 7-8)
- Binomial distribution
- Poisson distribution
- Geometric distribution

### Module 4: Continuous Distributions (Weeks 9-10)
- Normal distribution
- Uniform distribution
- Exponential distribution

### Module 5: Sampling & Estimation (Weeks 11-12)
- Sampling methods
- Central Limit Theorem
- Point estimation
- Confidence intervals

## 📂 Folder Contents

### `/notes`
Lecture notes and theory:
- `week-01-descriptive-stats.md`
- `week-02-probability.md`
- `formulas-cheatsheet.md`

### `/notebooks`
Jupyter notebooks with Python implementation:
- `01-descriptive-stats.ipynb` - Starter template ✅
- `02-probability.ipynb`
- `03-distributions.ipynb`
- `04-hypothesis-testing.ipynb`

### `/assignments`
Course assignments and solutions:
```
assignments/
├── assignment-01-descriptive/
│   ├── questions.pdf
│   ├── data.csv
│   └── solution.ipynb
```

### `/practice`
Practice problems:
- `descriptive-stats-practice.md`
- `probability-problems.md`
- `distribution-exercises.ipynb`

### `/resources`
- Statistical tables
- Formula sheets
- Dataset links
- Reference materials

## 🎯 Learning Tips

1. **Visualize everything** - Use plots to understand distributions
2. **Practice calculations** - Do both manual and Python calculations
3. **Understand concepts** - Don't just memorize formulas
4. **Use real data** - Apply concepts to actual datasets
5. **Connect to Python** - Implement every concept in code

## 📊 Key Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```

## 🔧 Common Operations

### Descriptive Statistics
```python
# Using NumPy
mean = np.mean(data)
median = np.median(data)
std = np.std(data, ddof=1)

# Using Pandas
df.describe()
df['column'].mean()
```

### Probability Distributions
```python
from scipy import stats

# Normal distribution
stats.norm.pdf(x, loc=mean, scale=std)
stats.norm.cdf(x, loc=mean, scale=std)

# Binomial distribution
stats.binom.pmf(k, n, p)
```

## 📈 Visualization Guide

### Histogram
```python
plt.hist(data, bins=30, edgecolor='black')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.title('Distribution')
```

### Box Plot
```python
plt.boxplot(data)
plt.ylabel('Values')
plt.title('Box Plot')
```

### Scatter Plot
```python
plt.scatter(x, y)
plt.xlabel('X variable')
plt.ylabel('Y variable')
```

## 🔗 Resources

### Online Resources
- [Khan Academy Statistics](https://www.khanacademy.org/math/statistics-probability)
- [StatQuest Videos](https://www.youtube.com/c/joshstarmer)
- [Seeing Theory - Visual Statistics](https://seeing-theory.brown.edu/)

### Books
- *Statistics* by David Freedman
- *The Art of Statistics* by David Spiegelhalter
- *Practical Statistics for Data Scientists*

### Datasets
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/)
- [Data.gov](https://data.gov/)

## 📝 Formula Quick Reference

### Measures of Central Tendency
```
Mean (μ) = Σx / n
Median = Middle value when sorted
Mode = Most frequent value
```

### Measures of Dispersion
```
Variance (σ²) = Σ(x - μ)² / n
Standard Deviation (σ) = √Variance
IQR = Q3 - Q1
```

### Probability
```
P(A) = Number of favorable outcomes / Total outcomes
P(A|B) = P(A ∩ B) / P(B)
```

### Normal Distribution
```
Z = (X - μ) / σ
```

## ✅ Progress Tracker

### Concepts
- [ ] Descriptive Statistics
- [ ] Measures of Central Tendency
- [ ] Measures of Dispersion
- [ ] Data Visualization
- [ ] Probability Basics
- [ ] Conditional Probability
- [ ] Discrete Distributions
- [ ] Continuous Distributions
- [ ] Central Limit Theorem
- [ ] Confidence Intervals

### Skills
- [ ] Calculate mean, median, mode
- [ ] Create histograms and box plots
- [ ] Work with probability problems
- [ ] Use Python for statistical analysis
- [ ] Interpret correlation
- [ ] Apply distributions to real problems

## 🎲 Practice Problems

### Easy
1. Calculate mean, median, mode of: [5, 10, 15, 10, 20, 25, 10]
2. Find probability of getting heads in 3 coin flips
3. Create a histogram for given data

### Medium
1. Calculate confidence interval for sample mean
2. Test if data follows normal distribution
3. Compare two distributions

### Hard
1. Apply Bayes' theorem to real-world scenario
2. Hypothesis testing for two groups
3. Analyze correlation between multiple variables

---

**Last Updated**: November 14, 2025
