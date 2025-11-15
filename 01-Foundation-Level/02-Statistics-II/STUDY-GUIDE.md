# BSMA1004: Statistics for Data Science II - Study Guide

**Course ID:** BSMA1004  
**Credits:** 4  
**Duration:** 12 weeks  
**Instructor:** Andrew Thangaraj  
**Prerequisites:** BSMA1002 (Statistics I), BSMA1001 (Mathematics I)  
**Co-requisites:** BSMA1003 (Mathematics II)

## 📚 Course Overview

Building on Statistics I, this course delves into advanced statistical concepts including multiple random variables, estimation theory, hypothesis testing, and regression analysis. These are critical tools for data science and statistical inference.

## 🎯 Learning Objectives

By the end of this course, you will be able to:
- Work with multiple random variables and joint distributions
- Calculate expectations, variance, and covariance
- Apply estimation techniques (point and interval estimation)
- Conduct hypothesis tests for means and variances
- Build and interpret simple linear regression models
- Validate statistical assumptions
- Apply inferential statistics to real-world data science problems

## 📖 Reference Materials

**Required Books (Available for Download):**
- **Joint Discrete Distributions (Vol 1)** - [Download from course page](https://drive.google.com/file/d/1AJ5KDvLDW7GKVjYK6I0fZ-fOi_7gcCsA/view)
- **Joint Continuous Distributions (Vol 2)** - [Download from course page](https://drive.google.com/file/d/1rryML2p2hKxde2Hc2LjUECQGPpMmJlnf/view)

**Prescribed Textbook:**
- **Probability and Statistics with Examples using R** by Siva Athreya, Deepayan Sarkar, and Steve Tanner

**Video Lectures:** [YouTube Playlist](https://www.youtube.com/playlist?list=PLZ2ps__7DhBbLZ6RdNTIXvFdaMpvqagt0)

---

## 📅 Week-by-Week Breakdown

### Week 1: Two Random Variables

**Topics Covered:**
- Joint probability distributions
- Marginal distributions
- Joint probability mass functions (discrete case)
- Contingency tables
- Understanding dependence between variables
- Examples with dice, cards, and real data

**Learning Activities:**
1. **Read:** Joint Discrete Distributions Vol 1, Chapter 1
2. **Watch:** Week 1 video lectures
3. **Practice:** Joint distribution problems
4. **Code:** Create and visualize joint distributions

**Key Concepts:**
- **Joint Distribution:** Probability distribution of two or more random variables
- **Marginal Distribution:** Distribution of one variable ignoring others
- **P(X=x, Y=y):** Probability that X=x AND Y=y simultaneously

**Practice Problems:**
- Create joint probability tables
- Calculate marginal distributions from joint distributions
- Verify that probabilities sum to 1
- Interpret contingency tables

**Python Applications:**
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Joint probability distribution (discrete)
# Example: Two dice rolls
def create_joint_dist_two_dice():
    joint_prob = np.zeros((6, 6))
    for i in range(6):
        for j in range(6):
            joint_prob[i, j] = 1/36
    return joint_prob

joint_dist = create_joint_dist_two_dice()

# Visualize as heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(joint_dist, annot=True, fmt='.4f', cmap='YlOrRd',
            xticklabels=range(1,7), yticklabels=range(1,7))
plt.xlabel('Die 2')
plt.ylabel('Die 1')
plt.title('Joint Probability Distribution: Two Dice')
plt.show()

# Marginal distributions
marginal_x = joint_dist.sum(axis=1)  # Sum across columns
marginal_y = joint_dist.sum(axis=0)  # Sum across rows

print(f"Marginal for X: {marginal_x}")
print(f"Marginal for Y: {marginal_y}")

# Conditional distribution P(Y|X=x)
def conditional_dist(joint, x_value):
    """Calculate P(Y|X=x)"""
    marginal_x = joint.sum(axis=1)
    return joint[x_value, :] / marginal_x[x_value]

# P(Y | X=3) - probability of die 2 given die 1 shows 4
cond_dist = conditional_dist(joint_dist, 3)
print(f"P(Y | X=4): {cond_dist}")
```

**Data Science Connection:** Joint distributions model relationships between features in datasets - crucial for understanding dependencies in multivariate data.

**Weekly Notebook:** `week-01-two-random-variables.ipynb`

---

### Week 2: Multiple Random Variables and Independence

**Topics Covered:**
- Extending to n random variables
- Joint distributions for multiple variables
- Independence of random variables
- Testing for independence
- Conditional independence
- Functions of random variables
- Transformations

**Learning Activities:**
1. **Read:** Joint Discrete Distributions Vol 1, Chapter 2-3
2. **Watch:** Week 2 video lectures
3. **Practice:** Independence problems
4. **Code:** Test independence in datasets

**Key Concepts:**
- **Independence:** P(X=x, Y=y) = P(X=x) × P(Y=y) for all x, y
- **Conditional Independence:** X and Y independent given Z
- **Transformation:** Finding distribution of g(X, Y)

**Practice Problems:**
- Verify independence mathematically
- Find distributions of sums and products
- Apply transformations to random variables

**Python Applications:**
```python
from scipy import stats

# Test independence using chi-square test
def test_independence(contingency_table):
    """
    Chi-square test for independence
    H0: Variables are independent
    H1: Variables are dependent
    """
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
    
    print(f"Chi-square statistic: {chi2:.4f}")
    print(f"P-value: {p_value:.4f}")
    print(f"Degrees of freedom: {dof}")
    
    if p_value < 0.05:
        print("Reject H0: Variables are DEPENDENT")
    else:
        print("Fail to reject H0: Variables appear INDEPENDENT")
    
    return chi2, p_value

# Example: Survey data - Education level vs Income level
data = pd.DataFrame({
    'Education': ['High School']*50 + ['Bachelor']*60 + ['Master']*40,
    'Income': ['Low']*30 + ['Medium']*15 + ['High']*5 +  # HS
              ['Low']*20 + ['Medium']*25 + ['High']*15 +  # Bach
              ['Low']*5 + ['Medium']*15 + ['High']*20     # Master
})

# Create contingency table
contingency = pd.crosstab(data['Education'], data['Income'])
print(contingency)

# Test independence
test_independence(contingency)

# Visualize
plt.figure(figsize=(10, 6))
sns.heatmap(contingency, annot=True, fmt='d', cmap='Blues')
plt.title('Contingency Table: Education vs Income')
plt.show()

# Simulate independent vs dependent variables
n = 1000
# Independent
X_indep = np.random.normal(0, 1, n)
Y_indep = np.random.normal(0, 1, n)

# Dependent
X_dep = np.random.normal(0, 1, n)
Y_dep = 2*X_dep + np.random.normal(0, 0.5, n)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X_indep, Y_indep, alpha=0.5)
axes[0].set_title('Independent Variables')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

axes[1].scatter(X_dep, Y_dep, alpha=0.5)
axes[1].set_title('Dependent Variables')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')
plt.tight_layout()
plt.show()
```

**Data Science Connection:** Testing independence is crucial for feature selection and understanding multicollinearity in machine learning models.

**Weekly Notebook:** `week-02-independence.ipynb`

---

### Week 3: Expectations and Variance

**Topics Covered:**
- Expected value of random variables
- Expected value of functions of random variables
- Variance and standard deviation revisited
- Covariance between two variables
- Correlation coefficient
- Properties of expectation and variance
- Conditional expectation
- Law of total expectation

**Learning Activities:**
1. **Read:** Joint Continuous Distributions Vol 2, Chapter 1
2. **Watch:** Week 3 video lectures
3. **Practice:** Calculate expectations and variances
4. **Code:** Compute and visualize covariance

**Key Concepts:**
- **E[X]:** Expected value (mean) of X
- **Var(X):** E[(X - E[X])²] = E[X²] - (E[X])²
- **Cov(X,Y):** E[(X - E[X])(Y - E[Y])]
- **Corr(X,Y):** Cov(X,Y) / (σ_X × σ_Y)

**Practice Problems:**
- Calculate expectations from probability distributions
- Compute variance using different formulas
- Find covariance and correlation
- Apply law of total expectation

**Python Applications:**
```python
# Expected value and variance for discrete distribution
values = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

# Expected value
E_X = np.sum(values * probabilities)
print(f"E[X] = {E_X:.4f}")

# Variance: E[X²] - (E[X])²
E_X2 = np.sum(values**2 * probabilities)
Var_X = E_X2 - E_X**2
print(f"Var(X) = {Var_X:.4f}")
print(f"Std(X) = {np.sqrt(Var_X):.4f}")

# Covariance and correlation for sample data
np.random.seed(42)
X = np.random.normal(10, 2, 1000)
Y = 3*X + np.random.normal(0, 5, 1000)  # Y depends on X

# Covariance
cov_matrix = np.cov(X, Y)
print(f"\nCovariance matrix:\n{cov_matrix}")
print(f"Cov(X,Y) = {cov_matrix[0,1]:.4f}")

# Correlation
corr_matrix = np.corrcoef(X, Y)
print(f"\nCorrelation matrix:\n{corr_matrix}")
print(f"Corr(X,Y) = {corr_matrix[0,1]:.4f}")

# Visualize relationship
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Scatter plot
axes[0].scatter(X, Y, alpha=0.5)
axes[0].set_title(f'Scatter Plot\nCorr = {corr_matrix[0,1]:.3f}')
axes[0].set_xlabel('X')
axes[0].set_ylabel('Y')

# Joint distribution (2D histogram)
axes[1].hist2d(X, Y, bins=30, cmap='YlOrRd')
axes[1].set_title('Joint Distribution')
axes[1].set_xlabel('X')
axes[1].set_ylabel('Y')

# Marginal distributions
axes[2].hist(X, bins=30, alpha=0.5, label='X', density=True)
axes[2].hist(Y, bins=30, alpha=0.5, label='Y', density=True)
axes[2].set_title('Marginal Distributions')
axes[2].legend()

plt.tight_layout()
plt.show()

# Properties of expectation
# E[aX + bY] = aE[X] + bE[Y]
a, b = 2, 3
E_aX_plus_bY = a * X.mean() + b * Y.mean()
print(f"\nE[{a}X + {b}Y] = {E_aX_plus_bY:.4f}")
print(f"Actual mean of {a}X + {b}Y: {(a*X + b*Y).mean():.4f}")

# Var(aX) = a²Var(X)
Var_aX = a**2 * X.var()
print(f"\nVar({a}X) = {a}² × Var(X) = {Var_aX:.4f}")
print(f"Actual variance: {(a*X).var():.4f}")
```

**Data Science Connection:** Understanding expectations and covariance is essential for portfolio optimization, risk analysis, and multivariate statistical methods.

**Weekly Notebook:** `week-03-expectations-variance.ipynb`

---

### Week 4: Continuous Random Variables (Review & Extension)

**Topics Covered:**
- Continuous random variables revisited
- Joint probability density functions
- Marginal densities from joint densities
- Conditional densities
- Transformations of continuous random variables
- Bivariate normal distribution

**Learning Activities:**
1. **Read:** Joint Continuous Distributions Vol 2, Chapter 2-3
2. **Watch:** Week 4 video lectures
3. **Practice:** Joint density problems
4. **Code:** Simulate and visualize continuous distributions

**Key Concepts:**
- **Joint PDF:** f(x,y) such that P(a≤X≤b, c≤Y≤d) = ∫∫ f(x,y) dx dy
- **Marginal PDF:** f_X(x) = ∫ f(x,y) dy
- **Conditional PDF:** f(y|x) = f(x,y) / f_X(x)

**Python Applications:**
```python
from scipy.stats import multivariate_normal

# Bivariate normal distribution
mean = [0, 0]
cov = [[1, 0.8],   # Variance of X, Covariance
       [0.8, 1]]    # Covariance, Variance of Y

# Create distribution
rv = multivariate_normal(mean, cov)

# Generate samples
n_samples = 1000
samples = rv.rvs(n_samples)

# Visualize
fig = plt.figure(figsize=(15, 5))

# 2D scatter
ax1 = fig.add_subplot(131)
ax1.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
ax1.set_title('Bivariate Normal Samples')
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.grid(True, alpha=0.3)

# 2D density (contour plot)
ax2 = fig.add_subplot(132)
x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x, y))
ax2.contourf(x, y, rv.pdf(pos), levels=20, cmap='viridis')
ax2.set_title('Joint PDF (Contour)')
ax2.set_xlabel('X')
ax2.set_ylabel('Y')

# 3D surface
from mpl_toolkits.mplot3d import Axes3D
ax3 = fig.add_subplot(133, projection='3d')
ax3.plot_surface(x, y, rv.pdf(pos), cmap='viridis', alpha=0.8)
ax3.set_title('Joint PDF (3D)')
ax3.set_xlabel('X')
ax3.set_ylabel('Y')
ax3.set_zlabel('Density')

plt.tight_layout()
plt.show()

# Marginal distributions
print("Marginal distributions are both N(0,1)")
print(f"Mean of X samples: {samples[:, 0].mean():.3f}")
print(f"Mean of Y samples: {samples[:, 1].mean():.3f}")
print(f"Std of X samples: {samples[:, 0].std():.3f}")
print(f"Std of Y samples: {samples[:, 1].std():.3f}")

# Correlation
sample_corr = np.corrcoef(samples[:, 0], samples[:, 1])[0, 1]
print(f"\nTrue correlation: 0.8")
print(f"Sample correlation: {sample_corr:.3f}")
```

**Data Science Connection:** Multivariate normal distribution is foundational for many ML algorithms including Gaussian Naive Bayes and Gaussian Processes.

**Weekly Notebook:** `week-04-continuous-joint-distributions.ipynb`

---

### Week 5-6: Sampling Distributions and Central Limit Theorem

**Topics Covered:**
- Sampling distributions
- Distribution of sample mean
- Central Limit Theorem (detailed)
- Distribution of sample variance
- Chi-square distribution
- Student's t-distribution
- F-distribution
- Applications to inference

**Learning Activities:**
1. **Read:** Prescribed textbook, Chapters on sampling distributions
2. **Watch:** Weeks 5-6 video lectures
3. **Practice:** CLT problems
4. **Code:** Demonstrate CLT with simulations

**Key Concepts:**
- **Sampling Distribution:** Distribution of a statistic across many samples
- **CLT:** Sample means approach normal distribution as n→∞
- **Chi-square:** Distribution of sum of squared standard normals
- **t-distribution:** Used when population variance unknown

**Python Applications:**
```python
# Comprehensive CLT demonstration
def demonstrate_clt(population_dist, population_params, 
                    sample_sizes, n_samples=10000):
    """
    Demonstrate Central Limit Theorem
    """
    fig, axes = plt.subplots(2, len(sample_sizes), figsize=(16, 8))
    
    # Generate population
    if population_dist == 'exponential':
        population = np.random.exponential(population_params, 100000)
        title_prefix = f"Exponential(λ={population_params})"
    elif population_dist == 'uniform':
        population = np.random.uniform(*population_params, 100000)
        title_prefix = f"Uniform{population_params}"
    elif population_dist == 'bimodal':
        pop1 = np.random.normal(0, 1, 50000)
        pop2 = np.random.normal(5, 1, 50000)
        population = np.concatenate([pop1, pop2])
        title_prefix = "Bimodal"
    
    pop_mean = population.mean()
    pop_std = population.std()
    
    for idx, n in enumerate(sample_sizes):
        # Generate sample means
        sample_means = []
        for _ in range(n_samples):
            sample = np.random.choice(population, size=n, replace=True)
            sample_means.append(sample.mean())
        
        sample_means = np.array(sample_means)
        
        # Expected values for sampling distribution
        expected_mean = pop_mean
        expected_std = pop_std / np.sqrt(n)
        
        # Population distribution (only show once)
        if idx == 0:
            axes[0, idx].hist(population, bins=50, density=True, 
                            alpha=0.7, edgecolor='black')
            axes[0, idx].set_title(f'Population: {title_prefix}')
            axes[0, idx].set_ylabel('Density')
        else:
            axes[0, idx].text(0.5, 0.5, f'Same\nPopulation', 
                            ha='center', va='center',
                            transform=axes[0, idx].transAxes, fontsize=14)
            axes[0, idx].set_xticks([])
            axes[0, idx].set_yticks([])
        
        # Sampling distribution of mean
        axes[1, idx].hist(sample_means, bins=50, density=True,
                         alpha=0.7, edgecolor='black', label='Sample means')
        
        # Overlay theoretical normal
        x = np.linspace(sample_means.min(), sample_means.max(), 100)
        axes[1, idx].plot(x, stats.norm(expected_mean, expected_std).pdf(x),
                         'r-', linewidth=2, label='Theoretical N')
        
        axes[1, idx].set_title(f'Sampling Dist: n={n}')
        axes[1, idx].set_xlabel('Sample Mean')
        if idx == 0:
            axes[1, idx].set_ylabel('Density')
        axes[1, idx].legend(fontsize=8)
        
        # Print statistics
        print(f"\nSample size n={n}:")
        print(f"  Expected: μ={expected_mean:.3f}, σ={expected_std:.3f}")
        print(f"  Actual:   μ={sample_means.mean():.3f}, σ={sample_means.std():.3f}")
    
    plt.tight_layout()
    plt.show()

# Demonstrate with exponential distribution (highly skewed)
demonstrate_clt('exponential', 2, [5, 10, 30, 100])

# t-distribution vs normal distribution
degrees_of_freedom = [1, 3, 10, 30]
x = np.linspace(-4, 4, 1000)

plt.figure(figsize=(12, 6))
plt.plot(x, stats.norm(0, 1).pdf(x), 'k-', linewidth=2, label='Normal')

for df in degrees_of_freedom:
    plt.plot(x, stats.t(df).pdf(x), label=f't(df={df})')

plt.title('t-distribution vs Normal Distribution')
plt.xlabel('x')
plt.ylabel('Density')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nNote: t-distribution has heavier tails than normal")
print("As df increases, t-distribution approaches normal distribution")
```

**Data Science Connection:** CLT justifies using normal distribution for inference in many scenarios, even when underlying data isn't normal.

**Weekly Notebook:** `week-05-06-clt-sampling-distributions.ipynb`

---

### Week 7-8: Point and Interval Estimation

**Topics Covered:**
- Point estimation
- Properties of estimators (unbiased, consistent, efficient)
- Method of moments
- Maximum likelihood estimation (MLE)
- Confidence intervals for mean
- Confidence intervals for proportion
- Confidence intervals for variance
- Margin of error and sample size determination

**Learning Activities:**
1. **Read:** Prescribed textbook, Chapters on estimation
2. **Watch:** Weeks 7-8 video lectures
3. **Practice:** Calculate estimators and confidence intervals
4. **Code:** Implement MLE and bootstrap confidence intervals

**Key Concepts:**
- **Unbiased Estimator:** E[θ̂] = θ
- **MLE:** Value of parameter that maximizes likelihood
- **Confidence Interval:** Range with specified probability of containing parameter
- **95% CI:** We're 95% confident the parameter lies in this interval

**Python Applications:**
```python
# Maximum Likelihood Estimation
from scipy.optimize import minimize

def mle_normal(data):
    """
    Find MLE for normal distribution parameters
    """
    def neg_log_likelihood(params):
        mu, sigma = params
        if sigma <= 0:
            return np.inf
        n = len(data)
        return n/2 * np.log(2*np.pi*sigma**2) + np.sum((data - mu)**2) / (2*sigma**2)
    
    # Initial guess
    initial_guess = [data.mean(), data.std()]
    
    # Optimize
    result = minimize(neg_log_likelihood, initial_guess, method='Nelder-Mead')
    
    return result.x

# Generate data
np.random.seed(42)
true_mu, true_sigma = 10, 2
data = np.random.normal(true_mu, true_sigma, 100)

# MLE estimates
mle_mu, mle_sigma = mle_normal(data)
print(f"True parameters: μ={true_mu}, σ={true_sigma}")
print(f"MLE estimates: μ̂={mle_mu:.3f}, σ̂={mle_sigma:.3f}")
print(f"Sample estimates: μ̂={data.mean():.3f}, σ̂={data.std():.3f}")

# Confidence Intervals
def confidence_interval_mean(data, confidence=0.95):
    """
    Calculate confidence interval for mean
    """
    n = len(data)
    mean = np.mean(data)
    std_error = stats.sem(data)
    
    # Use t-distribution
    df = n - 1
    t_critical = stats.t.ppf((1 + confidence) / 2, df)
    
    margin_of_error = t_critical * std_error
    ci = (mean - margin_of_error, mean + margin_of_error)
    
    return ci, margin_of_error

# Calculate 95% CI
ci_95, margin = confidence_interval_mean(data, 0.95)
print(f"\n95% Confidence Interval for mean: ({ci_95[0]:.3f}, {ci_95[1]:.3f})")
print(f"Margin of error: ±{margin:.3f}")
print(f"True mean {true_mu} is in the interval: {ci_95[0] <= true_mu <= ci_95[1]}")

# Demonstrate coverage of confidence intervals
def demonstrate_ci_coverage(true_mu, true_sigma, n, confidence, n_simulations=1000):
    """
    Show that confidence intervals contain true parameter at specified rate
    """
    contains_true = 0
    cis = []
    
    for _ in range(n_simulations):
        sample = np.random.normal(true_mu, true_sigma, n)
        ci, _ = confidence_interval_mean(sample, confidence)
        cis.append(ci)
        
        if ci[0] <= true_mu <= ci[1]:
            contains_true += 1
    
    coverage = contains_true / n_simulations
    
    # Visualize first 100 CIs
    plt.figure(figsize=(12, 6))
    for i in range(min(100, n_simulations)):
        color = 'green' if cis[i][0] <= true_mu <= cis[i][1] else 'red'
        plt.plot([cis[i][0], cis[i][1]], [i, i], color=color, alpha=0.5)
    
    plt.axvline(true_mu, color='blue', linewidth=2, label=f'True μ={true_mu}')
    plt.xlabel('Value')
    plt.ylabel('Sample number')
    plt.title(f'{confidence*100:.0f}% Confidence Intervals (Coverage: {coverage*100:.1f}%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"\nExpected coverage: {confidence*100:.1f}%")
    print(f"Actual coverage: {coverage*100:.1f}%")
    
    return coverage

# Test CI coverage
demonstrate_ci_coverage(10, 2, n=30, confidence=0.95, n_simulations=1000)

# Bootstrap confidence intervals
def bootstrap_ci(data, n_bootstrap=10000, confidence=0.95):
    """
    Calculate bootstrap confidence interval
    """
    bootstrap_means = []
    n = len(data)
    
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=n, replace=True)
        bootstrap_means.append(sample.mean())
    
    bootstrap_means = np.array(bootstrap_means)
    
    # Percentile method
    alpha = 1 - confidence
    lower = np.percentile(bootstrap_means, alpha/2 * 100)
    upper = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
    
    return (lower, upper), bootstrap_means

# Compare bootstrap CI with traditional CI
boot_ci, boot_means = bootstrap_ci(data)
print(f"\nTraditional 95% CI: ({ci_95[0]:.3f}, {ci_95[1]:.3f})")
print(f"Bootstrap 95% CI:   ({boot_ci[0]:.3f}, {boot_ci[1]:.3f})")

# Visualize bootstrap distribution
plt.figure(figsize=(10, 6))
plt.hist(boot_means, bins=50, density=True, alpha=0.7, edgecolor='black')
plt.axvline(boot_ci[0], color='r', linestyle='--', label='Bootstrap CI')
plt.axvline(boot_ci[1], color='r', linestyle='--')
plt.axvline(data.mean(), color='g', linewidth=2, label='Sample mean')
plt.axvline(true_mu, color='b', linewidth=2, label='True mean')
plt.xlabel('Bootstrap Sample Mean')
plt.ylabel('Density')
plt.title('Bootstrap Distribution of Sample Mean')
plt.legend()
plt.show()
```

**Data Science Connection:** Confidence intervals quantify uncertainty in model parameters, A/B test results, and predictions.

**Weekly Notebook:** `week-07-08-estimation.ipynb`

---

### Week 9-10: Hypothesis Testing

**Topics Covered:**
- Null and alternative hypotheses
- Type I and Type II errors
- Significance level and p-values
- One-sample t-test
- Two-sample t-test
- Paired t-test
- Tests for proportions
- Chi-square test for variance
- Power of a test

**Learning Activities:**
1. **Read:** Prescribed textbook, Chapters on hypothesis testing
2. **Watch:** Weeks 9-10 video lectures
3. **Practice:** Conduct various hypothesis tests
4. **Code:** Implement testing procedures

**Key Concepts:**
- **H₀:** Null hypothesis (status quo)
- **H₁:** Alternative hypothesis
- **Type I Error:** Rejecting true H₀ (false positive)
- **Type II Error:** Failing to reject false H₀ (false negative)
- **p-value:** Probability of observing data if H₀ is true

**Python Applications:**
```python
# One-sample t-test
def perform_one_sample_ttest(data, hypothesized_mean, alpha=0.05):
    """
    Test if sample mean differs from hypothesized value
    H0: μ = hypothesized_mean
    H1: μ ≠ hypothesized_mean (two-tailed)
    """
    t_statistic, p_value = stats.ttest_1samp(data, hypothesized_mean)
    
    print(f"One-Sample t-test")
    print(f"H0: μ = {hypothesized_mean}")
    print(f"Sample mean: {data.mean():.3f}")
    print(f"t-statistic: {t_statistic:.3f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < alpha:
        print(f"Reject H0 at α={alpha} (statistically significant)")
    else:
        print(f"Fail to reject H0 at α={alpha} (not significant)")
    
    return t_statistic, p_value

# Example: Test if average height is 170 cm
heights = np.random.normal(172, 8, 50)
perform_one_sample_ttest(heights, 170, alpha=0.05)

# Two-sample t-test
def perform_two_sample_ttest(group1, group2, alpha=0.05):
    """
    Test if two groups have different means
    H0: μ1 = μ2
    H1: μ1 ≠ μ2
    """
    # Test for equal variances first
    _, p_levene = stats.levene(group1, group2)
    equal_var = p_levene > 0.05
    
    # Perform t-test
    t_statistic, p_value = stats.ttest_ind(group1, group2, equal_var=equal_var)
    
    print(f"\nTwo-Sample t-test")
    print(f"H0: μ1 = μ2")
    print(f"Group 1 mean: {group1.mean():.3f}")
    print(f"Group 2 mean: {group2.mean():.3f}")
    print(f"Equal variances: {equal_var}")
    print(f"t-statistic: {t_statistic:.3f}")
    print(f"p-value: {p_value:.4f}")
    
    if p_value < alpha:
        print(f"Reject H0 at α={alpha} (groups differ significantly)")
    else:
        print(f"Fail to reject H0 at α={alpha} (no significant difference)")
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((group1.std()**2 + group2.std()**2) / 2)
    cohens_d = (group1.mean() - group2.mean()) / pooled_std
    print(f"Effect size (Cohen's d): {cohens_d:.3f}")
    
    return t_statistic, p_value

# Example: A/B test
control_group = np.random.normal(100, 15, 50)
treatment_group = np.random.normal(105, 15, 50)
perform_two_sample_ttest(control_group, treatment_group)

# Visualize hypothesis test
def visualize_hypothesis_test(data, hypothesized_mean, alpha=0.05):
    """Visualize one-sample t-test"""
    n = len(data)
    sample_mean = data.mean()
    std_error = stats.sem(data)
    
    # t-statistic
    t_stat = (sample_mean - hypothesized_mean) / std_error
    df = n - 1
    
    # Critical values
    t_crit = stats.t.ppf(1 - alpha/2, df)
    
    # Plot t-distribution
    x = np.linspace(-4, 4, 1000)
    y = stats.t.pdf(x, df)
    
    plt.figure(figsize=(12, 6))
    plt.plot(x, y, 'b-', linewidth=2, label='t-distribution')
    
    # Rejection regions
    rejection_left = x[x < -t_crit]
    rejection_right = x[x > t_crit]
    plt.fill_between(rejection_left, 0, stats.t.pdf(rejection_left, df),
                     alpha=0.3, color='red', label='Rejection region')
    plt.fill_between(rejection_right, 0, stats.t.pdf(rejection_right, df),
                     alpha=0.3, color='red')
    
    # Test statistic
    plt.axvline(t_stat, color='green', linewidth=2, 
                label=f't-statistic = {t_stat:.3f}')
    plt.axvline(-t_crit, color='red', linestyle='--', 
                label=f'Critical value = ±{t_crit:.3f}')
    plt.axvline(t_crit, color='red', linestyle='--')
    
    plt.xlabel('t-value')
    plt.ylabel('Density')
    plt.title(f'Hypothesis Test Visualization (α={alpha})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print(f"\nTest statistic: {t_stat:.3f}")
    print(f"Critical value: ±{t_crit:.3f}")
    print(f"Reject H0: {abs(t_stat) > t_crit}")

visualize_hypothesis_test(heights, 170)

# Power analysis
def calculate_power(effect_size, n, alpha=0.05):
    """
    Calculate statistical power
    Power = P(Reject H0 | H0 is false)
    """
    from statsmodels.stats.power import ttest_power
    
    power = ttest_power(effect_size, n, alpha)
    return power

# Plot power curve
effect_sizes = np.linspace(0.1, 1.5, 50)
sample_sizes = [10, 30, 50, 100]

plt.figure(figsize=(10, 6))
for n in sample_sizes:
    powers = [calculate_power(es, n) for es in effect_sizes]
    plt.plot(effect_sizes, powers, label=f'n={n}', linewidth=2)

plt.axhline(y=0.8, color='r', linestyle='--', label='80% power')
plt.xlabel('Effect Size (Cohen\'s d)')
plt.ylabel('Statistical Power')
plt.title('Power Analysis for t-test')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("\nInterpretation:")
print("- Larger sample size → Higher power")
print("- Larger effect size → Higher power")
print("- 80% power is often considered adequate")
```

**Data Science Connection:** Hypothesis testing is fundamental for A/B testing, feature importance, and validating model improvements.

**Weekly Notebook:** `week-09-10-hypothesis-testing.ipynb`

---

### Week 11-12: Simple Linear Regression

**Topics Covered:**
- Simple linear regression model
- Least squares estimation
- Assumptions of linear regression
- R-squared and model fit
- Hypothesis tests for regression coefficients
- Confidence and prediction intervals
- Residual analysis
- Model diagnostics

**Learning Activities:**
1. **Read:** Prescribed textbook, Regression chapters
2. **Watch:** Weeks 11-12 video lectures
3. **Practice:** Build regression models
4. **Code:** Comprehensive regression analysis

**Key Concepts:**
- **Linear Model:** Y = β₀ + β₁X + ε
- **Least Squares:** Minimize Σ(yᵢ - ŷᵢ)²
- **R²:** Proportion of variance explained
- **Residuals:** Observed - Predicted values

**Python Applications:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Generate sample data
np.random.seed(42)
n = 100
X = np.random.uniform(0, 10, n)
true_slope = 2.5
true_intercept = 5
noise = np.random.normal(0, 2, n)
y = true_intercept + true_slope * X + noise

# Fit linear regression
X_reshaped = X.reshape(-1, 1)
model = LinearRegression()
model.fit(X_reshaped, y)

# Predictions
y_pred = model.predict(X_reshaped)

# Coefficients
print(f"True coefficients: β₀={true_intercept}, β₁={true_slope}")
print(f"Estimated coefficients: β₀={model.intercept_:.3f}, β₁={model.coef_[0]:.3f}")

# Model evaluation
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
print(f"\nR² = {r2:.4f}")
print(f"RMSE = {rmse:.4f}")

# Comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Scatter plot with regression line
axes[0, 0].scatter(X, y, alpha=0.6, label='Data')
axes[0, 0].plot(X, y_pred, 'r-', linewidth=2, label='Regression line')
axes[0, 0].set_xlabel('X')
axes[0, 0].set_ylabel('Y')
axes[0, 0].set_title(f'Linear Regression (R² = {r2:.3f})')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 2. Residuals vs Fitted
residuals = y - y_pred
axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
axes[0, 1].axhline(y=0, color='r', linestyle='--')
axes[0, 1].set_xlabel('Fitted values')
axes[0, 1].set_ylabel('Residuals')
axes[0, 1].set_title('Residuals vs Fitted')
axes[0, 1].grid(True, alpha=0.3)

# 3. Q-Q plot (check normality of residuals)
from scipy import stats
stats.probplot(residuals, dist="norm", plot=axes[1, 0])
axes[1, 0].set_title('Q-Q Plot (Normality Check)')
axes[1, 0].grid(True, alpha=0.3)

# 4. Histogram of residuals
axes[1, 1].hist(residuals, bins=20, edgecolor='black', alpha=0.7, density=True)
x_hist = np.linspace(residuals.min(), residuals.max(), 100)
axes[1, 1].plot(x_hist, stats.norm(0, residuals.std()).pdf(x_hist),
               'r-', linewidth=2, label='Normal')
axes[1, 1].set_xlabel('Residuals')
axes[1, 1].set_ylabel('Density')
axes[1, 1].set_title('Distribution of Residuals')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Statistical inference for coefficients
from scipy import stats as sp_stats

# Manual calculation of standard errors
n = len(X)
X_with_intercept = np.column_stack([np.ones(n), X])
residuals = y - y_pred
residual_var = np.sum(residuals**2) / (n - 2)

# Variance-covariance matrix
X_transpose_X_inv = np.linalg.inv(X_with_intercept.T @ X_with_intercept)
var_coef = residual_var * X_transpose_X_inv

# Standard errors
se_intercept = np.sqrt(var_coef[0, 0])
se_slope = np.sqrt(var_coef[1, 1])

# t-statistics
t_intercept = model.intercept_ / se_intercept
t_slope = model.coef_[0] / se_slope

# p-values (two-tailed)
p_intercept = 2 * (1 - sp_stats.t.cdf(abs(t_intercept), n-2))
p_slope = 2 * (1 - sp_stats.t.cdf(abs(t_slope), n-2))

print(f"\nHypothesis test for slope:")
print(f"H0: β₁ = 0 (no relationship)")
print(f"t-statistic: {t_slope:.3f}")
print(f"p-value: {p_slope:.6f}")
if p_slope < 0.05:
    print("Reject H0: Slope is significantly different from zero")
else:
    print("Fail to reject H0: No significant relationship")

# Confidence intervals for coefficients
alpha = 0.05
t_crit = sp_stats.t.ppf(1 - alpha/2, n-2)

ci_slope = (model.coef_[0] - t_crit * se_slope,
            model.coef_[0] + t_crit * se_slope)
print(f"\n95% CI for slope: ({ci_slope[0]:.3f}, {ci_slope[1]:.3f})")

# Prediction intervals
def prediction_interval(X_new, confidence=0.95):
    """Calculate prediction interval for new observations"""
    X_new = np.array(X_new).reshape(-1, 1)
    y_pred = model.predict(X_new)
    
    # Standard error of prediction
    X_new_with_intercept = np.column_stack([np.ones(len(X_new)), X_new])
    se_pred = np.sqrt(residual_var * (1 + np.sum((X_new_with_intercept @ X_transpose_X_inv) * X_new_with_intercept, axis=1)))
    
    # t-critical value
    t_crit = sp_stats.t.ppf((1 + confidence) / 2, n-2)
    
    # Prediction interval
    lower = y_pred - t_crit * se_pred
    upper = y_pred + t_crit * se_pred
    
    return y_pred, lower, upper

# Visualize confidence and prediction intervals
X_range = np.linspace(X.min(), X.max(), 100)
y_pred_range, lower_pred, upper_pred = prediction_interval(X_range)

plt.figure(figsize=(12, 6))
plt.scatter(X, y, alpha=0.6, label='Data')
plt.plot(X_range, y_pred_range, 'r-', linewidth=2, label='Regression line')
plt.fill_between(X_range, lower_pred, upper_pred, alpha=0.2, 
                 color='red', label='95% Prediction interval')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression with Prediction Interval')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Comprehensive regression summary (like R's summary)
def regression_summary(X, y, model):
    """Print comprehensive regression summary"""
    n = len(X)
    y_pred = model.predict(X)
    residuals = y - y_pred
    
    # R-squared
    ss_total = np.sum((y - y.mean())**2)
    ss_residual = np.sum(residuals**2)
    r2 = 1 - (ss_residual / ss_total)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - 2)
    
    print("=" * 60)
    print("LINEAR REGRESSION SUMMARY")
    print("=" * 60)
    print(f"\nDependent variable: Y")
    print(f"Number of observations: {n}")
    print(f"\nCoefficients:")
    print(f"  Intercept: {model.intercept_:.4f}")
    print(f"  Slope:     {model.coef_[0]:.4f}")
    print(f"\nModel fit:")
    print(f"  R-squared:          {r2:.4f}")
    print(f"  Adjusted R-squared: {adj_r2:.4f}")
    print(f"  RMSE:               {np.sqrt(mean_squared_error(y, y_pred)):.4f}")
    print(f"\nF-statistic: {(r2 / 1) / ((1 - r2) / (n - 2)):.2f}")
    print("=" * 60)

regression_summary(X_reshaped, y, model)
```

**Data Science Connection:** Linear regression is one of the most widely used ML algorithms, foundation for many advanced techniques, and essential for causal inference.

**Weekly Notebook:** `week-11-12-linear-regression.ipynb`

---

## 🎯 Assessment Structure

- **Weekly Online Assignments:** 10-20%
- **Quiz 1 (In-person):** 15-20%
- **Quiz 2 (In-person):** 15-20%
- **End Term Exam (In-person):** 50-60%

**Passing Grade:** 40% overall with at least 40% in end-term exam

---

## 💡 Study Tips

1. **Master the Theory:** Understand WHY, not just HOW
2. **Practice Interpretation:** Focus on explaining results in plain English
3. **Code Everything:** Implement concepts from scratch before using libraries
4. **Real Data:** Apply techniques to real datasets from Kaggle
5. **Connect Concepts:** See how estimation, testing, and regression relate
6. **Simulation:** Verify theoretical results with simulations
7. **Review Stats I:** This course builds heavily on Stats I concepts
8. **Study Groups:** Discuss interpretations of p-values and confidence intervals

---

## 🔗 Important Links

- **Course Page:** https://study.iitm.ac.in/ds/course_pages/BSMA1004.html
- **Video Lectures:** [YouTube Playlist](https://www.youtube.com/playlist?list=PLZ2ps__7DhBbLZ6RdNTIXvFdaMpvqagt0)
- **Instructor Website:** [Prof. Andrew Thangaraj](http://www.ee.iitm.ac.in/andrew/)

---

## 📚 Additional Resources

- **OpenIntro Statistics** - Free comprehensive textbook
- **Statistical Inference** by Casella & Berger - Advanced reference
- **StatQuest YouTube Channel** - Excellent explanations
- **Seeing Theory** - Visual introduction to probability and statistics

---

**Remember:** Statistical inference is the bridge between data and decisions. Master these concepts to make data-driven conclusions with confidence! 🎯
