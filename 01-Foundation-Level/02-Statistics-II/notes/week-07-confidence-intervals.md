# Week 07: Confidence Intervals

**Course**: BSST1002 - Statistics II
**Level**: Foundation Level

## Learning Objectives
- Understand confidence intervals as a measure of estimation uncertainty
- Apply t-distribution when population variance is unknown
- Construct confidence intervals for means and proportions
- Determine appropriate sample sizes for desired precision

---

## 1. Confidence Intervals for Means

### 1.1 Theory

A **confidence interval** quantifies the uncertainty in our point estimate by providing a range of plausible values for the population parameter.

**Key Interpretation**: A 95% confidence interval means that if we repeated the sampling process many times, approximately 95% of the constructed intervals would contain the true population parameter.

> **Important**: The confidence level refers to the procedure, not the probability that any single interval contains the parameter.

### 1.2 Mathematical Definition

**Case 1: Known Population Variance (σ known)**

$$\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

Where:
- $\bar{X}$ = sample mean
- $z_{\alpha/2}$ = critical value from standard normal distribution
- $\sigma$ = population standard deviation
- $n$ = sample size

**Case 2: Unknown Population Variance (σ unknown)**

$$\bar{X} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

Where:
- $s$ = sample standard deviation
- $t_{\alpha/2, n-1}$ = critical value from t-distribution with (n-1) degrees of freedom

### 1.3 Common Confidence Levels

| Confidence Level | $\alpha$ | $z_{\alpha/2}$ |
|-----------------|----------|----------------|
| 90% | 0.10 | 1.645 |
| 95% | 0.05 | 1.960 |
| 99% | 0.01 | 2.576 |

### 1.4 Margin of Error

The **margin of error** (E) is the half-width of the confidence interval:

$$E = z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}} \quad \text{or} \quad E = t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$$

### 1.5 Supply Chain Application

**Retail Context**:
- **Safety Stock Planning**: A 95% CI for average daily demand helps determine appropriate safety stock levels
- **Lead Time Analysis**: CI for supplier lead time guides reorder point decisions
- **Capacity Planning**: CI for order processing time informs staffing requirements

---

## 2. The t-Distribution

### 2.1 Theory

When the population standard deviation σ is **unknown** and must be estimated from sample data, we use the **t-distribution** instead of the normal distribution.

**Key Properties**:
- Symmetric and bell-shaped (like Normal)
- Heavier tails than Normal distribution
- Shape depends on degrees of freedom (df = n - 1)
- More uncertainty due to estimating σ with s

### 2.2 Mathematical Definition

The **t-statistic** follows a t-distribution:

$$T = \frac{\bar{X} - \mu}{S/\sqrt{n}} \sim t_{n-1}$$

Where:
- $\bar{X}$ = sample mean
- $\mu$ = population mean
- $S$ = sample standard deviation
- $n$ = sample size
- $t_{n-1}$ = t-distribution with (n-1) degrees of freedom

### 2.3 Properties

1. **Convergence**: As $n \to \infty$, $t_{n-1} \to N(0,1)$
2. **Heavier Tails**: More probability in the tails accounts for extra uncertainty
3. **Degrees of Freedom**: df = n - 1 (one degree lost estimating the mean)

### 2.4 When to Use t vs. z

| Condition | Distribution | Use Case |
|-----------|--------------|----------|
| σ known, any n | z (Normal) | Rare in practice |
| σ unknown, n ≥ 30 | z or t | Both give similar results |
| σ unknown, n < 30 | t | Must use t-distribution |

---

## 3. Confidence Intervals for Proportions

### 3.1 Theory

For **proportions** (binary outcomes like defect/no defect, success/failure), we use the Normal approximation to the binomial distribution.

**Conditions for Normal Approximation**:
- $n\hat{p} \geq 10$
- $n(1-\hat{p}) \geq 10$

### 3.2 Mathematical Definition

**Large Sample CI for Proportion**:

$$\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

Where:
- $\hat{p}$ = sample proportion = x/n
- $x$ = number of successes
- $n$ = sample size

**Standard Error of Proportion**:

$$SE(\hat{p}) = \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$$

### 3.3 Supply Chain Application

**Retail Context**:
- **Quality Control**: CI for defect rate informs supplier quality assessments
- **Service Levels**: CI for fill rate and on-time delivery rate
- **Conversion Rates**: CI for website conversion or promotion response rates
- **SLA Compliance**: Intervals help determine if service level agreements are met

---

## 4. Sample Size Determination

### 4.1 For Means

To achieve a desired margin of error E:

$$n = \left(\frac{z_{\alpha/2} \cdot \sigma}{E}\right)^2$$

### 4.2 For Proportions

To achieve a desired margin of error E:

$$n = \left(\frac{z_{\alpha/2}}{E}\right)^2 \cdot \hat{p}(1-\hat{p})$$

**Conservative Approach**: Use $\hat{p} = 0.5$ when proportion is unknown (maximizes required sample size)

---

## 5. Factors Affecting Interval Width

| Factor | Effect on CI Width |
|--------|-------------------|
| Increase confidence level | Wider interval |
| Increase sample size n | Narrower interval |
| Higher variability (σ or s) | Wider interval |
| Decrease margin of error | Requires larger n |

---

## Summary

| Concept | Formula | Key Insight |
|---------|---------|-------------|
| CI for Mean (σ known) | $\bar{X} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$ | Uses z-distribution |
| CI for Mean (σ unknown) | $\bar{X} \pm t_{\alpha/2, n-1} \cdot \frac{s}{\sqrt{n}}$ | Uses t-distribution |
| CI for Proportion | $\hat{p} \pm z_{\alpha/2} \sqrt{\frac{\hat{p}(1-\hat{p})}{n}}$ | Normal approximation |
| t-Distribution | $T = \frac{\bar{X} - \mu}{S/\sqrt{n}}$ | Heavier tails than Normal |

## Key Takeaways
- Confidence intervals quantify estimation uncertainty with a range of plausible values
- Use t-distribution when population variance σ is unknown (most practical cases)
- Proportion CIs use Normal approximation to binomial distribution
- Larger sample sizes lead to narrower (more precise) confidence intervals

## Next Week Preview
Week 8 covers **Hypothesis Testing I** - testing claims about population parameters using statistical evidence.

---
*IIT Madras BS Degree in Data Science*
