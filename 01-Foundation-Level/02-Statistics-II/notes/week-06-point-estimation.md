# Week 06: Point Estimation

**Course**: BSST1002 - Statistics II
**Level**: Foundation

---

## Learning Objectives
- Understand properties of good estimators
- Master Maximum Likelihood Estimation (MLE)
- Learn Method of Moments (MoM)

---

## 1. Estimator Properties

### 1.1 Theory

An **estimator** $\hat{\theta}$ is a function of sample data used to estimate an unknown parameter $\theta$. Good estimators have desirable properties.

### 1.2 Key Properties

| Property | Definition | Formula |
|----------|------------|---------|
| **Unbiased** | Expected value equals true parameter | $E[\hat{\theta}] = \theta$ |
| **Consistent** | Converges to true value as $n \to \infty$ | $\hat{\theta}_n \xrightarrow{P} \theta$ |
| **Efficient** | Minimum variance among unbiased estimators | Achieves Cramér-Rao lower bound |

### 1.3 Bias and Variance

| Concept | Formula |
|---------|---------|
| **Bias** | $\text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta$ |
| **Variance** | $\text{Var}(\hat{\theta})$ |
| **Mean Squared Error** | $\text{MSE} = \text{Bias}^2 + \text{Variance}$ |

### 1.4 Bias-Variance Trade-off

- **Unbiased** estimator: $\text{Bias} = 0$
- **Low variance** estimator: Tight around its mean
- Sometimes a **biased** estimator with lower variance has smaller MSE

### 1.5 Common Estimators

| Parameter | Estimator | Unbiased? |
|-----------|-----------|-----------|
| Population mean $\mu$ | $\bar{X} = \frac{1}{n}\sum X_i$ | ✓ Yes |
| Population variance $\sigma^2$ | $S^2 = \frac{1}{n-1}\sum(X_i - \bar{X})^2$ | ✓ Yes |
| Population variance | $\frac{1}{n}\sum(X_i - \bar{X})^2$ | ✗ No (biased low) |

### 1.6 Supply Chain Application

**Retail Context**:
- **Estimating average demand** for inventory planning
- **Estimating defect rate** for quality control
- **Estimating lead time parameters** for reorder point calculation

---

## 2. Maximum Likelihood Estimation (MLE)

### 2.1 Theory

**MLE** finds parameter values that maximize the probability (likelihood) of observing the data we actually observed.

**Intuition**: "What parameter value would have made our data most likely?"

### 2.2 Mathematical Definition

**Likelihood Function**:

$$L(\theta) = \prod_{i=1}^n f(x_i; \theta)$$

**Log-Likelihood** (easier to work with):

$$\ell(\theta) = \sum_{i=1}^n \log f(x_i; \theta)$$

**MLE**:

$$\hat{\theta}_{MLE} = \arg\max_\theta \ell(\theta)$$

### 2.3 Finding MLE

1. Write the likelihood $L(\theta)$
2. Take the log to get $\ell(\theta)$
3. Differentiate: $\frac{d\ell}{d\theta} = 0$
4. Solve for $\hat{\theta}$
5. Verify it's a maximum (second derivative < 0)

### 2.4 Common MLEs

| Distribution | Parameter | MLE |
|--------------|-----------|-----|
| **Normal** | $\mu$ | $\hat{\mu} = \bar{X}$ |
| **Normal** | $\sigma^2$ | $\hat{\sigma}^2 = \frac{1}{n}\sum(X_i - \bar{X})^2$ |
| **Poisson** | $\lambda$ | $\hat{\lambda} = \bar{X}$ |
| **Bernoulli** | $p$ | $\hat{p} = \bar{X}$ (sample proportion) |
| **Exponential** | $\lambda$ | $\hat{\lambda} = 1/\bar{X}$ |

### 2.5 Properties of MLE

| Property | Description |
|----------|-------------|
| **Consistent** | $\hat{\theta}_{MLE} \xrightarrow{P} \theta$ |
| **Asymptotically Normal** | $\hat{\theta}_{MLE} \approx N(\theta, I(\theta)^{-1})$ for large $n$ |
| **Asymptotically Efficient** | Achieves minimum variance for large $n$ |
| **Invariant** | MLE of $g(\theta)$ is $g(\hat{\theta}_{MLE})$ |

### 2.6 Supply Chain Application

**Retail Context**:
- **Fit demand distribution**: Estimate Poisson $\lambda$ or Normal $(\mu, \sigma^2)$
- **Lead time modeling**: Fit Exponential or Gamma parameters
- **Conversion rate estimation**: MLE for Bernoulli $p$

---

## 3. Method of Moments (MoM)

### 3.1 Theory

**Method of Moments** estimates parameters by matching sample moments to population moments.

### 3.2 Mathematical Definition

**Population moments**:
- 1st moment: $E[X] = \mu$
- 2nd moment: $E[X^2]$
- $k$th moment: $E[X^k]$

**Sample moments**:
- 1st: $\frac{1}{n}\sum X_i = \bar{X}$
- 2nd: $\frac{1}{n}\sum X_i^2$
- $k$th: $\frac{1}{n}\sum X_i^k$

**MoM Procedure**: Set sample moments equal to population moments and solve for parameters.

### 3.3 Example: Normal Distribution

For $X \sim N(\mu, \sigma^2)$:

| Moment Equation | Solution |
|-----------------|----------|
| $\bar{X} = \mu$ | $\hat{\mu} = \bar{X}$ |
| $\frac{1}{n}\sum X_i^2 = \mu^2 + \sigma^2$ | $\hat{\sigma}^2 = \frac{1}{n}\sum X_i^2 - \bar{X}^2$ |

### 3.4 MoM vs MLE Comparison

| Aspect | MLE | MoM |
|--------|-----|-----|
| **Principle** | Maximize likelihood | Match moments |
| **Computation** | Often harder | Usually simpler |
| **Efficiency** | Optimal (asymptotically) | May be less efficient |
| **Existence** | May not always exist | Usually exists |
| **Bias** | May be biased | May be biased |

### 3.5 When to Use Each

| Use MLE When | Use MoM When |
|--------------|--------------|
| Efficiency matters | Quick estimate needed |
| Distribution known | Closed-form moments available |
| Software available | MLE is intractable |

---

## 4. Standard Error

### 4.1 Definition

The **standard error** (SE) measures the variability of an estimator:

$$SE(\hat{\theta}) = \sqrt{\text{Var}(\hat{\theta})}$$

### 4.2 Common Standard Errors

| Estimator | Standard Error |
|-----------|----------------|
| Sample mean $\bar{X}$ | $SE = \sigma/\sqrt{n}$ |
| Sample proportion $\hat{p}$ | $SE = \sqrt{p(1-p)/n}$ |

### 4.3 Estimated Standard Error

When $\sigma$ is unknown, use sample standard deviation:

$$\widehat{SE}(\bar{X}) = S/\sqrt{n}$$

---

## Summary Table

| Concept | Definition | Key Formula |
|---------|------------|-------------|
| **Unbiased** | $E[\hat{\theta}] = \theta$ | No systematic error |
| **Consistent** | $\hat{\theta}_n \to \theta$ | Improves with data |
| **MLE** | Maximize likelihood | $\hat{\theta} = \arg\max \ell(\theta)$ |
| **MoM** | Match moments | Set $\bar{X}^k = E[X^k]$ |
| **Standard Error** | Estimator variability | $SE = \sigma/\sqrt{n}$ |

---

## Key Takeaways

1. **Good estimators**: Unbiased, consistent, efficient
2. **MLE**: Maximize probability of observed data — optimal asymptotically
3. **MoM**: Match sample and population moments — simpler but may be less efficient
4. **Standard error** quantifies estimator uncertainty

---

## Next Week Preview

Week 7 covers **Confidence Intervals** - quantifying estimation uncertainty.

---

*IIT Madras BS Degree in Data Science*
