# Week 05: Limit Theorems

**Course**: BSST1002 - Statistics II
**Level**: Foundation

---

## Learning Objectives
- Understand the Law of Large Numbers (LLN)
- Master the Central Limit Theorem (CLT)
- Apply these theorems to justify statistical methods

---

## 1. Law of Large Numbers (LLN)

### 1.1 Theory

The **Law of Large Numbers** states that the sample mean converges to the population mean as sample size increases. This justifies using historical averages as estimates.

### 1.2 Mathematical Definition

**Weak Law of Large Numbers**:

$$\bar{X}_n \xrightarrow{P} \mu \quad \text{as } n \to \infty$$

More precisely:

$$P(|\bar{X}_n - \mu| > \epsilon) \to 0 \quad \forall \epsilon > 0$$

Where:
- $\bar{X}_n = \frac{1}{n}\sum_{i=1}^n X_i$ is the sample mean
- $\mu = E[X]$ is the population mean
- $\xrightarrow{P}$ denotes convergence in probability

### 1.3 Requirements

- Random variables must be **independent**
- Random variables must be **identically distributed** (i.i.d.)
- Population mean $\mu$ must **exist** (finite)

### 1.4 Interpretation

| Sample Size | Sample Mean Behavior |
|-------------|---------------------|
| Small $n$ | High variability around $\mu$ |
| Large $n$ | Concentrates near $\mu$ |
| $n \to \infty$ | Converges to $\mu$ |

### 1.5 Rate of Convergence

The standard error of the mean:

$$SE(\bar{X}_n) = \frac{\sigma}{\sqrt{n}}$$

Decreases as $\sqrt{n}$, so:
- 4× data → 2× precision
- 100× data → 10× precision

### 1.6 Supply Chain Application

**Retail Context**:
- **Historical average demand** converges to true mean demand
- **More data → better estimates**
- Justifies **moving averages** for forecasting
- Supports using **sample statistics** for planning

---

## 2. Central Limit Theorem (CLT)

### 2.1 Theory

The **Central Limit Theorem** states that the sum (or average) of many independent random variables is approximately Normal, **regardless of the original distribution**.

### 2.2 Mathematical Definition

For i.i.d. random variables $X_1, X_2, \ldots, X_n$ with mean $\mu$ and variance $\sigma^2$:

$$\frac{\bar{X}_n - \mu}{\sigma/\sqrt{n}} \xrightarrow{d} N(0, 1) \quad \text{as } n \to \infty$$

Where $\xrightarrow{d}$ denotes convergence in distribution.

### 2.3 Practical Approximation

For large $n$:

$$\bar{X}_n \approx N\left(\mu, \frac{\sigma^2}{n}\right)$$

$$\sum_{i=1}^n X_i \approx N(n\mu, n\sigma^2)$$

### 2.4 Rule of Thumb for "Large n"

| Original Distribution | Required $n$ |
|----------------------|--------------|
| Symmetric, light tails | $n \geq 15$ |
| Moderately skewed | $n \geq 30$ |
| Highly skewed | $n \geq 50$ or more |

### 2.5 Why CLT is Remarkable

| Original Distribution | Sum/Average Distribution |
|----------------------|-------------------------|
| Uniform | → Normal |
| Exponential | → Normal |
| Poisson | → Normal |
| Bernoulli | → Normal |
| Any (with finite variance) | → Normal |

### 2.6 CLT for Sums vs. Means

| Quantity | Approximate Distribution |
|----------|-------------------------|
| **Sample Mean** $\bar{X}_n$ | $N(\mu, \sigma^2/n)$ |
| **Sum** $\sum X_i$ | $N(n\mu, n\sigma^2)$ |

### 2.7 Supply Chain Application

**Retail Context**:
- **Aggregate demand** over many periods is approximately Normal
- Even if **daily demand** is Poisson or skewed
- Justifies **Normal approximation** in inventory models
- Enables **z-score based safety stock** calculations

---

## 3. Comparing LLN and CLT

### 3.1 Key Differences

| Aspect | Law of Large Numbers | Central Limit Theorem |
|--------|---------------------|----------------------|
| **Focus** | Value of $\bar{X}_n$ | Distribution of $\bar{X}_n$ |
| **Statement** | $\bar{X}_n \to \mu$ | $\bar{X}_n \sim N(\mu, \sigma^2/n)$ |
| **Convergence** | In probability | In distribution |
| **Application** | Point estimation | Confidence intervals, hypothesis tests |

### 3.2 Complementary Roles

- **LLN** tells us **where** the sample mean goes (to $\mu$)
- **CLT** tells us **how** it gets there (via Normal distribution)

---

## 4. Applications in Statistics

### 4.1 Confidence Intervals

Using CLT, for large $n$:

$$\bar{X}_n \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

is an approximate $(1-\alpha)$ confidence interval for $\mu$.

### 4.2 Hypothesis Testing

Test statistic under $H_0: \mu = \mu_0$:

$$Z = \frac{\bar{X}_n - \mu_0}{\sigma/\sqrt{n}} \approx N(0,1)$$

### 4.3 Normal Approximation to Binomial

For $X \sim \text{Binomial}(n, p)$ with large $n$:

$$X \approx N(np, np(1-p))$$

Use when $np \geq 5$ and $n(1-p) \geq 5$.

### 4.4 Normal Approximation to Poisson

For $X \sim \text{Poisson}(\lambda)$ with large $\lambda$:

$$X \approx N(\lambda, \lambda)$$

Use when $\lambda \geq 10$.

---

## Summary Table

| Theorem | Statement | Implication |
|---------|-----------|-------------|
| **LLN** | $\bar{X}_n \to \mu$ | Sample means are consistent estimators |
| **CLT** | $\bar{X}_n \approx N(\mu, \sigma^2/n)$ | Can use Normal-based inference |
| **CLT (sum)** | $\sum X_i \approx N(n\mu, n\sigma^2)$ | Aggregate quantities are Normal |

---

## Key Takeaways

1. **LLN**: Sample mean → population mean as $n \to \infty$
2. **CLT**: Sums/averages are approximately Normal for large $n$
3. These theorems **justify** many statistical methods
4. Enable **Normal approximation** in inventory optimization and forecasting

---

## Next Week Preview

Week 6 covers **Point Estimation** - finding good estimators.

---

*IIT Madras BS Degree in Data Science*
