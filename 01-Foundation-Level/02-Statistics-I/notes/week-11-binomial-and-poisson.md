# Week 11: Binomial and Poisson Distributions

**Course**: BSST1001 - Statistics I
**Level**: Foundation

---

## Learning Objectives
- Master the binomial distribution for fixed trials
- Understand the Poisson distribution for event counts
- Know when to use each distribution

---

## 1. Binomial Distribution

### 1.1 Theory

The **binomial distribution** models the number of successes in $n$ independent trials, where each trial has the same probability $p$ of success.

### 1.2 Conditions for Binomial

1. **Fixed number** of trials ($n$)
2. **Independent** trials
3. **Binary** outcomes (success/failure)
4. **Constant** probability $p$ for each trial

### 1.3 Mathematical Definition

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

Where:
- $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ = number of ways to choose $k$ successes
- $p^k$ = probability of $k$ successes
- $(1-p)^{n-k}$ = probability of $n-k$ failures

### 1.4 Parameters and Moments

| Property | Formula |
|----------|---------|
| **Parameters** | $n$ (trials), $p$ (success probability) |
| **Expected Value** | $E[X] = np$ |
| **Variance** | $\text{Var}(X) = np(1-p)$ |
| **Standard Deviation** | $\sigma = \sqrt{np(1-p)}$ |

### 1.5 Python Implementation

```python
from scipy import stats
# X ~ Binomial(n=20, p=0.3)
binom = stats.binom(n=20, p=0.3)
binom.pmf(k)     # P(X = k)
binom.cdf(k)     # P(X ≤ k)
binom.mean()     # E[X]
binom.var()      # Var(X)
```

### 1.6 Supply Chain Application

**Retail Context**:
- **Defective items** in a batch of $n$ products
- **On-time deliveries** out of $n$ shipments
- **Conversion rate** — customers who purchase out of visitors
- **Quality control** — pass/fail inspections

---

## 2. Poisson Distribution

### 2.1 Theory

The **Poisson distribution** models the number of events occurring in a fixed interval (time, space, etc.) when events happen independently at a constant average rate.

### 2.2 Conditions for Poisson

1. Events occur **independently**
2. Events occur at a **constant average rate** $\lambda$
3. Two events cannot occur at **exactly the same instant**
4. Probability is proportional to **interval length**

### 2.3 Mathematical Definition

$$P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}$$

Where:
- $\lambda$ = average number of events per interval
- $e \approx 2.718$ (Euler's number)
- $k!$ = factorial of $k$

### 2.4 Parameters and Moments

| Property | Formula |
|----------|---------|
| **Parameter** | $\lambda$ (rate) |
| **Expected Value** | $E[X] = \lambda$ |
| **Variance** | $\text{Var}(X) = \lambda$ |
| **Standard Deviation** | $\sigma = \sqrt{\lambda}$ |

**Key property**: Mean = Variance = $\lambda$

### 2.5 Python Implementation

```python
from scipy import stats
# X ~ Poisson(λ=5)
poisson = stats.poisson(mu=5)
poisson.pmf(k)     # P(X = k)
poisson.cdf(k)     # P(X ≤ k)
poisson.mean()     # E[X] = λ
poisson.var()      # Var(X) = λ
```

### 2.6 Supply Chain Application

**Retail Context**:
- **Customer arrivals** per hour
- **Orders per day** — especially for slow-moving items
- **Equipment failures** per month
- **Returns per week**
- **Demand for spare parts** — classic Poisson application

---

## 3. When to Use Each Distribution

### 3.1 Decision Framework

| Feature | Binomial | Poisson |
|---------|----------|---------|
| **Fixed trials?** | Yes ($n$ known) | No (continuous interval) |
| **Probability** | Known ($p$) | Rate known ($\lambda$) |
| **Outcome type** | Binary (success/fail) | Counts (0, 1, 2, ...) |
| **Mean-Variance** | $\text{Var} < \mu$ when $p < 0.5$ | $\text{Var} = \mu$ |

### 3.2 Quick Selection Guide

| Scenario | Distribution |
|----------|--------------|
| Defects in 100 items sampled | Binomial |
| Customer arrivals per hour | Poisson |
| Successful deliveries out of 50 | Binomial |
| Orders per day (low volume) | Poisson |
| Conversion from 1000 visitors | Binomial |
| Equipment breakdowns per month | Poisson |

### 3.3 Poisson Approximation to Binomial

When $n$ is large and $p$ is small, Binomial$(n, p) \approx$ Poisson$(\lambda = np)$.

**Rule of thumb**: Use approximation when $n \geq 20$ and $p \leq 0.05$

---

## 4. Comparing the Distributions

### 4.1 Shape Characteristics

| Distribution | Shape |
|--------------|-------|
| **Binomial** | Symmetric when $p = 0.5$; skewed otherwise |
| **Poisson** | Right-skewed; approaches symmetric as $\lambda$ increases |

### 4.2 Variance Comparison

| Distribution | Variance |
|--------------|----------|
| **Binomial** | $np(1-p) < np$ (always less than mean when $p < 1$) |
| **Poisson** | $\lambda$ (equal to mean) |

**Practical implication**: If observed variance > mean, consider negative binomial instead of Poisson.

---

## Summary Table

| Concept | Binomial | Poisson |
|---------|----------|---------|
| **PMF** | $\binom{n}{k}p^k(1-p)^{n-k}$ | $\frac{\lambda^k e^{-\lambda}}{k!}$ |
| **Parameters** | $n, p$ | $\lambda$ |
| **Mean** | $np$ | $\lambda$ |
| **Variance** | $np(1-p)$ | $\lambda$ |
| **Use Case** | Fixed trials, binary | Events in interval |
| **Supply Chain** | Quality control, conversion | Demand, arrivals |

---

## Key Takeaways

1. **Binomial**: Fixed $n$ trials, binary outcomes, known $p$
2. **Poisson**: Counts in interval, rare events, known rate $\lambda$
3. **Poisson approximates Binomial** when $n$ large, $p$ small ($\lambda = np$)
4. Both are **fundamental for supply chain analytics**

---

## Next Week Preview

Week 12 covers **Continuous Distributions** - Normal, Uniform, Exponential.

---

*IIT Madras BS Degree in Data Science*
