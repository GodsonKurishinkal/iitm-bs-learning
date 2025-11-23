# Week 12: Introduction to Continuous Random Variables

---
**Date**: 2025-11-22
**Course**: BSMA1002 - Statistics for Data Science I
**Level**: Foundation
**Week**: 12 of 12
**Source**: IIT Madras Statistics I Week 12
**Topic Area**: Continuous Probability Distributions
**Tags**: #BSMA1002 #ContinuousRV #PDF #Uniform #Exponential #Week12 #Foundation
---

## 📋 Bottom Line Up Front (BLUF)

**What**: Continuous random variables represent measurements that can take any value in an interval. Instead of PMF, we use probability density functions (PDF) where probabilities are areas under curves. Uniform and Exponential distributions are two fundamental continuous models.

**Why**: Continuous distributions model real-world measurements (time, distance, temperature) and form the foundation for advanced statistical inference, including the normal distribution central to hypothesis testing.

**Key Takeaway**: For continuous variables, $P(X = x) = 0$ for any specific $x$; probabilities are computed over intervals using integration. Uniform models equal likelihood, Exponential models waiting times.

---

## 🎯 Learning Objectives

By the end of this week, you should be able to:

1. ✅ Understand continuous random variables and their properties
2. ✅ Work with probability density functions (PDF) and interpret them
3. ✅ Calculate probabilities as areas under PDF curves
4. ✅ Apply Uniform distribution for equal-probability scenarios
5. ✅ Use Exponential distribution for modeling waiting times
6. ✅ Compute expectation and variance for continuous distributions
7. ✅ Implement continuous distributions in Python using scipy.stats

---

## 📚 Table of Contents

1. [From Discrete to Continuous](#from-discrete-to-continuous)
2. [Probability Density Function (PDF)](#probability-density-function-pdf)
3. [Area Under the Curve](#area-under-the-curve)
4. [Properties of PDF](#properties-of-pdf)
5. [Uniform Distribution](#uniform-distribution)
6. [Exponential Distribution](#exponential-distribution)
7. [Data Science Applications](#data-science-applications)
8. [Common Pitfalls](#common-pitfalls)
9. [Python Implementation](#python-implementation)
10. [Course Summary and Next Steps](#course-summary-and-next-steps)

---

## 🔄 From Discrete to Continuous

### The Fundamental Shift

**Discrete Random Variables:**
- Countable values: $\{0, 1, 2, 3, ...\}$ or $\{1, 2, 3, 4, 5, 6\}$
- PMF: $p_X(x) = P(X = x)$ (probability at each point)
- Sum probabilities: $\sum_{\text{all } x} p_X(x) = 1$

**Continuous Random Variables:**
- Uncountable values: any value in interval $[a, b]$ or $(-\infty, \infty)$
- PDF: $f_X(x)$ (density function, NOT a probability)
- Integrate density: $\int_{-\infty}^{\infty} f_X(x) dx = 1$

### The Key Difference

**Critical insight:** For continuous random variables:
$$P(X = x) = 0 \quad \text{for any specific value } x$$

**Why?** Infinitely many possible values → each individual point has zero probability.

**Instead, we ask:** What's the probability $X$ falls in an interval?
$$P(a \leq X \leq b) = \int_a^b f_X(x) dx$$

### Example 1: Bus Waiting Time

**Discrete thinking (wrong):** "What's probability of waiting exactly 7.3245... minutes?"
- Answer: 0 (infinitely precise value has zero probability)

**Continuous thinking (correct):** "What's probability of waiting between 7 and 8 minutes?"
- Answer: $P(7 \leq X \leq 8) = \int_7^8 f_X(x) dx$ (non-zero)

### Measurement Precision Analogy

**Discrete:** Counting people - exactly 5 people (precise count)

**Continuous:** Measuring height - "6 feet" really means "between 5.95 and 6.05 feet" (interval)

When you increase measurement precision, you're narrowing the interval, not finding an exact value.

---

## 📊 Probability Density Function (PDF)

### Definition

**Definition:** For a continuous random variable $X$, the **probability density function (PDF)** $f_X(x)$ satisfies:

$$P(a \leq X \leq b) = \int_a^b f_X(x) dx$$

**Purpose:** The PDF describes the relative likelihood of different values - higher $f_X(x)$ means $x$ is more likely (but $f_X(x)$ itself is NOT a probability!).

### Key Properties

**1. Non-negativity:**
$$f_X(x) \geq 0 \quad \text{for all } x$$

**2. Normalization (total area = 1):**
$$\int_{-\infty}^{\infty} f_X(x) dx = 1$$

**3. Probability as area:**
$$P(a \leq X \leq b) = \int_a^b f_X(x) dx = \text{Area under } f_X \text{ from } a \text{ to } b$$

### Important Clarifications

**$f_X(x)$ can be greater than 1!**

Example: If $X$ uniform on $[0, 0.5]$, then $f_X(x) = 2$ on that interval.
- This is OK because $f_X(x)$ is a **density**, not a probability
- Total area still equals 1: $\int_0^{0.5} 2 dx = 2 \times 0.5 = 1$ ✓

**Endpoints don't matter:**
$$P(a \leq X \leq b) = P(a < X < b) = P(a \leq X < b) = P(a < X \leq b)$$

All equal because $P(X = a) = P(X = b) = 0$.

### Example 2: Simple PDF

**PDF:**
$$f_X(x) = \begin{cases}
2x & \text{if } 0 \leq x \leq 1 \\
0 & \text{otherwise}
\end{cases}$$

**Verify it's a valid PDF:**

**Check 1: Non-negativity:** $2x \geq 0$ for $x \in [0, 1]$ ✓

**Check 2: Normalization:**
$$\int_{-\infty}^{\infty} f_X(x) dx = \int_0^1 2x dx = \left[ x^2 \right]_0^1 = 1 - 0 = 1 \checkmark$$

**Calculate probability:** $P(0.5 \leq X \leq 1)$

$$P(0.5 \leq X \leq 1) = \int_{0.5}^{1} 2x dx = \left[ x^2 \right]_{0.5}^{1} = 1^2 - (0.5)^2 = 1 - 0.25 = 0.75$$

**Interpretation:** 75% chance $X$ is between 0.5 and 1.

**Why higher density at $x=1$?** $f_X(1) = 2 > f_X(0) = 0$, so values near 1 are more likely.

---

## 📐 Area Under the Curve

### Geometric Interpretation

**Probability = Area under PDF curve**

For continuous $X$:
$$P(a \leq X \leq b) = \int_a^b f_X(x) dx = \text{Area under curve from } a \text{ to } b$$

### Visual Understanding

```
  f_X(x)
    │    ╱─────╲
    │   ╱       ╲
    │  ╱         ╲
    │ ╱   AREA   ╲
    │╱             ╲
    └─────┴─────────┴──→ x
         a         b

Area = P(a ≤ X ≤ b)
```

**Total area under entire curve = 1** (certainty that $X$ takes some value)

### Example 3: Triangular Distribution

**PDF:**
$$f_X(x) = \begin{cases}
x & \text{if } 0 \leq x \leq 1 \\
2 - x & \text{if } 1 < x \leq 2 \\
0 & \text{otherwise}
\end{cases}$$

**Shape:** Triangle with peak at $x = 1$ where $f_X(1) = 1$

**Verification:**
$$\int_0^2 f_X(x) dx = \int_0^1 x dx + \int_1^2 (2-x) dx$$

$$= \left[\frac{x^2}{2}\right]_0^1 + \left[2x - \frac{x^2}{2}\right]_1^2$$

$$= \frac{1}{2} + \left[(4 - 2) - (2 - 0.5)\right] = 0.5 + (2 - 1.5) = 0.5 + 0.5 = 1 \checkmark$$

**Calculate:** $P(0.5 \leq X \leq 1.5)$

$$= \int_{0.5}^{1} x dx + \int_1^{1.5} (2-x) dx$$

$$= \left[\frac{x^2}{2}\right]_{0.5}^{1} + \left[2x - \frac{x^2}{2}\right]_1^{1.5}$$

$$= \left(\frac{1}{2} - \frac{0.25}{2}\right) + \left[(3 - 1.125) - (2 - 0.5)\right]$$

$$= (0.5 - 0.125) + (1.875 - 1.5) = 0.375 + 0.375 = 0.75$$

**Interpretation:** 75% chance $X \in [0.5, 1.5]$.

### Connection to CDF

**Cumulative Distribution Function (CDF):** Same definition as discrete!

$$F_X(x) = P(X \leq x) = \int_{-\infty}^{x} f_X(t) dt$$

**Relationship:** PDF is the derivative of CDF:
$$f_X(x) = \frac{d}{dx} F_X(x)$$

**Probability of interval using CDF:**
$$P(a \leq X \leq b) = F_X(b) - F_X(a)$$

---

## ✅ Properties of PDF

### Summary of Requirements

For $f_X(x)$ to be a valid PDF:

1. **Non-negativity:** $f_X(x) \geq 0$ for all $x$
2. **Normalization:** $\int_{-\infty}^{\infty} f_X(x) dx = 1$
3. **Probabilities are areas:** $P(a \leq X \leq b) = \int_a^b f_X(x) dx$

### Common PDF Shapes

**Uniform:** Constant (flat) over interval
```
f_X(x)
  │  ┌────────┐
  │  │        │
  │  │        │
  └──┴────────┴──→ x
    a        b
```

**Exponential:** Decreasing from left
```
f_X(x)
  │╲
  │ ╲
  │  ╲
  │   ╲___
  └────────→ x
  0
```

**Normal (bell curve):** Symmetric bell shape (coming next course!)
```
f_X(x)
  │    ╱─╲
  │   ╱   ╲
  │  ╱     ╲
  │ ╱       ╲
  └─────────────→ x
```

### Example 4: Checking Validity

**Is this a valid PDF?**

$$f_X(x) = \begin{cases}
cx^2 & \text{if } 0 \leq x \leq 2 \\
0 & \text{otherwise}
\end{cases}$$

**Solution:**

**Step 1:** Find $c$ using normalization.

$$\int_0^2 cx^2 dx = 1$$

$$c \left[\frac{x^3}{3}\right]_0^2 = 1$$

$$c \cdot \frac{8}{3} = 1$$

$$c = \frac{3}{8}$$

**Step 2:** Check non-negativity.

$$f_X(x) = \frac{3}{8}x^2 \geq 0 \text{ for } x \in [0, 2] \checkmark$$

**Valid PDF:**
$$f_X(x) = \begin{cases}
\frac{3}{8}x^2 & \text{if } 0 \leq x \leq 2 \\
0 & \text{otherwise}
\end{cases}$$

---

## 📏 Uniform Distribution

### Definition

**Definition:** A random variable $X$ has a **uniform distribution** on $[a, b]$ if all values in the interval are equally likely.

**Notation:** $X \sim \text{Uniform}(a, b)$ or $X \sim U(a, b)$

**PDF:**
$$f_X(x) = \begin{cases}
\frac{1}{b-a} & \text{if } a \leq x \leq b \\
0 & \text{otherwise}
\end{cases}$$

**Purpose:** Models situations where all outcomes in a range are equally probable.

### Verification

**Normalization:**
$$\int_a^b \frac{1}{b-a} dx = \frac{1}{b-a} \cdot (b - a) = 1 \checkmark$$

**Non-negativity:** $\frac{1}{b-a} > 0$ if $b > a$ ✓

### Probabilities in Uniform

**Key insight:** For Uniform, probability is proportional to length of interval.

$$P(c \leq X \leq d) = \frac{d - c}{b - a}$$

**Interpretation:** Probability = (length of target interval) / (length of total interval)

### Example 5: Bus Arrival

**Scenario:** Bus arrives uniformly between 0 and 15 minutes.

**Distribution:** $X \sim \text{Uniform}(0, 15)$ where $X$ = waiting time

**PDF:**
$$f_X(x) = \begin{cases}
\frac{1}{15} & \text{if } 0 \leq x \leq 15 \\
0 & \text{otherwise}
\end{cases}$$

**Calculate probabilities:**

**(a) $P(5 \leq X \leq 10)$ (wait between 5-10 minutes)**

$$P(5 \leq X \leq 10) = \int_5^{10} \frac{1}{15} dx = \frac{1}{15} \cdot 5 = \frac{5}{15} = \frac{1}{3} \approx 0.333$$

33.3% chance.

**(b) $P(X \leq 3)$ (wait at most 3 minutes)**

$$P(X \leq 3) = \frac{3 - 0}{15 - 0} = \frac{3}{15} = 0.2$$

20% chance.

**(c) $P(X > 12)$ (wait more than 12 minutes)**

$$P(X > 12) = \frac{15 - 12}{15} = \frac{3}{15} = 0.2$$

20% chance.

### Uniform Expectation and Variance

**Expectation (midpoint):**
$$E[X] = \frac{a + b}{2}$$

**Variance:**
$$\text{Var}(X) = \frac{(b-a)^2}{12}$$

**Standard deviation:**
$$\sigma_X = \frac{b-a}{\sqrt{12}} = \frac{b-a}{2\sqrt{3}}$$

### Example 6: Uniform Calculations

**Distribution:** $X \sim \text{Uniform}(2, 8)$

**Expectation:**
$$E[X] = \frac{2 + 8}{2} = 5$$

**Variance:**
$$\text{Var}(X) = \frac{(8-2)^2}{12} = \frac{36}{12} = 3$$

**Standard deviation:**
$$\sigma_X = \sqrt{3} \approx 1.73$$

**Interpretation:** Average value is 5, typically varies by ±1.73.

---

## ⏱️ Exponential Distribution

### Definition

**Definition:** The **exponential distribution** models the time until an event occurs, when events happen continuously and independently at a constant average rate.

**Notation:** $X \sim \text{Exponential}(\lambda)$ or $X \sim \text{Exp}(\lambda)$

**Parameter:** $\lambda$ (lambda) = rate parameter ($\lambda > 0$)

**PDF:**
$$f_X(x) = \begin{cases}
\lambda e^{-\lambda x} & \text{if } x \geq 0 \\
0 & \text{if } x < 0
\end{cases}$$

**Purpose:** Models waiting times, lifetimes, time between events.

### Verification

**Normalization:**
$$\int_0^{\infty} \lambda e^{-\lambda x} dx = \lambda \left[-\frac{1}{\lambda} e^{-\lambda x}\right]_0^{\infty} = \lambda \cdot \frac{1}{\lambda} = 1 \checkmark$$

**Non-negativity:** $\lambda e^{-\lambda x} > 0$ for $x \geq 0$ ✓

### CDF of Exponential

**Cumulative Distribution Function:**
$$F_X(x) = P(X \leq x) = 1 - e^{-\lambda x} \quad \text{for } x \geq 0$$

**Derivation:**
$$F_X(x) = \int_0^x \lambda e^{-\lambda t} dt = \left[-e^{-\lambda t}\right]_0^x = -e^{-\lambda x} - (-1) = 1 - e^{-\lambda x}$$

**Probability calculations:**
$$P(X > x) = 1 - F_X(x) = e^{-\lambda x}$$

**Interval probability:**
$$P(a \leq X \leq b) = F_X(b) - F_X(a) = e^{-\lambda a} - e^{-\lambda b}$$

### Example 7: Server Uptime

**Scenario:** Time until server failure follows $\text{Exponential}(\lambda = 0.1)$ per hour.

**PDF:**
$$f_X(x) = 0.1 e^{-0.1x} \text{ for } x \geq 0$$

**Calculate probabilities:**

**(a) $P(X > 5)$ (survives more than 5 hours)**

$$P(X > 5) = e^{-0.1 \times 5} = e^{-0.5} \approx 0.6065$$

60.65% chance.

**(b) $P(X \leq 10)$ (fails within 10 hours)**

$$P(X \leq 10) = 1 - e^{-0.1 \times 10} = 1 - e^{-1} = 1 - 0.3679 = 0.6321$$

63.21% chance.

**(c) $P(5 \leq X \leq 10)$ (fails between 5-10 hours)**

$$P(5 \leq X \leq 10) = e^{-0.5} - e^{-1} = 0.6065 - 0.3679 = 0.2386$$

23.86% chance.

### Exponential Expectation and Variance

**Expectation (mean waiting time):**
$$E[X] = \frac{1}{\lambda}$$

**Variance:**
$$\text{Var}(X) = \frac{1}{\lambda^2}$$

**Standard deviation:**
$$\sigma_X = \frac{1}{\lambda}$$

**Note:** For Exponential, $E[X] = \sigma_X = \frac{1}{\lambda}$

### Example 8: Customer Service Time

**Scenario:** Service time per customer follows $\text{Exponential}(\lambda = 0.2)$ per minute.

**Mean service time:**
$$E[X] = \frac{1}{0.2} = 5 \text{ minutes}$$

**Standard deviation:**
$$\sigma_X = \frac{1}{0.2} = 5 \text{ minutes}$$

**Probability of serving customer in less than 3 minutes:**
$$P(X \leq 3) = 1 - e^{-0.2 \times 3} = 1 - e^{-0.6} = 1 - 0.5488 = 0.4512$$

45.12% chance.

### Memoryless Property

**Key property of Exponential:** Memorylessness

$$P(X > s + t | X > s) = P(X > t)$$

**Interpretation:** "The future doesn't depend on the past."

If the server has been running for 5 hours, the probability it runs another 3 hours is the same as if it just started.

**Example:** $\lambda = 0.1$, $s = 5$, $t = 3$

$$P(X > 8 | X > 5) = \frac{P(X > 8)}{P(X > 5)} = \frac{e^{-0.8}}{e^{-0.5}} = e^{-0.3} = P(X > 3)$$

Verified! ✓

---

## 💼 Data Science Applications

### Application 1: A/B Test Duration with Exponential

**Problem:** Users spend time on website. Model session duration with $\text{Exponential}(\lambda = 0.1)$ per minute.

**Mean session time:** $E[X] = \frac{1}{0.1} = 10$ minutes

**Business question:** What fraction of users leave within 5 minutes?

$$P(X \leq 5) = 1 - e^{-0.1 \times 5} = 1 - e^{-0.5} = 0.3935$$

**Interpretation:** 39.35% leave within 5 minutes (high bounce rate).

**Optimization target:** Reduce $\lambda$ (increase mean session time).

### Application 2: Queueing Theory with Uniform

**Problem:** Customer arrivals uniformly distributed over hour (0-60 minutes).

**Model:** $X \sim \text{Uniform}(0, 60)$ where $X$ = arrival time

**Question:** Probability a customer arrives in first 15 minutes?

$$P(X \leq 15) = \frac{15}{60} = 0.25$$

25% arrive in first quarter-hour.

**Resource planning:** Staff accordingly to handle 25% of daily customers in first 15 min.

### Application 3: Response Time SLA

**Problem:** API response times follow $\text{Exponential}(\lambda = 2)$ per second.

**Mean response time:** $E[X] = \frac{1}{2} = 0.5$ seconds

**SLA:** 95% of requests must complete within 2 seconds.

**Check compliance:**
$$P(X \leq 2) = 1 - e^{-2 \times 2} = 1 - e^{-4} = 1 - 0.0183 = 0.9817$$

**Result:** 98.17% complete within 2 sec → SLA met! ✓

### Application 4: Simulating Random Data

**Problem:** Generate synthetic data for testing ML pipeline.

**Scenario 1: Feature values uniformly distributed**
- $X \sim \text{Uniform}(0, 1)$
- Use for random feature initialization

**Scenario 2: Time-to-event uniformly distributed**
- $T \sim \text{Uniform}(0, 100)$ days
- Simulate churn times

**Scenario 3: Inter-arrival times exponentially distributed**
- $\Delta t \sim \text{Exponential}(\lambda = 5)$ per hour
- Simulate streaming data arrivals

---

## ⚠️ Common Pitfalls

### ❌ Pitfall 1: PDF is Not Probability

**Wrong:** "$f_X(0.5) = 2$, so $P(X = 0.5) = 2$" (treating PDF as probability)

**Right:** $P(X = 0.5) = 0$ (point probability is zero for continuous)

**Remember:** $f_X(x)$ is a **density**, not a probability. It can be > 1.

### ❌ Pitfall 2: Forgetting Integration

**Wrong:** "$P(1 \leq X \leq 3)$ for Uniform$(0, 10)$ is $\frac{1}{10}$"

**Right:**
$$P(1 \leq X \leq 3) = \int_1^3 \frac{1}{10} dx = \frac{1}{10} \times 2 = \frac{2}{10} = 0.2$$

Must integrate over the interval, not evaluate PDF at a point.

### ❌ Pitfall 3: Exponential Parameter Confusion

**Wrong:** "Mean waiting time is 5 minutes → $X \sim \text{Exp}(5)$"

**Right:** Mean = $\frac{1}{\lambda}$, so if mean = 5, then $\lambda = \frac{1}{5} = 0.2$

$$X \sim \text{Exp}(0.2)$$

**Remember:** $\lambda$ is the **rate**, not the mean.

### ❌ Pitfall 4: Interval Endpoints

**Wrong:** "Need to carefully distinguish $P(a \leq X \leq b)$ from $P(a < X < b)$"

**Right:** For continuous variables, these are **equal**:
$$P(a \leq X \leq b) = P(a < X < b)$$

Because $P(X = a) = P(X = b) = 0$.

### ❌ Pitfall 5: Uniform Variance Formula

**Wrong:** "$\text{Var}(X)$ for Uniform$(a,b)$ is $\frac{b-a}{12}$"

**Right:**
$$\text{Var}(X) = \frac{(b-a)^2}{12}$$

Don't forget to square the range!

---

## 💻 Python Implementation

### Implementation 1: Uniform Distribution

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Define Uniform distribution on [2, 8]
a, b = 2, 8
uniform_rv = stats.uniform(loc=a, scale=b-a)

# PDF
x = 5.5
pdf_value = uniform_rv.pdf(x)
print(f"f_X({x}) = {pdf_value:.4f}")  # 0.1667 (= 1/6)

# CDF
cdf_value = uniform_rv.cdf(x)
print(f"F_X({x}) = P(X ≤ {x}) = {cdf_value:.4f}")  # 0.5833

# Probability of interval
prob = uniform_rv.cdf(7) - uniform_rv.cdf(4)
print(f"P(4 ≤ X ≤ 7) = {prob:.4f}")  # 0.5000

# Mean and variance
print(f"E[X] = {uniform_rv.mean():.2f}")  # 5.00
print(f"Var(X) = {uniform_rv.var():.2f}")  # 3.00
print(f"σ_X = {uniform_rv.std():.2f}")  # 1.73

# Generate random samples
samples = uniform_rv.rvs(size=1000)
print(f"Sample mean: {np.mean(samples):.2f}")  # ≈ 5.00
print(f"Sample std: {np.std(samples, ddof=1):.2f}")  # ≈ 1.73
```

### Implementation 2: Visualizing Uniform PDF and CDF

```python
# Plot PDF and CDF
x_values = np.linspace(0, 10, 1000)
pdf_values = uniform_rv.pdf(x_values)
cdf_values = uniform_rv.cdf(x_values)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# PDF
ax1.plot(x_values, pdf_values, 'b-', linewidth=2, label='PDF')
ax1.fill_between(x_values, pdf_values, where=(x_values >= 4) & (x_values <= 7),
                 alpha=0.3, color='red', label='P(4 ≤ X ≤ 7)')
ax1.axhline(1/6, color='gray', linestyle='--', alpha=0.5)
ax1.set_xlabel('x')
ax1.set_ylabel('f_X(x)')
ax1.set_title('Uniform(2, 8) PDF')
ax1.legend()
ax1.grid(True, alpha=0.3)

# CDF
ax2.plot(x_values, cdf_values, 'b-', linewidth=2, label='CDF')
ax2.axhline(0, color='gray', linestyle='--', alpha=0.3)
ax2.axhline(1, color='gray', linestyle='--', alpha=0.3)
ax2.set_xlabel('x')
ax2.set_ylabel('F_X(x) = P(X ≤ x)')
ax2.set_title('Uniform(2, 8) CDF')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### Implementation 3: Exponential Distribution

```python
# Define Exponential distribution with λ=0.5
lambda_param = 0.5
exp_rv = stats.expon(scale=1/lambda_param)  # Note: scale = 1/λ

# PDF
x = 2
pdf_value = exp_rv.pdf(x)
print(f"f_X({x}) = {pdf_value:.4f}")  # λe^(-λx)

# CDF
cdf_value = exp_rv.cdf(x)
print(f"F_X({x}) = P(X ≤ {x}) = {cdf_value:.4f}")  # 1 - e^(-λx)

# Survival function: P(X > x)
survival = 1 - exp_rv.cdf(x)
print(f"P(X > {x}) = {survival:.4f}")  # e^(-λx)

# Or use built-in survival function
survival_builtin = exp_rv.sf(x)
print(f"P(X > {x}) [sf] = {survival_builtin:.4f}")  # Same

# Mean and variance
print(f"E[X] = {exp_rv.mean():.2f}")  # 2.00 (= 1/λ)
print(f"Var(X) = {exp_rv.var():.2f}")  # 4.00 (= 1/λ²)
print(f"σ_X = {exp_rv.std():.2f}")  # 2.00

# Probability of interval
prob_interval = exp_rv.cdf(5) - exp_rv.cdf(2)
print(f"P(2 ≤ X ≤ 5) = {prob_interval:.4f}")  # 0.2865
```

### Implementation 4: Comparing Exponential for Different λ

```python
# Compare Exponential distributions with different rates
lambdas = [0.5, 1.0, 2.0]
x_vals = np.linspace(0, 8, 500)

plt.figure(figsize=(12, 6))

for lam in lambdas:
    exp_rv_temp = stats.expon(scale=1/lam)
    pdf_vals = exp_rv_temp.pdf(x_vals)
    plt.plot(x_vals, pdf_vals, linewidth=2,
             label=f'λ={lam} (mean={1/lam:.1f})')

plt.xlabel('x')
plt.ylabel('f_X(x)')
plt.title('Exponential PDF for Different Rate Parameters')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(0, 2.5)
plt.show()

# Observation: Higher λ → faster decay → shorter waiting times
```

### Implementation 5: Memoryless Property Verification

```python
# Verify memoryless property of Exponential
lambda_param = 0.1
exp_rv = stats.expon(scale=1/lambda_param)

s, t = 5, 3

# P(X > s+t | X > s) should equal P(X > t)
prob_conditional = (1 - exp_rv.cdf(s + t)) / (1 - exp_rv.cdf(s))
prob_direct = 1 - exp_rv.cdf(t)

print(f"P(X > {s+t} | X > {s}) = {prob_conditional:.6f}")
print(f"P(X > {t}) = {prob_direct:.6f}")
print(f"Difference: {abs(prob_conditional - prob_direct):.10f}")

# Should be essentially zero (within numerical precision)
if abs(prob_conditional - prob_direct) < 1e-10:
    print("Memoryless property verified! ✓")
```

---

## 📝 Practice Problems

### Basic Level

**Problem 1:** $X \sim \text{Uniform}(0, 10)$

(a) Write the PDF.
(b) Calculate $P(3 \leq X \leq 7)$.
(c) Find $E[X]$ and $\text{Var}(X)$.

<details>
<summary>Solution</summary>

(a) $$f_X(x) = \begin{cases} \frac{1}{10} & \text{if } 0 \leq x \leq 10 \\ 0 & \text{otherwise} \end{cases}$$

(b) $$P(3 \leq X \leq 7) = \frac{7-3}{10-0} = \frac{4}{10} = 0.4$$

(c) $$E[X] = \frac{0+10}{2} = 5$$
$$\text{Var}(X) = \frac{(10-0)^2}{12} = \frac{100}{12} = 8.33$$
</details>

**Problem 2:** $Y \sim \text{Exponential}(0.5)$

(a) Calculate $P(Y > 2)$.
(b) Find $E[Y]$ and $\text{Var}(Y)$.

<details>
<summary>Solution</summary>

(a) $$P(Y > 2) = e^{-0.5 \times 2} = e^{-1} = 0.3679$$

(b) $$E[Y] = \frac{1}{0.5} = 2$$
$$\text{Var}(Y) = \frac{1}{0.5^2} = 4$$
</details>

**Problem 3:** For continuous $X$, is $P(X = 5)$ equal to 0? Why?

<details>
<summary>Solution</summary>

Yes, $P(X = 5) = 0$ for any continuous random variable.

**Reason:** Probability of a single point in a continuous distribution is zero because there are uncountably many possible values. Probability is spread over intervals, not concentrated at points.
</details>

### Intermediate Level

**Problem 4:** Random variable has PDF:

$$f_X(x) = \begin{cases}
cx & \text{if } 0 \leq x \leq 3 \\
0 & \text{otherwise}
\end{cases}$$

(a) Find $c$.
(b) Calculate $P(1 \leq X \leq 2)$.
(c) Find the CDF $F_X(x)$ for $0 \leq x \leq 3$.

<details>
<summary>Solution</summary>

(a) Normalization:
$$\int_0^3 cx dx = 1$$
$$c \left[\frac{x^2}{2}\right]_0^3 = c \cdot \frac{9}{2} = 1$$
$$c = \frac{2}{9}$$

(b) $$P(1 \leq X \leq 2) = \int_1^2 \frac{2}{9}x dx = \frac{2}{9} \left[\frac{x^2}{2}\right]_1^2 = \frac{2}{9} \cdot \frac{4-1}{2} = \frac{2}{9} \cdot \frac{3}{2} = \frac{1}{3}$$

(c) $$F_X(x) = \int_0^x \frac{2}{9}t dt = \frac{2}{9} \cdot \frac{x^2}{2} = \frac{x^2}{9} \quad \text{for } 0 \leq x \leq 3$$
</details>

**Problem 5:** Bus arrives uniformly between 8:00 and 8:20 AM. You arrive at 8:07.

(a) Model waiting time as continuous uniform. Parameters?
(b) Probability you wait more than 5 minutes?

<details>
<summary>Solution</summary>

(a) Arrival time $T \sim \text{Uniform}(0, 20)$ (minutes after 8:00)

Given you arrive at 7 minutes, bus arrives uniformly on $[7, 20]$

Waiting time $W = T - 7 \sim \text{Uniform}(0, 13)$

(b) $$P(W > 5) = \frac{13-5}{13-0} = \frac{8}{13} \approx 0.615$$

61.5% chance of waiting more than 5 minutes.
</details>

### Advanced Level

**Problem 6:** Server uptime follows $\text{Exponential}(\lambda = 0.05)$ per hour.

(a) What's the mean time until failure?
(b) Probability server lasts at least 20 hours?
(c) Given server has lasted 20 hours, what's probability it lasts another 10 hours?

<details>
<summary>Solution</summary>

(a) $$E[X] = \frac{1}{\lambda} = \frac{1}{0.05} = 20 \text{ hours}$$

(b) $$P(X \geq 20) = e^{-0.05 \times 20} = e^{-1} = 0.3679$$

36.79% chance.

(c) Using memoryless property:
$$P(X \geq 30 | X \geq 20) = P(X \geq 10) = e^{-0.05 \times 10} = e^{-0.5} = 0.6065$$

60.65% chance (same as if it just started!).
</details>

**Problem 7:** API response times: 80% follow $\text{Exponential}(2)$, 20% follow $\text{Exponential}(0.5)$.

(a) What's overall expected response time?
(b) If a request takes more than 2 seconds, what's the probability it's the slow type?

<details>
<summary>Solution</summary>

(a) Let $F$ = fast, $S$ = slow

$$E[X] = P(F) E[X|F] + P(S) E[X|S]$$
$$= 0.8 \times \frac{1}{2} + 0.2 \times \frac{1}{0.5}$$
$$= 0.8 \times 0.5 + 0.2 \times 2$$
$$= 0.4 + 0.4 = 0.8 \text{ seconds}$$

(b) Use Bayes' theorem:

$$P(S | X > 2) = \frac{P(X > 2 | S) P(S)}{P(X > 2)}$$

$$P(X > 2 | F) = e^{-2 \times 2} = e^{-4} \approx 0.0183$$
$$P(X > 2 | S) = e^{-0.5 \times 2} = e^{-1} \approx 0.3679$$

$$P(X > 2) = 0.8 \times 0.0183 + 0.2 \times 0.3679 = 0.0146 + 0.0736 = 0.0882$$

$$P(S | X > 2) = \frac{0.3679 \times 0.2}{0.0882} = \frac{0.0736}{0.0882} \approx 0.834$$

83.4% probability it's the slow type.
</details>

---

## 🎓 Course Summary and Next Steps

### What We've Learned (12 Weeks)

**Weeks 1-3: Data Foundations**
- Data types (categorical, numerical)
- Summary statistics (mean, median, mode, variance)
- Visualization (bar charts, histograms, box plots)

**Weeks 4-6: Probability Basics**
- Counting principles (permutations, combinations)
- Probability axioms and rules
- Conditional probability and Bayes' theorem

**Weeks 7-8: Discrete Probability**
- Random variables and PMF
- Binomial and Poisson distributions

**Weeks 9-11: Expected Values**
- Expectation and variance
- Properties of distributions
- Applications to decision-making

**Week 12: Continuous Probability**
- PDF and continuous distributions
- Uniform and Exponential models
- Foundation for normal distribution

### Connection to Statistics II

**Building on this course:**

**Statistics II will cover:**
- **Normal distribution** (the famous bell curve)
- **Central Limit Theorem** (why normal is everywhere)
- **Sampling distributions** (distribution of sample statistics)
- **Hypothesis testing** (making decisions from data)
- **Confidence intervals** (quantifying uncertainty)
- **Regression and correlation** (modeling relationships)

**The foundation you've built:**
- Understanding of probability enables hypothesis testing
- PMF/PDF concepts extend to sampling distributions
- Expectation/variance underpin all statistical inference
- Binomial/Poisson are workhorses of real-world modeling

### Key Takeaways

1. **Probability is the language of uncertainty** - essential for data science
2. **Distributions model reality** - choose the right one for your problem
3. **Expectation and variance summarize distributions** - mean and spread
4. **Discrete vs Continuous** - counting vs measuring
5. **Python enables computation** - scipy.stats is your friend

---

## 📚 Quick Reference Summary

### Distribution Comparison

| Distribution | Type | Parameters | Support | Mean | Variance | Use Case |
|--------------|------|------------|---------|------|----------|----------|
| Bernoulli | Discrete | $p$ | {0,1} | $p$ | $p(1-p)$ | Single trial |
| Binomial | Discrete | $n, p$ | {0,...,n} | $np$ | $np(1-p)$ | $n$ trials |
| Poisson | Discrete | $\lambda$ | {0,1,2,...} | $\lambda$ | $\lambda$ | Rare events |
| Uniform | Continuous | $a, b$ | $[a,b]$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | Equal likelihood |
| Exponential | Continuous | $\lambda$ | $[0,\infty)$ | $\frac{1}{\lambda}$ | $\frac{1}{\lambda^2}$ | Waiting times |

### Python Quick Reference

```python
from scipy import stats

# Uniform
uniform_rv = stats.uniform(loc=a, scale=b-a)

# Exponential
exp_rv = stats.expon(scale=1/lambda_param)

# Common methods
rv.pdf(x)   # Probability density at x
rv.cdf(x)   # P(X ≤ x)
rv.sf(x)    # P(X > x) = 1 - cdf(x)
rv.mean()   # E[X]
rv.var()    # Var(X)
rv.std()    # σ_X
rv.rvs(size=n)  # Generate n random samples
```

---

## 🎉 Congratulations!

You've completed **Statistics for Data Science I**!

You now have a solid foundation in:
✅ Descriptive statistics
✅ Probability theory
✅ Random variables and distributions
✅ Expected values and variance
✅ Common probability models

**Next:** Statistics for Data Science II - where these concepts power real statistical inference and machine learning!

---

**End of Week 12 Notes - End of Statistics I Course**
