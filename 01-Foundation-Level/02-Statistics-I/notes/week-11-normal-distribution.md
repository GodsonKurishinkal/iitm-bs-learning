# Week 11: Binomial and Poisson Distributions

---
**Date**: 2025-11-22
**Course**: BSMA1002 - Statistics for Data Science I
**Level**: Foundation
**Week**: 11 of 12
**Source**: IIT Madras Statistics I Week 11
**Topic Area**: Probability Distributions, Discrete Models
**Tags**: #BSMA1002 #Binomial #Poisson #BernoulliTrials #Week11 #Foundation
---

## 📋 Bottom Line Up Front (BLUF)

**What**: Binomial and Poisson distributions are two fundamental discrete probability models - Binomial models the number of successes in $n$ independent trials, while Poisson models rare events occurring at a constant average rate.

**Why**: These distributions are ubiquitous in data science - from A/B testing (Binomial) to modeling website traffic, defects, and rare events (Poisson). They provide closed-form formulas for PMF, expectation, and variance.

**Key Takeaway**: Binomial requires fixed $n$ trials with constant success probability $p$; Poisson models events with average rate $\lambda$ over time/space. Together they cover most discrete counting scenarios in practice.

---

## 🎯 Learning Objectives

By the end of this week, you should be able to:

1. ✅ Define Bernoulli trials and independent identically distributed (i.i.d.) random variables
2. ✅ Recognize when to apply Binomial distribution and compute probabilities
3. ✅ Calculate Binomial expectation ($np$) and variance ($np(1-p)$)
4. ✅ Understand Poisson distribution as limiting case of Binomial
5. ✅ Apply Poisson distribution to model rare events
6. ✅ Implement both distributions in Python using scipy.stats
7. ✅ Choose between Binomial and Poisson for real-world problems

---

## 📚 Table of Contents

1. [Bernoulli Trials](#bernoulli-trials)
2. [Independent and Identically Distributed (i.i.d.)](#independent-and-identically-distributed-iid)
3. [Binomial Distribution](#binomial-distribution)
4. [Binomial Expectation and Variance](#binomial-expectation-and-variance)
5. [Poisson Distribution](#poisson-distribution)
6. [Poisson as Limit of Binomial](#poisson-as-limit-of-binomial)
7. [Data Science Applications](#data-science-applications)
8. [Common Pitfalls](#common-pitfalls)
9. [Python Implementation](#python-implementation)
10. [Practice Problems](#practice-problems)

---

## 🎲 Bernoulli Trials

### Definition

**Definition:** A **Bernoulli trial** is a random experiment with exactly two possible outcomes:
- **Success** (often coded as 1) with probability $p$
- **Failure** (often coded as 0) with probability $1 - p$ or $q$

**Bernoulli Random Variable:**
$$X = \begin{cases}
1 & \text{with probability } p \\
0 & \text{with probability } 1-p
\end{cases}$$

**Notation:** $X \sim \text{Bernoulli}(p)$

### Properties of Bernoulli

**PMF:**
$$P(X = x) = \begin{cases}
p & \text{if } x = 1 \\
1-p & \text{if } x = 0 \\
0 & \text{otherwise}
\end{cases}$$

**Compact form:** $P(X = x) = p^x (1-p)^{1-x}$ for $x \in \{0, 1\}$

**Expectation:**
$$E[X] = 0 \cdot (1-p) + 1 \cdot p = p$$

**Variance:**
$$\text{Var}(X) = p(1-p)$$

**Proof:**
$$E[X^2] = 0^2(1-p) + 1^2(p) = p$$
$$\text{Var}(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1-p)$$

### Example 1: Coin Flip

**Experiment:** Flip a biased coin with $P(\text{Heads}) = 0.6$

**Bernoulli variable:** $X = 1$ if Heads, $X = 0$ if Tails

**Probability:** $p = 0.6$, $q = 1-p = 0.4$

**Expectation:** $E[X] = 0.6$

**Variance:** $\text{Var}(X) = 0.6 \times 0.4 = 0.24$

**Interpretation:** On average, the outcome is 0.6 (closer to 1 than 0, as expected for biased coin).

### Example 2: Customer Conversion

**Experiment:** Show ad to customer, observe if they convert (purchase).

**Bernoulli variable:** $X = 1$ if convert, $X = 0$ if don't convert

**Historical conversion rate:** $p = 0.08$ (8%)

**Expectation:** $E[X] = 0.08$

**Variance:** $\text{Var}(X) = 0.08 \times 0.92 = 0.0736$

**Interpretation:** Most customers don't convert (high variance reflects this uncertainty).

---

## 🔗 Independent and Identically Distributed (i.i.d.)

### Definition

**Independent:** Events/trials where the outcome of one does not affect the outcome of another.

**Identically Distributed:** All trials have the same probability distribution (same $p$).

**i.i.d. Random Variables:** A sequence $X_1, X_2, ..., X_n$ where:
1. Each $X_i$ has the same distribution
2. The $X_i$ are mutually independent

**Purpose:** i.i.d. assumption is fundamental for Binomial distribution and most statistical inference.

### Example 3: Coin Flips

**Experiment:** Flip the same coin 10 times.

**Random variables:** $X_1, X_2, ..., X_{10}$ where $X_i$ = result of $i$-th flip

**Identically distributed:** Each $X_i \sim \text{Bernoulli}(p)$ with same $p$

**Independent:** Outcome of flip 5 doesn't affect flip 6

**Conclusion:** $X_1, ..., X_{10}$ are i.i.d. Bernoulli$(p)$ random variables.

### Example 4: When i.i.d. Fails

**Scenario 1: Not identical**
- Flip coin 1 (fair, $p=0.5$) then flip coin 2 (biased, $p=0.7$)
- Independent ✓, but NOT identically distributed ✗

**Scenario 2: Not independent**
- Draw 2 cards from deck without replacement
- First card affects second (dependent) ✗
- (With replacement would be independent ✓)

**Scenario 3: Survey sampling**
- Sample 100 people from population without replacement
- Technically not independent, but if population >> 100, approximately i.i.d. ✓

**Key insight:** i.i.d. is often an idealization, but a useful one when:
- Population is large relative to sample
- Replacement is used
- Trials are truly isolated

---

## 📊 Binomial Distribution

### Definition

**Setup:** Perform $n$ independent Bernoulli trials, each with success probability $p$.

**Random Variable:** $X$ = number of successes in $n$ trials

**Notation:** $X \sim \text{Binomial}(n, p)$ or $X \sim B(n, p)$

**Parameters:**
- $n$: number of trials (positive integer)
- $p$: probability of success on each trial ($0 \leq p \leq 1$)

### Probability Mass Function (PMF)

$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

for $k = 0, 1, 2, ..., n$

**Where:**
- $\binom{n}{k} = \frac{n!}{k!(n-k)!}$ is the binomial coefficient ("n choose k")
- $p^k$ = probability of $k$ successes
- $(1-p)^{n-k}$ = probability of $n-k$ failures
- $\binom{n}{k}$ counts the number of ways to arrange $k$ successes among $n$ trials

### Derivation Logic

**Question:** What's $P(X = k)$ = probability of exactly $k$ successes?

**Step 1:** Probability of a specific sequence with $k$ successes:
$$p \cdot p \cdots p \cdot (1-p) \cdot (1-p) \cdots (1-p) = p^k (1-p)^{n-k}$$

**Step 2:** How many such sequences? Choose which $k$ positions (out of $n$) are successes:
$$\binom{n}{k} = \frac{n!}{k!(n-k)!}$$

**Step 3:** Multiply:
$$P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$$

### Example 5: Coin Flips - Binomial

**Experiment:** Flip fair coin 5 times. Find $P(\text{exactly 3 heads})$.

**Random Variable:** $X$ = number of heads $\sim \text{Binomial}(5, 0.5)$

**Calculate:** $P(X = 3)$

$$\begin{align}
P(X = 3) &= \binom{5}{3} (0.5)^3 (0.5)^{5-3} \\
&= \frac{5!}{3!2!} (0.5)^3 (0.5)^2 \\
&= \frac{5 \times 4}{2 \times 1} \times (0.5)^5 \\
&= 10 \times \frac{1}{32} \\
&= \frac{10}{32} = 0.3125
\end{align}$$

**Interpretation:** 31.25% chance of exactly 3 heads in 5 flips.

### Example 6: Quality Control

**Problem:** Manufacturing process produces 2% defective items. Inspect 20 items. What's probability of finding exactly 1 defect?

**Setup:**
- $n = 20$ items
- $p = 0.02$ (defect probability)
- $X$ = number of defects $\sim \text{Binomial}(20, 0.02)$

**Calculate:** $P(X = 1)$

$$\begin{align}
P(X = 1) &= \binom{20}{1} (0.02)^1 (0.98)^{19} \\
&= 20 \times 0.02 \times (0.98)^{19} \\
&= 0.4 \times 0.6676 \\
&= 0.267
\end{align}$$

**Interpretation:** 26.7% chance of exactly 1 defect.

**Also useful:** $P(X = 0)$ (no defects)

$$P(X = 0) = \binom{20}{0} (0.02)^0 (0.98)^{20} = 1 \times 1 \times 0.6676 = 0.668$$

66.8% chance of no defects in the sample.

### Example 7: Cumulative Probabilities

**Problem:** Website has 10% conversion rate. Show ad to 30 users. What's probability of at least 2 conversions?

**Setup:** $X \sim \text{Binomial}(30, 0.10)$

**Want:** $P(X \geq 2) = 1 - P(X < 2) = 1 - [P(X=0) + P(X=1)]$

$$P(X = 0) = \binom{30}{0} (0.1)^0 (0.9)^{30} = (0.9)^{30} = 0.0424$$

$$P(X = 1) = \binom{30}{1} (0.1)^1 (0.9)^{29} = 30 \times 0.1 \times (0.9)^{29} = 0.1413$$

$$P(X \geq 2) = 1 - (0.0424 + 0.1413) = 1 - 0.1837 = 0.8163$$

**Interpretation:** 81.63% chance of at least 2 conversions.

**Practical note:** For large $n$, compute using CDF tables or software (scipy.stats).

---

## 📈 Binomial Expectation and Variance

### Expectation

**Theorem:** If $X \sim \text{Binomial}(n, p)$, then:
$$E[X] = np$$

**Intuitive reasoning:**
- Each trial contributes $p$ on average (Bernoulli expectation)
- $n$ trials → total expectation = $n \times p$

**Rigorous proof:** Let $X = X_1 + X_2 + \cdots + X_n$ where $X_i \sim \text{Bernoulli}(p)$ (i.i.d.)

$$E[X] = E[X_1 + X_2 + \cdots + X_n] = E[X_1] + E[X_2] + \cdots + E[X_n] = p + p + \cdots + p = np$$

### Variance

**Theorem:** If $X \sim \text{Binomial}(n, p)$, then:
$$\text{Var}(X) = np(1-p)$$

**Proof:** Using $X = X_1 + \cdots + X_n$ where $X_i$ are i.i.d. Bernoulli$(p)$:

$$\text{Var}(X) = \text{Var}(X_1) + \cdots + \text{Var}(X_n)$$

(Variance adds for independent variables)

$$= p(1-p) + p(1-p) + \cdots + p(1-p) = n \cdot p(1-p)$$

**Standard deviation:**
$$\sigma_X = \sqrt{np(1-p)}$$

### Properties

**Symmetry:** If $p = 0.5$, distribution is symmetric around $n/2$.

**Skewness:**
- If $p < 0.5$, distribution skewed right (toward higher values)
- If $p > 0.5$, distribution skewed left (toward lower values)

**Variance maximized** when $p = 0.5$ (maximum uncertainty).

### Example 8: Expected Conversions

**Problem:** E-commerce site shows ads to 1000 users with 5% conversion rate. How many conversions expected? What's the standard deviation?

**Setup:** $X \sim \text{Binomial}(1000, 0.05)$

**Expectation:**
$$E[X] = np = 1000 \times 0.05 = 50$$

Expected 50 conversions.

**Variance:**
$$\text{Var}(X) = np(1-p) = 1000 \times 0.05 \times 0.95 = 47.5$$

**Standard Deviation:**
$$\sigma_X = \sqrt{47.5} \approx 6.89$$

**Interpretation:** Expect about 50 conversions, typically varying by ±7.

**Confidence interval (rough):** $[50 - 2(7), 50 + 2(7)] = [36, 64]$ covers about 95% (empirical rule).

### Example 9: Comparing Scenarios

**Scenario A:** 100 trials, $p = 0.5$
- $E[X] = 100(0.5) = 50$
- $\text{Var}(X) = 100(0.5)(0.5) = 25$
- $\sigma_X = 5$

**Scenario B:** 100 trials, $p = 0.1$
- $E[X] = 100(0.1) = 10$
- $\text{Var}(X) = 100(0.1)(0.9) = 9$
- $\sigma_X = 3$

**Observation:** Lower $p$ → lower mean AND lower variance (less uncertainty in absolute terms).

**Coefficient of variation** (relative uncertainty): $\frac{\sigma}{E[X]}$
- Scenario A: $\frac{5}{50} = 0.10$ (10%)
- Scenario B: $\frac{3}{10} = 0.30$ (30%)

Scenario B has higher relative uncertainty!

---

## 🌟 Poisson Distribution

### Definition

**Definition:** The **Poisson distribution** models the number of events occurring in a fixed interval of time or space, when events occur at a constant average rate and independently.

**Random Variable:** $X$ = number of events in the interval

**Notation:** $X \sim \text{Poisson}(\lambda)$

**Parameter:**
- $\lambda$ (lambda): average rate of events per interval ($\lambda > 0$)

### Probability Mass Function (PMF)

$$P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}$$

for $k = 0, 1, 2, 3, ...$

**Where:**
- $e \approx 2.71828$ (Euler's number)
- $\lambda^k$ = rate parameter to the power $k$
- $k!$ = factorial of $k$

**Support:** $X$ can be 0, 1, 2, ..., ∞ (no upper limit, unlike Binomial)

### When to Use Poisson

**Criteria:**
1. **Count data:** Number of occurrences (not continuous)
2. **Fixed interval:** Time period or spatial region specified
3. **Constant rate:** Events occur at average rate $\lambda$
4. **Independence:** Occurrences don't affect each other
5. **Rare events:** $\lambda$ is relatively small

**Classic examples:**
- Number of calls to call center per hour
- Number of typos per page
- Number of arrivals at a store per day
- Number of server crashes per week
- Number of emails received per hour

### Example 10: Call Center

**Problem:** Call center receives average 4 calls per minute. What's probability of exactly 3 calls in next minute?

**Setup:** $X \sim \text{Poisson}(4)$ (λ = 4)

**Calculate:** $P(X = 3)$

$$\begin{align}
P(X = 3) &= \frac{e^{-4} \cdot 4^3}{3!} \\
&= \frac{e^{-4} \cdot 64}{6} \\
&= \frac{0.0183 \times 64}{6} \\
&= \frac{1.171}{6} \\
&= 0.195
\end{align}$$

**Interpretation:** 19.5% chance of exactly 3 calls.

### Poisson Expectation and Variance

**Expectation:**
$$E[X] = \lambda$$

**Variance:**
$$\text{Var}(X) = \lambda$$

**Key property:** For Poisson, mean equals variance!

$$E[X] = \text{Var}(X) = \lambda$$

**Standard deviation:**
$$\sigma_X = \sqrt{\lambda}$$

### Example 11: Website Traffic

**Problem:** Website averages 9 visitors per minute.

**Setup:** $X \sim \text{Poisson}(9)$

**Calculate:**

**Mean:** $E[X] = 9$ visitors/minute

**Variance:** $\text{Var}(X) = 9$ (visitors²)

**Standard deviation:** $\sigma_X = \sqrt{9} = 3$ visitors

**Interpretation:** Expect 9 visitors/min, typically varying by ±3.

**Probability calculations:**

$$P(X = 9) = \frac{e^{-9} \cdot 9^9}{9!} = \frac{0.000123 \times 387,420,489}{362,880} = 0.132$$

13.2% chance of exactly the mean (9 visitors).

$$P(X \leq 5) = \sum_{k=0}^{5} \frac{e^{-9} \cdot 9^k}{k!}$$

(Use software for cumulative probabilities)

### Example 12: Rare Events

**Problem:** Book has average 0.5 typos per page. What's probability a page has no typos?

**Setup:** $X \sim \text{Poisson}(0.5)$

**Calculate:** $P(X = 0)$

$$P(X = 0) = \frac{e^{-0.5} \cdot 0.5^0}{0!} = e^{-0.5} \cdot 1 = 0.6065$$

**Interpretation:** 60.65% of pages have no typos.

**Probability of at least 1 typo:**
$$P(X \geq 1) = 1 - P(X = 0) = 1 - 0.6065 = 0.3935$$

39.35% of pages have at least one typo.

---

## 🔄 Poisson as Limit of Binomial

### The Connection

**Key insight:** Poisson distribution arises as a limiting case of Binomial when:
- $n \to \infty$ (many trials)
- $p \to 0$ (rare events)
- $np = \lambda$ remains constant (fixed average rate)

**Formally:** If $X_n \sim \text{Binomial}(n, p_n)$ where $np_n \to \lambda$ as $n \to \infty$, then:

$$\lim_{n \to \infty} P(X_n = k) = \frac{e^{-\lambda} \lambda^k}{k!}$$

### When to Approximate Binomial with Poisson

**Rule of thumb:** Use Poisson approximation when:
- $n \geq 20$ (large number of trials)
- $p \leq 0.05$ (rare events)
- $np < 10$ (moderate expected count)

**Advantage:** Poisson calculation easier than Binomial for large $n$.

### Example 13: Defect Modeling

**Problem:** Factory produces 10,000 items per day with 0.03% defect rate. Approximate probability of exactly 2 defects.

**Exact (Binomial):** $X \sim \text{Binomial}(10000, 0.0003)$

$$P(X = 2) = \binom{10000}{2} (0.0003)^2 (0.9997)^{9998}$$

(Very tedious to compute!)

**Approximation (Poisson):**
$$\lambda = np = 10000 \times 0.0003 = 3$$

$$X \approx \text{Poisson}(3)$$

$$P(X = 2) = \frac{e^{-3} \cdot 3^2}{2!} = \frac{0.0498 \times 9}{2} = 0.224$$

**Interpretation:** 22.4% chance of exactly 2 defects.

**Verification:** Approximation excellent when $n$ large, $p$ small.

### Example 14: Comparing Binomial and Poisson

**Scenario:** $n = 100$, $p = 0.02$, $\lambda = np = 2$

**Binomial $X \sim B(100, 0.02)$:**
- $E[X] = 2$
- $\text{Var}(X) = 100(0.02)(0.98) = 1.96$

**Poisson $Y \sim \text{Poisson}(2)$:**
- $E[Y] = 2$
- $\text{Var}(Y) = 2$

**Probabilities:**

| $k$ | Binomial | Poisson | Difference |
|-----|----------|---------|------------|
| 0 | 0.1326 | 0.1353 | 0.0027 |
| 1 | 0.2707 | 0.2707 | 0.0000 |
| 2 | 0.2734 | 0.2707 | -0.0027 |
| 3 | 0.1823 | 0.1804 | -0.0019 |

Very close agreement! ✓

---

## 💼 Data Science Applications

### Application 1: A/B Testing with Binomial

**Problem:** Test new website design. Show to 500 users, observe conversions.

**Control group:** 500 users, historical conversion rate 10%
- $X_C \sim \text{Binomial}(500, 0.10)$
- $E[X_C] = 50$, $\sigma_{X_C} = \sqrt{500(0.1)(0.9)} = 6.71$

**Treatment group:** 500 users, new design
- $X_T \sim \text{Binomial}(500, p_T)$ where $p_T$ is unknown

**Observed:** $X_T = 65$ conversions

**Question:** Is this significantly different from expected 50?

**Analysis:**
$$z = \frac{X_T - E[X_C]}{\sigma_{X_C}} = \frac{65 - 50}{6.71} = 2.24$$

With $z = 2.24 > 2$, this is significant (more than 2 standard deviations above mean).

**Conclusion:** New design likely increases conversions (statistical significance achieved).

### Application 2: Server Monitoring with Poisson

**Problem:** Monitor server errors. Historically, 2 errors per hour on average.

**Model:** $X \sim \text{Poisson}(2)$ (errors per hour)

**Alerting rule:** Alert if $X \geq 5$ errors in an hour.

**Calculate false alarm rate:**
$$P(X \geq 5) = 1 - P(X \leq 4) = 1 - \sum_{k=0}^{4} \frac{e^{-2} \cdot 2^k}{k!}$$

Using software: $P(X \geq 5) \approx 0.053$ (5.3%)

**Interpretation:** 5.3% chance of false alarm (alerting when system is normal).

**Decision:** Acceptable false alarm rate for early detection.

### Application 3: Inventory Management

**Problem:** Store sells average 3 units per day of a product.

**Model:** Daily demand $D \sim \text{Poisson}(3)$

**Question:** How much stock to keep to satisfy demand with 95% probability?

**Find:** $k$ such that $P(D \leq k) \geq 0.95$

**Using Poisson CDF:**
- $P(D \leq 5) = 0.916$ (not enough)
- $P(D \leq 6) = 0.966$ ✓ (exceeds 95%)

**Decision:** Keep 6 units in stock daily.

**Expected daily shortage:**
$$E[\max(D - 6, 0)] = \sum_{k=7}^{\infty} (k-6) \frac{e^{-3} \cdot 3^k}{k!} \approx 0.03$$

About 0.03 units short per day on average (very low).

### Application 4: Feature Selection in ML

**Problem:** Training binary classifier with 10,000 features, 100 samples.

**Random chance:** If features are noise, each has $p = 0.05$ chance of appearing significant (Type I error rate α = 0.05).

**Model:** $X \sim \text{Binomial}(10000, 0.05)$ = number of false positives

**Expected false discoveries:**
$$E[X] = 10000 \times 0.05 = 500$$

**Ouch!** 500 features would appear significant by chance alone.

**Solution:** Bonferroni correction - use $\alpha^* = \frac{0.05}{10000} = 0.000005$

Now: $X \sim \text{Binomial}(10000, 0.000005)$

Approximate with Poisson: $\lambda = 10000 \times 0.000005 = 0.05$

$$P(X = 0) = e^{-0.05} \approx 0.951$$

95.1% chance of zero false positives - much better! ✓

---

## ⚠️ Common Pitfalls

### ❌ Pitfall 1: Confusing Binomial and Poisson

**Wrong:** "10 customers per hour, each converts with probability 0.1 → Binomial(10, 0.1)"

**Right:** If customers arrive randomly (not fixed 10), use Poisson:
- Average conversions per hour: $\lambda = 10 \times 0.1 = 1$
- $X \sim \text{Poisson}(1)$

**When to use which:**
- **Binomial:** Fixed $n$ trials (e.g., flip coin 10 times)
- **Poisson:** Random number of events in interval (e.g., phone calls per hour)

### ❌ Pitfall 2: Wrong Parameter for Poisson

**Wrong:** "5% of users convert, 100 users → Poisson(0.05)"

**Right:** Poisson parameter is the **rate** (expected count), not probability:
- Expected conversions: $\lambda = 100 \times 0.05 = 5$
- $X \sim \text{Poisson}(5)$

**Remember:** $\lambda = np$ (expected count, not $p$)

### ❌ Pitfall 3: Binomial Variance

**Wrong:** "$X \sim \text{Binomial}(100, 0.5)$, so $\text{Var}(X) = np = 50$"

**Right:**
$$\text{Var}(X) = np(1-p) = 100 \times 0.5 \times 0.5 = 25$$

Don't forget the $(1-p)$ term!

### ❌ Pitfall 4: Non-integer Binomial

**Wrong:** "$X \sim \text{Binomial}(100, 0.5)$, so $P(X = 50.5) = ?$"

**Right:** Binomial is discrete - only integer values possible.
$$P(X = 50.5) = 0$$

### ❌ Pitfall 5: Independence Violation

**Wrong:** "Sample 10 items from batch of 50 without replacement → Binomial"

**Right:** Without replacement violates independence. Use **hypergeometric distribution** instead.

**Exception:** If population >> sample (e.g., 1000 vs 10), Binomial approximation OK.

---

## 💻 Python Implementation

### Implementation 1: Binomial Distribution

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# Define Binomial random variable: n=10 trials, p=0.3
n, p = 10, 0.3
binom_rv = stats.binom(n, p)

# PMF
k = 3
prob_k = binom_rv.pmf(k)
print(f"P(X = {k}) = {prob_k:.4f}")  # 0.2668

# CDF
prob_leq_k = binom_rv.cdf(k)
print(f"P(X ≤ {k}) = {prob_leq_k:.4f}")  # 0.6496

# Expectation and variance
print(f"E[X] = {binom_rv.mean():.2f}")  # 3.00 (= np)
print(f"Var(X) = {binom_rv.var():.2f}")  # 2.10 (= np(1-p))
print(f"σ_X = {binom_rv.std():.2f}")  # 1.45

# Generate random samples
samples = binom_rv.rvs(size=1000)
print(f"Sample mean: {np.mean(samples):.2f}")  # ≈ 3.00
print(f"Sample std: {np.std(samples, ddof=1):.2f}")  # ≈ 1.45
```

### Implementation 2: Visualizing Binomial PMF

```python
# Plot PMF
x_values = np.arange(0, n+1)
pmf_values = binom_rv.pmf(x_values)

plt.figure(figsize=(10, 6))
plt.bar(x_values, pmf_values, color='skyblue', edgecolor='black', alpha=0.7)
plt.axvline(binom_rv.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean = {binom_rv.mean():.1f}')
plt.xlabel('Number of Successes (k)')
plt.ylabel('Probability P(X = k)')
plt.title(f'Binomial PMF: n={n}, p={p}')
plt.xticks(x_values)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Cumulative probabilities
print(f"P(X ≤ 2) = {binom_rv.cdf(2):.4f}")  # 0.3828
print(f"P(X ≥ 5) = {1 - binom_rv.cdf(4):.4f}")  # 0.1503
print(f"P(2 ≤ X ≤ 5) = {binom_rv.cdf(5) - binom_rv.cdf(1):.4f}")  # 0.7711
```

### Implementation 3: Poisson Distribution

```python
# Define Poisson random variable: λ=4
lambda_param = 4
poisson_rv = stats.poisson(lambda_param)

# PMF
k = 3
prob_k = poisson_rv.pmf(k)
print(f"P(X = {k}) = {prob_k:.4f}")  # 0.1954

# CDF
prob_leq_k = poisson_rv.cdf(k)
print(f"P(X ≤ {k}) = {prob_leq_k:.4f}")  # 0.4335

# Expectation and variance
print(f"E[X] = {poisson_rv.mean():.2f}")  # 4.00 (= λ)
print(f"Var(X) = {poisson_rv.var():.2f}")  # 4.00 (= λ)
print(f"σ_X = {poisson_rv.std():.2f}")  # 2.00

# Generate samples
samples = poisson_rv.rvs(size=1000)
print(f"Sample mean: {np.mean(samples):.2f}")  # ≈ 4.00
```

### Implementation 4: Comparing Binomial and Poisson

```python
# Large n, small p scenario
n_large = 1000
p_small = 0.003
lambda_approx = n_large * p_small  # 3.0

binom_large = stats.binom(n_large, p_small)
poisson_approx = stats.poisson(lambda_approx)

# Compare probabilities
k_values = np.arange(0, 10)
binom_probs = binom_large.pmf(k_values)
poisson_probs = poisson_approx.pmf(k_values)

plt.figure(figsize=(12, 6))
x_pos = np.arange(len(k_values))
width = 0.35

plt.bar(x_pos - width/2, binom_probs, width, label=f'Binomial({n_large}, {p_small})', alpha=0.7)
plt.bar(x_pos + width/2, poisson_probs, width, label=f'Poisson({lambda_approx})', alpha=0.7)

plt.xlabel('k')
plt.ylabel('Probability')
plt.title('Poisson Approximation to Binomial')
plt.xticks(x_pos, k_values)
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# Print comparison
print("k\tBinomial\tPoisson\t\tDifference")
for k in k_values:
    b = binom_large.pmf(k)
    p_val = poisson_approx.pmf(k)
    print(f"{k}\t{b:.6f}\t{p_val:.6f}\t{abs(b-p_val):.6f}")
```

### Implementation 5: A/B Test Simulation

```python
# Simulate A/B test
np.random.seed(42)

# Control: n=500, p=0.10
n_control = 500
p_control = 0.10
control_conversions = stats.binom(n_control, p_control).rvs()

# Treatment: n=500, p=0.12 (2% lift)
n_treatment = 500
p_treatment = 0.12
treatment_conversions = stats.binom(n_treatment, p_treatment).rvs()

print(f"Control conversions: {control_conversions}")
print(f"Treatment conversions: {treatment_conversions}")
print(f"Difference: {treatment_conversions - control_conversions}")

# Statistical test
from scipy.stats import norm

# Expected under null (no difference)
expected_control = n_control * p_control
std_control = np.sqrt(n_control * p_control * (1 - p_control))

z_score = (treatment_conversions - expected_control) / std_control
p_value = 1 - norm.cdf(z_score)  # One-sided test

print(f"\nZ-score: {z_score:.2f}")
print(f"P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Result: Statistically significant difference! ✓")
else:
    print("Result: Not statistically significant.")
```

---

## 📝 Practice Problems

### Basic Level

**Problem 1:** $X \sim \text{Binomial}(15, 0.4)$

(a) Calculate $P(X = 6)$
(b) Find $E[X]$ and $\text{Var}(X)$

<details>
<summary>Solution</summary>

(a) $$P(X = 6) = \binom{15}{6} (0.4)^6 (0.6)^9 = 5005 \times 0.004096 \times 0.010078 = 0.2066$$

(b) $$E[X] = np = 15 \times 0.4 = 6$$
$$\text{Var}(X) = np(1-p) = 15 \times 0.4 \times 0.6 = 3.6$$
</details>

**Problem 2:** $Y \sim \text{Poisson}(3)$

(a) Calculate $P(Y = 2)$
(b) Find $E[Y]$ and $\text{Var}(Y)$

<details>
<summary>Solution</summary>

(a) $$P(Y = 2) = \frac{e^{-3} \cdot 3^2}{2!} = \frac{0.0498 \times 9}{2} = 0.224$$

(b) $$E[Y] = \lambda = 3$$
$$\text{Var}(Y) = \lambda = 3$$
</details>

**Problem 3:** Flip a fair coin 8 times. What's the probability of exactly 5 heads?

<details>
<summary>Solution</summary>

$X \sim \text{Binomial}(8, 0.5)$

$$P(X = 5) = \binom{8}{5} (0.5)^5 (0.5)^3 = 56 \times (0.5)^8 = 56 \times \frac{1}{256} = \frac{56}{256} = 0.219$$

21.9% chance.
</details>

### Intermediate Level

**Problem 4:** A call center receives average 6 calls per 10 minutes.

(a) What distribution models number of calls in 10 min?
(b) Calculate probability of exactly 4 calls.
(c) What's probability of at least 8 calls?

<details>
<summary>Solution</summary>

(a) $X \sim \text{Poisson}(6)$

(b) $$P(X = 4) = \frac{e^{-6} \cdot 6^4}{4!} = \frac{0.00248 \times 1296}{24} = 0.134$$

(c) $$P(X \geq 8) = 1 - P(X \leq 7) = 1 - \sum_{k=0}^{7} \frac{e^{-6} \cdot 6^k}{k!}$$

Using software: $P(X \geq 8) \approx 0.256$
</details>

**Problem 5:** Manufacturing produces 5000 items with 0.04% defect rate.

(a) Should you use Binomial or Poisson? Why?
(b) Find expected number of defects.
(c) Probability of zero defects?

<details>
<summary>Solution</summary>

(a) Poisson appropriate: $n = 5000$ large, $p = 0.0004$ small

$\lambda = np = 5000 \times 0.0004 = 2$

(b) $E[X] = \lambda = 2$ defects

(c) $$P(X = 0) = \frac{e^{-2} \cdot 2^0}{0!} = e^{-2} = 0.135$$

13.5% chance of zero defects.
</details>

### Advanced Level

**Problem 6:** A/B test: 200 users each group. Control converts at 15%, treatment at unknown rate.

Observed: 40 conversions in treatment group.

(a) Find expected conversions and standard deviation for control.
(b) Compute z-score for observed treatment result.
(c) Is the result statistically significant (α = 0.05)?

<details>
<summary>Solution</summary>

(a) Control: $X_C \sim \text{Binomial}(200, 0.15)$

$$E[X_C] = 200 \times 0.15 = 30$$
$$\sigma_{X_C} = \sqrt{200 \times 0.15 \times 0.85} = \sqrt{25.5} = 5.05$$

(b) Treatment observed: 40 conversions

$$z = \frac{40 - 30}{5.05} = \frac{10}{5.05} = 1.98$$

(c) Critical value for α = 0.05 (one-sided): $z_{0.05} = 1.645$

Since $z = 1.98 > 1.645$, result is statistically significant ✓

Conclude: Treatment likely improves conversions.
</details>

**Problem 7:** Server errors follow Poisson(λ=2 per hour). System alerts if ≥5 errors in an hour.

(a) What's the false alarm rate (alerting when system normal)?
(b) If actual rate increases to λ=6, what's probability of detection (alert triggers)?

<details>
<summary>Solution</summary>

(a) Normal: $X \sim \text{Poisson}(2)$

$$P(\text{false alarm}) = P(X \geq 5) = 1 - P(X \leq 4)$$

$$= 1 - \sum_{k=0}^{4} \frac{e^{-2} \cdot 2^k}{k!} \approx 1 - 0.947 = 0.053$$

5.3% false alarm rate.

(b) Abnormal: $Y \sim \text{Poisson}(6)$

$$P(\text{detection}) = P(Y \geq 5) = 1 - P(Y \leq 4)$$

$$= 1 - \sum_{k=0}^{4} \frac{e^{-6} \cdot 6^k}{k!} \approx 1 - 0.285 = 0.715$$

71.5% detection rate (power of test).
</details>

---

## 🎓 Self-Assessment

**Conceptual:**
- [ ] Can you explain when to use Binomial vs Poisson?
- [ ] Do you understand i.i.d. assumption and why it matters?
- [ ] Can you interpret $\lambda$ in Poisson distribution?
- [ ] Why does Poisson have mean equal to variance?

**Computational:**
- [ ] Can you calculate Binomial probabilities by hand (small $n$)?
- [ ] Can you compute $E[X]$ and $\text{Var}(X)$ for Binomial and Poisson?
- [ ] Can you use Poisson to approximate Binomial?

**Application:**
- [ ] Can you model real-world problems with these distributions?
- [ ] Can you perform basic hypothesis testing with Binomial?
- [ ] Can you use Python scipy.stats effectively?

---

## 📚 Quick Reference

### Binomial Distribution

| Property | Formula |
|----------|---------|
| Notation | $X \sim \text{Binomial}(n, p)$ |
| PMF | $P(X = k) = \binom{n}{k} p^k (1-p)^{n-k}$ |
| Mean | $E[X] = np$ |
| Variance | $\text{Var}(X) = np(1-p)$ |
| Std Dev | $\sigma_X = \sqrt{np(1-p)}$ |

### Poisson Distribution

| Property | Formula |
|----------|---------|
| Notation | $X \sim \text{Poisson}(\lambda)$ |
| PMF | $P(X = k) = \frac{e^{-\lambda} \lambda^k}{k!}$ |
| Mean | $E[X] = \lambda$ |
| Variance | $\text{Var}(X) = \lambda$ |
| Std Dev | $\sigma_X = \sqrt{\lambda}$ |

### Python Quick Reference

```python
from scipy import stats

# Binomial
binom_rv = stats.binom(n, p)
binom_rv.pmf(k)  # P(X = k)
binom_rv.cdf(k)  # P(X ≤ k)

# Poisson
poisson_rv = stats.poisson(lambda_param)
poisson_rv.pmf(k)
poisson_rv.cdf(k)
```

---

## 🔜 Next Week Preview

**Week 12: Continuous Distributions**

Moving from discrete to continuous:
- **Uniform distribution**: Equal probability over interval
- **Exponential distribution**: Time until event occurs
- **Normal distribution preview**: The bell curve

---

**End of Week 11 Notes**
