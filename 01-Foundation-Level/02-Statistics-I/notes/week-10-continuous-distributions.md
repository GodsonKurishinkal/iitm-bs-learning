# Week 10: Expectation and Variance

---
**Date**: 2025-11-22
**Course**: BSMA1002 - Statistics for Data Science I
**Level**: Foundation
**Week**: 10 of 12
**Source**: IIT Madras Statistics I Week 10
**Topic Area**: Probability Theory, Expected Value
**Tags**: #BSMA1002 #Expectation #Variance #StandardDeviation #Week10 #Foundation
---

## 📋 Bottom Line Up Front (BLUF)

**What**: Expectation (expected value) $E[X]$ is the weighted average of a random variable, while variance $\text{Var}(X)$ measures how spread out the distribution is around the mean.

**Why**: These two numbers - expectation and variance - summarize an entire probability distribution, enabling quantitative decision-making under uncertainty in data science and machine learning.

**Key Takeaway**: Expectation tells us the "center" (long-run average), variance tells us the "spread" (risk/uncertainty). Together they're fundamental for statistical inference, model evaluation, and risk assessment.

---

## 🎯 Learning Objectives

By the end of this week, you should be able to:

1. ✅ Calculate expected value $E[X]$ for discrete random variables
2. ✅ Compute variance $\text{Var}(X)$ and standard deviation $\sigma_X$
3. ✅ Apply linearity properties: $E[aX + b]$, $\text{Var}(aX + b)$
4. ✅ Calculate expectation and variance of functions: $E[g(X)]$
5. ✅ Use expectation for decision-making under uncertainty
6. ✅ Implement calculations in Python and interpret results

---

## 📚 Table of Contents

1. [Expected Value: The Long-Run Average](#expected-value-the-long-run-average)
2. [Computing Expectation](#computing-expectation)
3. [Properties of Expectation](#properties-of-expectation)
4. [Variance and Standard Deviation](#variance-and-standard-deviation)
5. [Computing Variance](#computing-variance)
6. [Properties of Variance](#properties-of-variance)
7. [Data Science Applications](#data-science-applications)
8. [Common Pitfalls](#common-pitfalls)
9. [Python Implementation](#python-implementation)
10. [Practice Problems](#practice-problems)

---

## 🎲 Expected Value: The Long-Run Average

### Intuitive Understanding

**Definition:** The **expected value** (or **expectation** or **mean**) of a discrete random variable $X$ is:

$$E[X] = \sum_{x} x \cdot P(X = x) = \sum_{x} x \cdot p_X(x)$$

**Purpose:** $E[X]$ represents the long-run average value if we repeat the random experiment infinitely many times.

**Interpretation:** It's the "center of mass" of the probability distribution - where the distribution would balance if we placed weights at each value proportional to their probabilities.

### Physical Analogy

Imagine a seesaw with weights placed at different positions:
- Position = value of $x$
- Weight at position = probability $p_X(x)$

The expectation $E[X]$ is where you'd place the fulcrum to balance the seesaw.

### Example 1: Fair Die Roll

**Random Variable:** $X$ = outcome of rolling a fair six-sided die

**PMF:** $p_X(x) = \frac{1}{6}$ for $x \in \{1, 2, 3, 4, 5, 6\}$

**Expected Value:**
$$\begin{align}
E[X] &= \sum_{x=1}^{6} x \cdot P(X = x) \\
&= 1 \cdot \frac{1}{6} + 2 \cdot \frac{1}{6} + 3 \cdot \frac{1}{6} + 4 \cdot \frac{1}{6} + 5 \cdot \frac{1}{6} + 6 \cdot \frac{1}{6} \\
&= \frac{1 + 2 + 3 + 4 + 5 + 6}{6} \\
&= \frac{21}{6} = 3.5
\end{align}$$

**Interpretation:** If you roll a die many times and average the outcomes, you'll get approximately 3.5. Note that $E[X] = 3.5$ is not itself a possible outcome!

### Example 2: Biased Coin

**Random Variable:** $X$ = 1 if heads, 0 if tails; with $P(\text{Heads}) = 0.6$

**PMF:**
- $p_X(0) = 0.4$
- $p_X(1) = 0.6$

**Expected Value:**
$$E[X] = 0 \cdot 0.4 + 1 \cdot 0.6 = 0.6$$

**Interpretation:** The average outcome is 0.6. In 100 flips, we expect about 60 heads.

**Generalization:** For a Bernoulli random variable (0 or 1) with $P(X = 1) = p$:
$$E[X] = p$$

### Example 3: Custom Distribution

**Random Variable:** $X$ with PMF:

| $x$ | 1 | 2 | 5 | 10 |
|-----|---|---|---|----|
| $p_X(x)$ | 0.2 | 0.3 | 0.4 | 0.1 |

**Expected Value:**
$$\begin{align}
E[X] &= 1(0.2) + 2(0.3) + 5(0.4) + 10(0.1) \\
&= 0.2 + 0.6 + 2.0 + 1.0 \\
&= 3.8
\end{align}$$

**Interpretation:** On average, $X$ takes the value 3.8 (even though 3.8 itself is not a possible value).

### Key Insight: Weighted Average

The expectation is **not** the simple average of possible values. It's a **weighted** average where each value is weighted by its probability:

❌ **Wrong:** $\frac{1 + 2 + 5 + 10}{4} = 4.5$

✅ **Right:** $1(0.2) + 2(0.3) + 5(0.4) + 10(0.1) = 3.8$

More probable values contribute more to the expectation.

---

## 🧮 Computing Expectation

### Formula for Discrete Random Variables

For a discrete random variable $X$ with PMF $p_X(x)$:

$$E[X] = \sum_{\text{all } x} x \cdot p_X(x)$$

**Requirements:**
- Sum over all possible values of $X$
- Each term is $(\text{value}) \times (\text{probability})$
- The sum must converge (for infinite support)

### Example 4: Sum of Two Dice

**Random Variable:** $X$ = sum of two fair dice

**Approach 1: Direct calculation** using full PMF

Values: 2, 3, 4, ..., 12

$$\begin{align}
E[X] &= 2 \cdot \frac{1}{36} + 3 \cdot \frac{2}{36} + 4 \cdot \frac{3}{36} + 5 \cdot \frac{4}{36} + 6 \cdot \frac{5}{36} \\
&\quad + 7 \cdot \frac{6}{36} + 8 \cdot \frac{5}{36} + 9 \cdot \frac{4}{36} + 10 \cdot \frac{3}{36} + 11 \cdot \frac{2}{36} + 12 \cdot \frac{1}{36}
\end{align}$$

Computing:
$$E[X] = \frac{2 + 6 + 12 + 20 + 30 + 42 + 40 + 36 + 30 + 22 + 12}{36} = \frac{252}{36} = 7$$

**Approach 2: Using linearity** (we'll see this property next)

If $X = D_1 + D_2$ where $D_1$ and $D_2$ are two dice:
$$E[X] = E[D_1 + D_2] = E[D_1] + E[D_2] = 3.5 + 3.5 = 7$$

Much simpler!

### Example 5: Expectation of a Function

**Problem:** Given random variable $X$ with PMF, find $E[X^2]$ (expectation of the square).

**Random Variable:** $X$ with PMF:

| $x$ | 1 | 2 | 3 |
|-----|---|---|---|
| $p_X(x)$ | 0.5 | 0.3 | 0.2 |

**Formula for $E[g(X)]$:**
$$E[g(X)] = \sum_{\text{all } x} g(x) \cdot p_X(x)$$

**Calculate $E[X]$:**
$$E[X] = 1(0.5) + 2(0.3) + 3(0.2) = 0.5 + 0.6 + 0.6 = 1.7$$

**Calculate $E[X^2]$:**
$$E[X^2] = 1^2(0.5) + 2^2(0.3) + 3^2(0.2) = 1(0.5) + 4(0.3) + 9(0.2) = 0.5 + 1.2 + 1.8 = 3.5$$

**Note:** $E[X^2] \neq (E[X])^2$ !
- $E[X^2] = 3.5$
- $(E[X])^2 = (1.7)^2 = 2.89$

This difference is crucial for computing variance.

---

## 📐 Properties of Expectation

### Property 1: Linearity - Constants

**For any constant $c$:**
$$E[c] = c$$

**Proof:**
$$E[c] = \sum_{\text{all } x} c \cdot p_X(x) = c \sum_{\text{all } x} p_X(x) = c \cdot 1 = c$$

**Example:** $E[5] = 5$

### Property 2: Linearity - Scaling

**For any constant $a$:**
$$E[aX] = a \cdot E[X]$$

**Proof:**
$$E[aX] = \sum_{\text{all } x} (ax) \cdot p_X(x) = a \sum_{\text{all } x} x \cdot p_X(x) = a \cdot E[X]$$

**Example:** If $E[X] = 10$, then $E[3X] = 3 \cdot 10 = 30$

### Property 3: Linearity - Addition

**For any constants $a$ and $b$:**
$$E[aX + b] = a \cdot E[X] + b$$

**Proof:**
$$\begin{align}
E[aX + b] &= \sum_{\text{all } x} (ax + b) \cdot p_X(x) \\
&= \sum_{\text{all } x} ax \cdot p_X(x) + \sum_{\text{all } x} b \cdot p_X(x) \\
&= a \sum_{\text{all } x} x \cdot p_X(x) + b \sum_{\text{all } x} p_X(x) \\
&= a \cdot E[X] + b \cdot 1 \\
&= a \cdot E[X] + b
\end{align}$$

**Example:** Temperature conversion from Celsius to Fahrenheit: $F = \frac{9}{5}C + 32$

If $E[C] = 20$°C, then:
$$E[F] = \frac{9}{5}(20) + 32 = 36 + 32 = 68°F$$

### Property 4: Sum of Random Variables

**For any two random variables $X$ and $Y$:**
$$E[X + Y] = E[X] + E[Y]$$

**This holds whether or not $X$ and $Y$ are independent!**

**Example:** Two dice: $X = D_1 + D_2$
$$E[X] = E[D_1] + E[D_2] = 3.5 + 3.5 = 7$$

### Example 6: Applying Linearity

**Problem:** A data scientist's salary is $S = 50000 + 5000X$ where $X$ = years of experience (with $E[X] = 3$ years).

**Find:** $E[S]$

**Solution:**
$$E[S] = E[50000 + 5000X] = 50000 + 5000 \cdot E[X] = 50000 + 5000(3) = 65000$$

Expected salary: $\$65,000$.

### Example 7: Portfolio Expected Return

**Problem:** Investor puts 60% in stock A (expected return 8%) and 40% in stock B (expected return 12%).

**Random Variables:**
- $R_A$ = return on stock A, $E[R_A] = 0.08$
- $R_B$ = return on stock B, $E[R_B] = 0.12$

**Portfolio return:** $R_P = 0.6 R_A + 0.4 R_B$

**Expected portfolio return:**
$$E[R_P] = 0.6 E[R_A] + 0.4 E[R_B] = 0.6(0.08) + 0.4(0.12) = 0.048 + 0.048 = 0.096$$

Expected return: 9.6%.

---

## 📏 Variance and Standard Deviation

### Definition of Variance

**Definition:** The **variance** of a random variable $X$ measures the average squared deviation from the mean:

$$\text{Var}(X) = E[(X - \mu)^2]$$

where $\mu = E[X]$ is the mean.

**Purpose:** Quantifies how spread out the distribution is around the mean.
- Low variance → values clustered near mean
- High variance → values spread out

**Alternative formula** (computational formula):
$$\text{Var}(X) = E[X^2] - (E[X])^2$$

**Proof:**
$$\begin{align}
\text{Var}(X) &= E[(X - \mu)^2] \\
&= E[X^2 - 2\mu X + \mu^2] \\
&= E[X^2] - 2\mu E[X] + \mu^2 \\
&= E[X^2] - 2\mu \cdot \mu + \mu^2 \\
&= E[X^2] - \mu^2 \\
&= E[X^2] - (E[X])^2
\end{align}$$

### Standard Deviation

**Definition:** The **standard deviation** is the square root of variance:

$$\sigma_X = \text{SD}(X) = \sqrt{\text{Var}(X)}$$

**Purpose:** Measures spread in the same units as $X$ (whereas variance is in squared units).

**Interpretation:** Roughly the "average distance" from the mean.

### Example 8: Variance of Die Roll

**Random Variable:** $X$ = outcome of fair die roll

**We know:** $E[X] = 3.5$

**Calculate $E[X^2]$:**
$$E[X^2] = 1^2 \cdot \frac{1}{6} + 2^2 \cdot \frac{1}{6} + \cdots + 6^2 \cdot \frac{1}{6} = \frac{1 + 4 + 9 + 16 + 25 + 36}{6} = \frac{91}{6} \approx 15.17$$

**Variance:**
$$\text{Var}(X) = E[X^2] - (E[X])^2 = \frac{91}{6} - (3.5)^2 = 15.17 - 12.25 = 2.92$$

**Standard Deviation:**
$$\sigma_X = \sqrt{2.92} \approx 1.71$$

**Interpretation:** Outcomes typically deviate from 3.5 by about 1.71.

### Example 9: Variance of Bernoulli Variable

**Random Variable:** $X = 1$ with probability $p$, $X = 0$ with probability $1-p$

**Mean:** $E[X] = p$

**Calculate $E[X^2]$:**
$$E[X^2] = 0^2(1-p) + 1^2(p) = p$$

**Variance:**
$$\text{Var}(X) = E[X^2] - (E[X])^2 = p - p^2 = p(1-p)$$

**Key insight:** Variance is maximized when $p = 0.5$ (maximum uncertainty).

| $p$ | 0.1 | 0.3 | 0.5 | 0.7 | 0.9 |
|-----|-----|-----|-----|-----|-----|
| $\text{Var}(X)$ | 0.09 | 0.21 | 0.25 | 0.21 | 0.09 |

### Example 10: Comparing Two Distributions

**Distribution A:**

| $x$ | 10 | 20 | 30 |
|-----|----|----|---- |
| $p_X(x)$ | 0.25 | 0.50 | 0.25 |

**Distribution B:**

| $x$ | 0 | 20 | 40 |
|-----|---|----|----|
| $p_Y(y)$ | 0.25 | 0.50 | 0.25 |

**Calculate expectations:**
$$E[X] = 10(0.25) + 20(0.50) + 30(0.25) = 2.5 + 10 + 7.5 = 20$$
$$E[Y] = 0(0.25) + 20(0.50) + 40(0.25) = 0 + 10 + 10 = 20$$

**Same mean! But different spreads.**

**Calculate variances:**

For $X$:
$$E[X^2] = 100(0.25) + 400(0.50) + 900(0.25) = 25 + 200 + 225 = 450$$
$$\text{Var}(X) = 450 - 20^2 = 450 - 400 = 50$$
$$\sigma_X = \sqrt{50} \approx 7.07$$

For $Y$:
$$E[Y^2] = 0(0.25) + 400(0.50) + 1600(0.25) = 0 + 200 + 400 = 600$$
$$\text{Var}(Y) = 600 - 20^2 = 600 - 400 = 200$$
$$\sigma_Y = \sqrt{200} \approx 14.14$$

**Conclusion:** $Y$ has twice the variance → more spread out, higher risk.

---

## 🔧 Computing Variance

### Step-by-Step Process

**Method 1: Definition Formula**

1. Calculate $\mu = E[X]$
2. For each value $x$: compute $(x - \mu)^2$
3. Weight by $p_X(x)$ and sum: $\sum (x - \mu)^2 \cdot p_X(x)$

**Method 2: Computational Formula** (usually easier)

1. Calculate $E[X]$
2. Calculate $E[X^2] = \sum x^2 \cdot p_X(x)$
3. Compute $\text{Var}(X) = E[X^2] - (E[X])^2$

### Example 11: Complete Calculation

**Random Variable:** $X$ with PMF:

| $x$ | -2 | 0 | 3 |
|-----|----|----|---|
| $p_X(x)$ | 0.3 | 0.5 | 0.2 |

**Step 1:** Calculate $E[X]$
$$E[X] = (-2)(0.3) + 0(0.5) + 3(0.2) = -0.6 + 0 + 0.6 = 0$$

**Step 2:** Calculate $E[X^2]$
$$E[X^2] = (-2)^2(0.3) + 0^2(0.5) + 3^2(0.2) = 4(0.3) + 0 + 9(0.2) = 1.2 + 1.8 = 3.0$$

**Step 3:** Calculate $\text{Var}(X)$
$$\text{Var}(X) = E[X^2] - (E[X])^2 = 3.0 - 0^2 = 3.0$$

**Step 4:** Calculate $\sigma_X$
$$\sigma_X = \sqrt{3.0} \approx 1.73$$

**Verification using definition:**
$$\begin{align}
\text{Var}(X) &= (-2 - 0)^2(0.3) + (0 - 0)^2(0.5) + (3 - 0)^2(0.2) \\
&= 4(0.3) + 0(0.5) + 9(0.2) \\
&= 1.2 + 0 + 1.8 = 3.0 \checkmark
\end{align}$$

---

## ⚙️ Properties of Variance

### Property 1: Variance of a Constant

**For any constant $c$:**
$$\text{Var}(c) = 0$$

**Reasoning:** No variability in a constant!

### Property 2: Scaling by Constant

**For any constant $a$:**
$$\text{Var}(aX) = a^2 \cdot \text{Var}(X)$$

**Note the square!** Doubling a variable quadruples the variance.

**Proof:**
$$\begin{align}
\text{Var}(aX) &= E[(aX)^2] - (E[aX])^2 \\
&= E[a^2 X^2] - (a E[X])^2 \\
&= a^2 E[X^2] - a^2 (E[X])^2 \\
&= a^2 (E[X^2] - (E[X])^2) \\
&= a^2 \cdot \text{Var}(X)
\end{align}$$

### Property 3: Adding a Constant

**For any constant $b$:**
$$\text{Var}(X + b) = \text{Var}(X)$$

**Reasoning:** Shifting the distribution doesn't change its spread.

**Proof:**
$$\begin{align}
\text{Var}(X + b) &= E[(X + b)^2] - (E[X + b])^2 \\
&= E[X^2 + 2bX + b^2] - (E[X] + b)^2 \\
&= E[X^2] + 2b E[X] + b^2 - (E[X])^2 - 2b E[X] - b^2 \\
&= E[X^2] - (E[X])^2 \\
&= \text{Var}(X)
\end{align}$$

### Property 4: General Linear Transformation

**For constants $a$ and $b$:**
$$\text{Var}(aX + b) = a^2 \cdot \text{Var}(X)$$

**Example:** Temperature conversion $F = \frac{9}{5}C + 32$

If $\text{Var}(C) = 25$ (°C)², then:
$$\text{Var}(F) = \left(\frac{9}{5}\right)^2 \cdot 25 = \frac{81}{25} \cdot 25 = 81 \text{ (°F)}^2$$

**Standard deviation scales linearly:**
$$\sigma_F = \frac{9}{5} \sigma_C = \frac{9}{5}(5) = 9°F$$

### Property 5: Variance of Sum (Independent Variables)

**For independent random variables $X$ and $Y$:**
$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$$

**Warning:** This requires independence! (Not covered in detail this week)

### Example 12: Applying Variance Properties

**Problem:** A stock price model: $P = 50 + 10X$ where $X$ ~ random returns with $E[X] = 2$, $\text{Var}(X) = 4$.

**Find:** $E[P]$ and $\text{Var}(P)$

**Solution:**

**Expectation:**
$$E[P] = E[50 + 10X] = 50 + 10 E[X] = 50 + 10(2) = 70$$

**Variance:**
$$\text{Var}(P) = \text{Var}(50 + 10X) = 10^2 \cdot \text{Var}(X) = 100 \cdot 4 = 400$$

**Standard deviation:**
$$\sigma_P = \sqrt{400} = 20$$

**Interpretation:** Expected price $\$70$, typically varies by $\pm\$20$.

---

## 💼 Data Science Applications

### Application 1: A/B Testing Decision

**Problem:** Two website designs. Design A has average conversion rate 0.10, Design B has 0.12. Both tested on 1000 users.

**Random Variables:**
- $X_A$ = conversions for A, $E[X_A] = 100$, $\text{Var}(X_A) = 90$ (Binomial properties)
- $X_B$ = conversions for B, $E[X_B] = 120$, $\text{Var}(X_B) = 105.6$

**Expected difference:** $E[X_B - X_A] = 120 - 100 = 20$ more conversions

**Variance of difference (assuming independence):**
$$\text{Var}(X_B - X_A) = \text{Var}(X_B) + \text{Var}(X_A) = 105.6 + 90 = 195.6$$
$$\sigma_{diff} = \sqrt{195.6} \approx 14$$

**Interpretation:** Expected gain of 20 conversions, but with uncertainty (SD = 14). Difference is statistically significant.

### Application 2: Revenue Prediction

**Problem:** E-commerce site. Daily visitors $N$ ~ Poisson with $E[N] = 1000$, $\text{Var}(N) = 1000$. Average purchase $\$50$ per visitor.

**Random Variable:** Daily revenue $R = 50N$

**Expected revenue:**
$$E[R] = 50 \cdot E[N] = 50(1000) = \$50,000$$

**Variance of revenue:**
$$\text{Var}(R) = 50^2 \cdot \text{Var}(N) = 2500 \cdot 1000 = 2,500,000$$

**Standard deviation:**
$$\sigma_R = \sqrt{2,500,000} \approx \$1,581$$

**Interpretation:** Daily revenue around $\$50K$ with typical variation of $\pm\$1,581$.

### Application 3: Portfolio Risk Management

**Problem:** Invest in stock with random return $R$ where $E[R] = 0.08$ (8%), $\text{Var}(R) = 0.04$ (variance = 0.04).

**Investment amount:** $\$100,000$

**Random Variable:** Profit $P = 100000 \cdot R$

**Expected profit:**
$$E[P] = 100000 \cdot E[R] = 100000(0.08) = \$8,000$$

**Variance of profit:**
$$\text{Var}(P) = 100000^2 \cdot \text{Var}(R) = 10^{10} \cdot 0.04 = 4 \times 10^8$$

**Standard deviation:**
$$\sigma_P = \sqrt{4 \times 10^8} = 20,000$$

**Interpretation:** Expected profit $\$8K$, but with high risk (SD = $\$20K$).

**Risk-adjusted metric (Sharpe ratio):**
$$\text{Sharpe} = \frac{E[P]}{\sigma_P} = \frac{8000}{20000} = 0.4$$

### Application 4: Model Performance Evaluation

**Problem:** Binary classifier has accuracy = random variable $A$.

**Scenario 1:** Consistent model
- $E[A] = 0.85$
- $\text{Var}(A) = 0.01$ (low variance → reliable)

**Scenario 2:** Inconsistent model
- $E[A] = 0.85$
- $\text{Var}(A) = 0.09$ (high variance → unreliable)

**Decision:** Prefer Scenario 1 - same expected accuracy but more stable (lower risk).

**Standard deviations:**
- Scenario 1: $\sigma = 0.1$ (10 percentage points)
- Scenario 2: $\sigma = 0.3$ (30 percentage points)

---

## ⚠️ Common Pitfalls

### ❌ Pitfall 1: $E[X^2] \neq (E[X])^2$

**Wrong:**
"If $E[X] = 3$, then $E[X^2] = 9$"

**Right:**
$$E[X^2] \geq (E[X])^2 \quad \text{(Jensen's inequality for convex functions)}$$

Equality only if $X$ is constant.

**Example:** Die roll: $E[X] = 3.5$, $(E[X])^2 = 12.25$, but $E[X^2] = 15.17 \neq 12.25$

### ❌ Pitfall 2: Variance of Sum

**Wrong:**
"$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ always"

**Right:**
This holds only for **independent** random variables. For general case:
$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$

(Covariance covered in later courses)

### ❌ Pitfall 3: Units of Variance

**Wrong:**
"Variance and standard deviation have the same units"

**Right:**
- If $X$ is in meters, $\text{Var}(X)$ is in meters² (squared units)
- $\sigma_X$ is in meters (same units as $X$)

This is why standard deviation is often more interpretable.

### ❌ Pitfall 4: Scaling Variance

**Wrong:**
"$\text{Var}(2X) = 2 \cdot \text{Var}(X)$"

**Right:**
$$\text{Var}(2X) = 2^2 \cdot \text{Var}(X) = 4 \cdot \text{Var}(X)$$

Variance scales with the **square** of the constant.

### ❌ Pitfall 5: Interpreting Low Variance

**Wrong:**
"Low variance always means good"

**Right:**
- Low variance around a good mean → excellent (consistent accuracy)
- Low variance around a bad mean → consistently bad

Always consider both mean and variance together.

---

## 💻 Python Implementation

### Implementation 1: Basic Expectation and Variance

```python
import numpy as np
from scipy import stats

# Define discrete random variable (die roll)
values = np.array([1, 2, 3, 4, 5, 6])
probabilities = np.array([1/6, 1/6, 1/6, 1/6, 1/6, 1/6])

# Calculate expectation
expectation = np.sum(values * probabilities)
print(f"E[X] = {expectation:.2f}")  # 3.50

# Calculate E[X^2]
expectation_x2 = np.sum(values**2 * probabilities)
print(f"E[X²] = {expectation_x2:.2f}")  # 15.17

# Calculate variance (computational formula)
variance = expectation_x2 - expectation**2
print(f"Var(X) = {variance:.2f}")  # 2.92

# Standard deviation
std_dev = np.sqrt(variance)
print(f"σ_X = {std_dev:.2f}")  # 1.71
```

### Implementation 2: Using scipy.stats

```python
# Create discrete RV object
die_rv = stats.rv_discrete(values=(values, probabilities))

# Built-in methods
print(f"E[X] = {die_rv.mean():.2f}")  # 3.50
print(f"Var(X) = {die_rv.var():.2f}")  # 2.92
print(f"σ_X = {die_rv.std():.2f}")  # 1.71

# Moments
print(f"E[X²] = {die_rv.moment(2):.2f}")  # 15.17 (2nd moment)
print(f"E[X³] = {die_rv.moment(3):.2f}")  # 3rd moment
```

### Implementation 3: Custom Distribution

```python
# Tech support calls example
calls_values = np.array([0, 1, 2, 3, 4])
calls_probs = np.array([0.05, 0.20, 0.35, 0.25, 0.15])

# Manual calculation
mean_calls = np.sum(calls_values * calls_probs)
var_calls = np.sum(calls_values**2 * calls_probs) - mean_calls**2

print(f"Average calls: {mean_calls:.2f}")  # 2.25
print(f"Variance: {var_calls:.2f}")  # 1.29
print(f"Std dev: {np.sqrt(var_calls):.2f}")  # 1.13

# Using scipy
calls_rv = stats.rv_discrete(values=(calls_values, calls_probs))
print(f"\nVerification:")
print(f"Mean: {calls_rv.mean():.2f}")
print(f"Variance: {calls_rv.var():.2f}")
```

### Implementation 4: Transformation Properties

```python
# Original random variable
X_values = np.array([1, 2, 3])
X_probs = np.array([0.5, 0.3, 0.2])
X_rv = stats.rv_discrete(values=(X_values, X_probs))

print(f"E[X] = {X_rv.mean():.2f}")  # 1.70
print(f"Var(X) = {X_rv.var():.2f}")  # 0.61

# Transform: Y = 2X + 3
a, b = 2, 3

# E[Y] = a*E[X] + b
E_Y = a * X_rv.mean() + b
print(f"\nE[Y] = E[{a}X + {b}] = {E_Y:.2f}")  # 6.40

# Var(Y) = a²*Var(X)
Var_Y = a**2 * X_rv.var()
print(f"Var(Y) = Var({a}X + {b}) = {Var_Y:.2f}")  # 2.44

# Standard deviation scales linearly with |a|
Std_Y = abs(a) * X_rv.std()
print(f"σ_Y = {Std_Y:.2f}")  # 1.56

# Verify by creating transformed RV
Y_values = a * X_values + b
Y_rv = stats.rv_discrete(values=(Y_values, X_probs))
print(f"\nVerification:")
print(f"E[Y] = {Y_rv.mean():.2f}")
print(f"Var(Y) = {Y_rv.var():.2f}")
```

### Implementation 5: Visualization

```python
import matplotlib.pyplot as plt

# Create two distributions with same mean, different variance
# Distribution A: Low variance
A_values = np.array([8, 10, 12])
A_probs = np.array([0.25, 0.50, 0.25])
A_rv = stats.rv_discrete(values=(A_values, A_probs))

# Distribution B: High variance
B_values = np.array([0, 10, 20])
B_probs = np.array([0.25, 0.50, 0.25])
B_rv = stats.rv_discrete(values=(B_values, B_probs))

# Calculate statistics
print(f"Distribution A: E[X] = {A_rv.mean():.1f}, Var(X) = {A_rv.var():.1f}, σ = {A_rv.std():.1f}")
print(f"Distribution B: E[X] = {B_rv.mean():.1f}, Var(X) = {B_rv.var():.1f}, σ = {B_rv.std():.1f}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Distribution A
ax1.stem(A_values, A_probs, basefmt=' ')
ax1.axvline(A_rv.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean = {A_rv.mean():.1f}')
ax1.axvspan(A_rv.mean() - A_rv.std(), A_rv.mean() + A_rv.std(),
            alpha=0.2, color='red', label=f'±1σ ({A_rv.std():.1f})')
ax1.set_xlabel('Value')
ax1.set_ylabel('Probability')
ax1.set_title(f'Distribution A (Low Variance = {A_rv.var():.1f})')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Distribution B
ax2.stem(B_values, B_probs, basefmt=' ')
ax2.axvline(B_rv.mean(), color='r', linestyle='--', linewidth=2, label=f'Mean = {B_rv.mean():.1f}')
ax2.axvspan(B_rv.mean() - B_rv.std(), B_rv.mean() + B_rv.std(),
            alpha=0.2, color='red', label=f'±1σ ({B_rv.std():.1f})')
ax2.set_xlabel('Value')
ax2.set_ylabel('Probability')
ax2.set_title(f'Distribution B (High Variance = {B_rv.var():.1f})')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

## 📝 Practice Problems

### Basic Level

**Problem 1:** A random variable $X$ has PMF:

| $x$ | 0 | 1 | 2 |
|-----|---|---|---|
| $p_X(x)$ | 0.3 | 0.5 | 0.2 |

Calculate: (a) $E[X]$, (b) $E[X^2]$, (c) $\text{Var}(X)$, (d) $\sigma_X$

<details>
<summary>Solution</summary>

(a) $E[X] = 0(0.3) + 1(0.5) + 2(0.2) = 0 + 0.5 + 0.4 = 0.9$

(b) $E[X^2] = 0^2(0.3) + 1^2(0.5) + 2^2(0.2) = 0 + 0.5 + 0.8 = 1.3$

(c) $\text{Var}(X) = E[X^2] - (E[X])^2 = 1.3 - 0.9^2 = 1.3 - 0.81 = 0.49$

(d) $\sigma_X = \sqrt{0.49} = 0.7$
</details>

**Problem 2:** If $E[X] = 5$ and $\text{Var}(X) = 4$, find:

(a) $E[3X]$
(b) $E[X + 7]$
(c) $\text{Var}(3X)$
(d) $\text{Var}(X + 7)$

<details>
<summary>Solution</summary>

(a) $E[3X] = 3 E[X] = 3(5) = 15$

(b) $E[X + 7] = E[X] + 7 = 5 + 7 = 12$

(c) $\text{Var}(3X) = 3^2 \cdot \text{Var}(X) = 9(4) = 36$

(d) $\text{Var}(X + 7) = \text{Var}(X) = 4$ (adding constant doesn't change variance)
</details>

**Problem 3:** A biased coin shows heads with probability 0.7. Let $X = 1$ if heads, $X = 0$ if tails.

(a) Find $E[X]$
(b) Find $\text{Var}(X)$

<details>
<summary>Solution</summary>

(a) $E[X] = 0(0.3) + 1(0.7) = 0.7$

(b) Method 1:
$E[X^2] = 0^2(0.3) + 1^2(0.7) = 0.7$
$\text{Var}(X) = 0.7 - 0.7^2 = 0.7 - 0.49 = 0.21$

Method 2 (Bernoulli formula):
$\text{Var}(X) = p(1-p) = 0.7(0.3) = 0.21$ ✓
</details>

### Intermediate Level

**Problem 4:** Two independent dice are rolled. Let $X$ = sum of the two dice.

(a) Find $E[X]$ using linearity.
(b) If $\text{Var}(\text{one die}) = 35/12$, find $\text{Var}(X)$.

<details>
<summary>Solution</summary>

(a) Let $D_1$ and $D_2$ be the two dice. $X = D_1 + D_2$

$E[X] = E[D_1 + D_2] = E[D_1] + E[D_2] = 3.5 + 3.5 = 7$

(b) For independent variables:
$\text{Var}(X) = \text{Var}(D_1 + D_2) = \text{Var}(D_1) + \text{Var}(D_2)$

$\text{Var}(X) = \frac{35}{12} + \frac{35}{12} = \frac{70}{12} = \frac{35}{6} \approx 5.83$
</details>

**Problem 5:** A random variable has $E[X] = 10$, $E[X^2] = 120$.

(a) Find $\text{Var}(X)$
(b) Let $Y = 2X - 5$. Find $E[Y]$ and $\text{Var}(Y)$.

<details>
<summary>Solution</summary>

(a) $\text{Var}(X) = E[X^2] - (E[X])^2 = 120 - 10^2 = 120 - 100 = 20$

(b) $E[Y] = E[2X - 5] = 2E[X] - 5 = 2(10) - 5 = 15$

$\text{Var}(Y) = \text{Var}(2X - 5) = 2^2 \cdot \text{Var}(X) = 4(20) = 80$
</details>

### Advanced Level

**Problem 6:** An investment has random return $R$ with $E[R] = 0.10$ (10% return) and $\sigma_R = 0.20$ (20% std dev).

You invest $\$50,000$.

(a) What is the expected profit?
(b) What is the standard deviation of profit?
(c) What is the probability that profit is within ±1 standard deviation of the mean (approximate using 68% rule)?

<details>
<summary>Solution</summary>

Profit $P = 50000 \cdot R$

(a) $E[P] = 50000 \cdot E[R] = 50000(0.10) = \$5,000$

(b) $\text{Var}(P) = 50000^2 \cdot \text{Var}(R) = 50000^2 \cdot 0.20^2$

$\text{Var}(R) = \sigma_R^2 = 0.04$

$\text{Var}(P) = 2,500,000,000 \cdot 0.04 = 100,000,000$

$\sigma_P = \sqrt{100,000,000} = \$10,000$

(c) Approximately 68% chance profit is in $[\$5,000 - \$10,000, \$5,000 + \$10,000]$

= $[-\$5,000, \$15,000]$

(Empirical rule for normal-like distributions)
</details>

**Problem 7:** A data scientist models website traffic as $X$ visitors per hour with $E[X] = 500$, $\text{Var}(X) = 2500$.

Revenue per visitor is $\$0.50$.

(a) What is expected hourly revenue?
(b) What is the variance of hourly revenue?
(c) If we want to reduce revenue uncertainty (variance) by 75%, what would the new variance of $X$ need to be?

<details>
<summary>Solution</summary>

Revenue $R = 0.50 \cdot X$

(a) $E[R] = 0.50 \cdot E[X] = 0.50(500) = \$250$

(b) $\text{Var}(R) = 0.50^2 \cdot \text{Var}(X) = 0.25 \cdot 2500 = 625$

$\sigma_R = \sqrt{625} = \$25$

(c) Want $\text{Var}(R_{\text{new}}) = 0.25 \cdot 625 = 156.25$

$0.50^2 \cdot \text{Var}(X_{\text{new}}) = 156.25$

$\text{Var}(X_{\text{new}}) = \frac{156.25}{0.25} = 625$

Need to reduce traffic variance from 2500 to 625 (by 75%).
</details>

---

## 🎓 Self-Assessment Questions

**Conceptual:**
- [ ] Can you explain what expectation represents intuitively?
- [ ] Why is variance defined as the expected squared deviation (not just deviation)?
- [ ] How does scaling a random variable affect variance differently than mean?
- [ ] When would two distributions have the same mean but different variances?

**Computational:**
- [ ] Can you calculate $E[X]$ and $\text{Var}(X)$ from a PMF?
- [ ] Can you apply linearity properties to find $E[aX + b]$?
- [ ] Can you compute $\text{Var}(aX + b)$ correctly?
- [ ] Can you use both formulas for variance (definition and computational)?

**Application:**
- [ ] Can you interpret expectation and variance in a business context?
- [ ] Can you use these concepts for risk assessment?
- [ ] Can you compare different scenarios using mean and variance?

---

## 📚 Quick Reference Summary

### Key Formulas

| Concept | Formula |
|---------|---------|
| Expectation | $E[X] = \sum x \cdot p_X(x)$ |
| Variance (definition) | $\text{Var}(X) = E[(X - \mu)^2]$ |
| Variance (computational) | $\text{Var}(X) = E[X^2] - (E[X])^2$ |
| Standard deviation | $\sigma_X = \sqrt{\text{Var}(X)}$ |
| Expectation of function | $E[g(X)] = \sum g(x) \cdot p_X(x)$ |

### Properties

| Property | Expectation | Variance |
|----------|-------------|----------|
| Constant | $E[c] = c$ | $\text{Var}(c) = 0$ |
| Scaling | $E[aX] = a E[X]$ | $\text{Var}(aX) = a^2 \text{Var}(X)$ |
| Shift | $E[X + b] = E[X] + b$ | $\text{Var}(X + b) = \text{Var}(X)$ |
| Linear | $E[aX + b] = aE[X] + b$ | $\text{Var}(aX + b) = a^2\text{Var}(X)$ |
| Sum | $E[X + Y] = E[X] + E[Y]$ | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ (if independent) |

---

## 🔜 Next Week Preview

**Week 11: Binomial and Poisson Distributions**

We'll study two important discrete probability distributions:
- **Binomial**: Models number of successes in $n$ independent trials
- **Poisson**: Models rare events occurring at a constant rate

Both build directly on expectation and variance concepts!

---

**End of Week 10 Notes**
