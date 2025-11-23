# Week 9: Random Variables and Probability Functions

---
**Date**: 2025-11-22
**Course**: BSMA1002 - Statistics for Data Science I
**Level**: Foundation
**Week**: 9 of 12
**Source**: IIT Madras Statistics I Week 9
**Topic Area**: Probability Theory, Random Variables
**Tags**: #BSMA1002 #RandomVariables #PMF #CDF #Week9 #Foundation
---

## 📋 Bottom Line Up Front (BLUF)

**What**: Random variables transform outcomes of random experiments into numerical values, enabling quantitative probability analysis through probability mass functions (PMF) and cumulative distribution functions (CDF).

**Why**: Random variables are the mathematical foundation for modeling uncertainty in data science - from predicting customer behavior to assessing model confidence.

**Key Takeaway**: Understanding PMF (probability at specific values) and CDF (cumulative probability up to a value) is essential for working with probability distributions in statistical inference and machine learning.

---

## 🎯 Learning Objectives

By the end of this week, you should be able to:

1. ✅ Define random variables and distinguish between discrete and continuous types
2. ✅ Construct and interpret probability mass functions (PMF) for discrete random variables
3. ✅ Compute and apply cumulative distribution functions (CDF)
4. ✅ Verify properties of valid probability distributions
5. ✅ Implement PMF and CDF calculations in Python using scipy.stats
6. ✅ Apply random variable concepts to model data science problems

---

## 📚 Table of Contents

1. [Random Experiments and Sample Spaces](#random-experiments-and-sample-spaces)
2. [Random Variables: From Outcomes to Numbers](#random-variables-from-outcomes-to-numbers)
3. [Discrete vs Continuous Random Variables](#discrete-vs-continuous-random-variables)
4. [Probability Mass Function (PMF)](#probability-mass-function-pmf)
5. [Cumulative Distribution Function (CDF)](#cumulative-distribution-function-cdf)
6. [Relationship Between PMF and CDF](#relationship-between-pmf-and-cdf)
7. [Data Science Applications](#data-science-applications)
8. [Common Pitfalls](#common-pitfalls)
9. [Python Implementation](#python-implementation)
10. [Practice Problems](#practice-problems)

---

## 🔬 Random Experiments and Sample Spaces

### Definition: Random Experiment

**Definition:** A **random experiment** is a process or action whose outcome cannot be predicted with certainty before it is performed, but whose set of all possible outcomes is known.

**Purpose:** Random experiments form the foundation for probability theory and statistical modeling.

**Properties:**
- Can be repeated under identical conditions
- The outcome is uncertain before the experiment
- All possible outcomes can be enumerated or described

### Sample Space

**Definition:** The **sample space** $S$ (or $\Omega$) is the set of all possible outcomes of a random experiment.

**Examples:**

| Experiment | Sample Space | Type |
|-----------|--------------|------|
| Coin flip | $S = \{\text{Heads}, \text{Tails}\}$ | Finite |
| Die roll | $S = \{1, 2, 3, 4, 5, 6\}$ | Finite |
| Number of customers in hour | $S = \{0, 1, 2, 3, ...\}$ | Countably infinite |
| Time until server failure | $S = [0, \infty)$ | Uncountably infinite |

### Events

**Definition:** An **event** is a subset of the sample space - a collection of outcomes.

**Example:** For a die roll:
- Event A = "roll an even number" = $\{2, 4, 6\}$
- Event B = "roll greater than 4" = $\{5, 6\}$
- Event C = "roll 7" = $\emptyset$ (impossible event)

**The limitation:** Sample spaces can contain non-numerical outcomes (like "Heads" or "Red"). We need a way to work with numbers...

---

## 🔢 Random Variables: From Outcomes to Numbers

### The Power of Numerical Mapping

**Insight:** While sample spaces can have non-numerical outcomes, mathematical analysis requires numbers. Random variables provide this crucial transformation.

**Definition:** A **random variable** $X$ is a function that assigns a real number to each outcome in the sample space:

$$X: S \to \mathbb{R}$$

**Purpose:** Random variables bridge the gap between qualitative outcomes and quantitative analysis.

### Example 1: Coin Flips to Numbers

**Experiment:** Flip a coin twice.

**Sample Space:** $S = \{HH, HT, TH, TT\}$

**Random Variable:** Let $X$ = "number of heads"

**Mapping:**
- $X(HH) = 2$
- $X(HT) = 1$
- $X(TH) = 1$
- $X(TT) = 0$

**Range of $X$:** $\{0, 1, 2\}$ (possible values)

**Now we can ask:** What's $P(X = 1)$? This equals $P(\{HT, TH\}) = \frac{2}{4} = 0.5$

### Example 2: Customer Arrivals

**Experiment:** Monitor a store for 1 hour.

**Sample Space:** Complex (each customer arrival pattern is different)

**Random Variable:** Let $X$ = "number of customers arriving in 1 hour"

**Why useful:** Instead of dealing with complex arrival patterns, we work with a simple number $X \in \{0, 1, 2, ...\}$

### Example 3: Website Response Time

**Experiment:** Measure time for webpage to load.

**Sample Space:** $S = [0, \infty)$ (any positive real number)

**Random Variable:** Let $X$ = "response time in seconds"

**Why useful:** $X$ directly quantifies performance; we can compute $P(X \leq 2)$ = "probability of loading within 2 seconds"

### Multiple Random Variables from Same Experiment

From a single experiment, we can define multiple random variables:

**Experiment:** Roll two dice.

**Random Variables:**
- $X$ = sum of the two dice
- $Y$ = maximum of the two dice
- $Z$ = absolute difference between the two dice

**For outcome $(3, 5)$:**
- $X(3,5) = 8$
- $Y(3,5) = 5$
- $Z(3,5) = |3-5| = 2$

---

## 🔀 Discrete vs Continuous Random Variables

### Discrete Random Variables

**Definition:** A random variable $X$ is **discrete** if it can take on a countable number of values (finite or countably infinite).

**Characteristics:**
- Values can be listed: $x_1, x_2, x_3, ...$
- Often result from counting processes
- Probabilities sum to 1: $\sum P(X = x_i) = 1$

**Examples:**

| Random Variable | Possible Values | Context |
|----------------|-----------------|---------|
| Number of heads in 10 coin flips | $\{0, 1, 2, ..., 10\}$ | Counting |
| Number of defective items in batch | $\{0, 1, 2, ..., n\}$ | Quality control |
| Number of website visitors | $\{0, 1, 2, ...\}$ | Web analytics |
| Number of emails received per day | $\{0, 1, 2, ...\}$ | Communication |

### Continuous Random Variables

**Definition:** A random variable $X$ is **continuous** if it can take any value in an interval or collection of intervals.

**Characteristics:**
- Values form a continuum (uncountably infinite)
- Often result from measurement processes
- $P(X = x) = 0$ for any specific value $x$
- Probabilities are defined over intervals

**Examples:**

| Random Variable | Possible Values | Context |
|----------------|-----------------|---------|
| Height of a person | $(0, \infty)$ or $[100, 250]$ cm | Measurement |
| Temperature | $(-\infty, \infty)$ or $[-50, 50]$ °C | Environment |
| Stock price | $(0, \infty)$ | Finance |
| Time until server crash | $[0, \infty)$ | Reliability |

### Key Distinction: Probability at a Point

**Discrete:**
$$P(X = 5) > 0 \text{ (can be non-zero)}$$

**Continuous:**
$$P(X = 5.0) = 0 \text{ (always zero for specific values)}$$

For continuous variables, we ask about intervals:
$$P(4.9 \leq X \leq 5.1) > 0$$

### Example 4: Discrete Random Variable - Die Roll

**Experiment:** Roll a fair six-sided die.

**Random Variable:** $X$ = outcome of the roll

**Possible values:** $\{1, 2, 3, 4, 5, 6\}$ (discrete, finite)

**Probabilities:**
$$P(X = k) = \frac{1}{6} \quad \text{for } k \in \{1, 2, 3, 4, 5, 6\}$$

**Verification:**
$$\sum_{k=1}^{6} P(X = k) = 6 \times \frac{1}{6} = 1 \checkmark$$

### Example 5: Continuous Random Variable - Bus Arrival

**Experiment:** Measure time waiting for bus (arrives every 15 minutes).

**Random Variable:** $X$ = waiting time in minutes

**Possible values:** $[0, 15]$ (continuous interval)

**Uniform distribution:**
$$P(a \leq X \leq b) = \frac{b - a}{15} \quad \text{for } 0 \leq a \leq b \leq 15$$

**Probability of waiting 5-10 minutes:**
$$P(5 \leq X \leq 10) = \frac{10 - 5}{15} = \frac{5}{15} = \frac{1}{3}$$

**Probability of waiting exactly 7 minutes:**
$$P(X = 7) = 0 \quad \text{(continuous random variable)}$$

---

## 📊 Probability Mass Function (PMF)

### Definition and Properties

**Definition:** For a discrete random variable $X$, the **probability mass function (PMF)** is:

$$p_X(x) = P(X = x)$$

This gives the probability that $X$ takes the specific value $x$.

**Required Properties:**

1. **Non-negativity:** $p_X(x) \geq 0$ for all $x$
2. **Normalization:** $\sum_{\text{all } x} p_X(x) = 1$

**Purpose:** The PMF completely characterizes the probability distribution of a discrete random variable.

### Example 6: PMF of Die Roll

**Random Variable:** $X$ = outcome of fair die roll

**PMF:**
$$p_X(x) = \begin{cases}
\frac{1}{6} & \text{if } x \in \{1, 2, 3, 4, 5, 6\} \\
0 & \text{otherwise}
\end{cases}$$

**Verification:**
- Non-negativity: $\frac{1}{6} > 0$ ✓
- Normalization: $\sum_{x=1}^{6} \frac{1}{6} = 1$ ✓

**Visual representation:**
```
Probability
   1/6  ├─┐  ┌─┐  ┌─┐  ┌─┐  ┌─┐  ┌─┐
        │ │  │ │  │ │  │ │  │ │  │ │
     0  └─┴──┴─┴──┴─┴──┴─┴──┴─┴──┴─┴──
         1   2   3   4   5   6    x
```

### Example 7: PMF of Sum of Two Dice

**Random Variable:** $X$ = sum of two fair dice

**Possible values:** $\{2, 3, 4, ..., 12\}$

**Calculating PMF:**
- $X = 2$: Only $(1,1)$ → $p_X(2) = \frac{1}{36}$
- $X = 3$: $(1,2), (2,1)$ → $p_X(3) = \frac{2}{36}$
- $X = 4$: $(1,3), (2,2), (3,1)$ → $p_X(4) = \frac{3}{36}$
- $X = 7$: $(1,6), (2,5), (3,4), (4,3), (5,2), (6,1)$ → $p_X(7) = \frac{6}{36}$

**Full PMF:**
$$p_X(x) = \begin{cases}
\frac{|x-7|}{36} & \text{if } x \in \{2, 3, ..., 12\}, x \neq 7 \\
\frac{6}{36} & \text{if } x = 7 \\
0 & \text{otherwise}
\end{cases}$$

(Note: For $x < 7$: $p_X(x) = \frac{x-1}{36}$; for $x > 7$: $p_X(x) = \frac{13-x}{36}$)

**Verification:**
$$\sum_{x=2}^{12} p_X(x) = \frac{1+2+3+4+5+6+5+4+3+2+1}{36} = \frac{36}{36} = 1 \checkmark$$

### Example 8: Custom PMF - Tech Support Calls

**Scenario:** A tech support center receives calls. Let $X$ = number of calls in a 10-minute window.

**Historical data gives PMF:**

| $x$ | 0 | 1 | 2 | 3 | 4+ |
|-----|---|---|---|---|----|
| $p_X(x)$ | 0.05 | 0.20 | 0.35 | 0.25 | 0.15 |

**Verification:**
$$\sum p_X(x) = 0.05 + 0.20 + 0.35 + 0.25 + 0.15 = 1.00 \checkmark$$

**Useful probability calculations:**

1. **Probability of at most 2 calls:**
$$P(X \leq 2) = p_X(0) + p_X(1) + p_X(2) = 0.05 + 0.20 + 0.35 = 0.60$$

2. **Probability of at least 2 calls:**
$$P(X \geq 2) = p_X(2) + p_X(3) + p_X(4+) = 0.35 + 0.25 + 0.15 = 0.75$$

3. **Probability of more than 1 but less than 4 calls:**
$$P(1 < X < 4) = p_X(2) + p_X(3) = 0.35 + 0.25 = 0.60$$

### Constructing a Valid PMF

**Example 9:** Is this a valid PMF?

$$p_X(x) = \begin{cases}
cx & \text{if } x \in \{1, 2, 3, 4\} \\
0 & \text{otherwise}
\end{cases}$$

**Solution:**

**Step 1:** Apply normalization condition.
$$\sum_{x=1}^{4} cx = 1$$
$$c(1 + 2 + 3 + 4) = 1$$
$$10c = 1$$
$$c = 0.1$$

**Step 2:** Verify non-negativity.
$$p_X(x) = 0.1x > 0 \text{ for } x \in \{1, 2, 3, 4\} \checkmark$$

**Step 3:** Write complete PMF.
$$p_X(x) = \begin{cases}
0.1 & \text{if } x = 1 \\
0.2 & \text{if } x = 2 \\
0.3 & \text{if } x = 3 \\
0.4 & \text{if } x = 4 \\
0 & \text{otherwise}
\end{cases}$$

**Interpretation:** Higher values are more likely (weighted toward 4).

---

## 📈 Cumulative Distribution Function (CDF)

### Definition and Properties

**Definition:** For any random variable $X$ (discrete or continuous), the **cumulative distribution function (CDF)** is:

$$F_X(x) = P(X \leq x)$$

This gives the probability that $X$ takes a value less than or equal to $x$.

**Required Properties:**

1. **Range:** $0 \leq F_X(x) \leq 1$ for all $x$
2. **Non-decreasing:** If $x_1 \leq x_2$, then $F_X(x_1) \leq F_X(x_2)$
3. **Limits:**
   - $\lim_{x \to -\infty} F_X(x) = 0$
   - $\lim_{x \to \infty} F_X(x) = 1$
4. **Right-continuous:** $\lim_{h \to 0^+} F_X(x + h) = F_X(x)$

**Purpose:** CDF provides cumulative probability information and works for both discrete and continuous random variables.

### For Discrete Random Variables

**Relationship to PMF:**
$$F_X(x) = \sum_{t \leq x} p_X(t)$$

The CDF is the cumulative sum of the PMF.

### Example 10: CDF of Die Roll

**PMF:** $p_X(x) = \frac{1}{6}$ for $x \in \{1, 2, 3, 4, 5, 6\}$

**CDF Calculation:**

$$F_X(x) = \begin{cases}
0 & \text{if } x < 1 \\
\frac{1}{6} & \text{if } 1 \leq x < 2 \\
\frac{2}{6} & \text{if } 2 \leq x < 3 \\
\frac{3}{6} & \text{if } 3 \leq x < 4 \\
\frac{4}{6} & \text{if } 4 \leq x < 5 \\
\frac{5}{6} & \text{if } 5 \leq x < 6 \\
1 & \text{if } x \geq 6
\end{cases}$$

**Visual representation:**
```
F_X(x)
   1.0 ├────────────────┐
       │                │
  5/6  ├──────────────┐ │
       │              │ │
  4/6  ├────────────┐ │ │
       │            │ │ │
  3/6  ├──────────┐ │ │ │
       │          │ │ │ │
  2/6  ├────────┐ │ │ │ │
       │        │ │ │ │ │
  1/6  ├──────┐ │ │ │ │ │
       │      │ │ │ │ │ │
   0   └──────┴─┴─┴─┴─┴─┴──→ x
       0  1  2  3  4  5  6
```

**Note:** The CDF is a step function for discrete random variables.

### Example 11: Using CDF to Find Probabilities

**Given:** $X$ = die roll with CDF from Example 10.

**Calculate:**

1. **$P(X \leq 4)$:**
$$P(X \leq 4) = F_X(4) = \frac{4}{6} = \frac{2}{3}$$

2. **$P(X > 4)$:**
$$P(X > 4) = 1 - P(X \leq 4) = 1 - F_X(4) = 1 - \frac{4}{6} = \frac{2}{6} = \frac{1}{3}$$

3. **$P(X = 3)$:**
$$P(X = 3) = F_X(3) - F_X(2) = \frac{3}{6} - \frac{2}{6} = \frac{1}{6}$$

4. **$P(2 < X \leq 5)$:**
$$P(2 < X \leq 5) = F_X(5) - F_X(2) = \frac{5}{6} - \frac{2}{6} = \frac{3}{6} = \frac{1}{2}$$

### Example 12: CDF of Tech Support Calls

**From Example 8:** $X$ = calls in 10 minutes with PMF:

| $x$ | 0 | 1 | 2 | 3 | 4+ |
|-----|---|---|---|---|----|
| $p_X(x)$ | 0.05 | 0.20 | 0.35 | 0.25 | 0.15 |

**Compute CDF:**

$$F_X(x) = \begin{cases}
0 & \text{if } x < 0 \\
0.05 & \text{if } 0 \leq x < 1 \\
0.25 & \text{if } 1 \leq x < 2 \\
0.60 & \text{if } 2 \leq x < 3 \\
0.85 & \text{if } 3 \leq x < 4 \\
1.00 & \text{if } x \geq 4
\end{cases}$$

**Application:**
- $P(X \leq 2) = F_X(2) = 0.60$ (60% chance of at most 2 calls)
- $P(X > 3) = 1 - F_X(3) = 1 - 0.85 = 0.15$ (15% chance of more than 3 calls)

---

## 🔗 Relationship Between PMF and CDF

### From PMF to CDF

**For discrete random variables:**
$$F_X(x) = \sum_{t \leq x} p_X(t)$$

**Algorithm:**
1. List all values $t$ of the random variable where $t \leq x$
2. Sum their probabilities

### From CDF to PMF

**For discrete random variables:**
$$p_X(x) = F_X(x) - F_X(x^-)$$

where $x^-$ means "just before $x$" (the limit from the left).

**Practical formula:**
$$p_X(x_i) = F_X(x_i) - F_X(x_{i-1})$$

### Example 13: Converting Between PMF and CDF

**Given CDF:**
$$F_X(x) = \begin{cases}
0 & \text{if } x < 1 \\
0.3 & \text{if } 1 \leq x < 2 \\
0.6 & \text{if } 2 \leq x < 3 \\
1.0 & \text{if } x \geq 3
\end{cases}$$

**Find PMF:**

$$\begin{align}
p_X(1) &= F_X(1) - F_X(0) = 0.3 - 0 = 0.3 \\
p_X(2) &= F_X(2) - F_X(1) = 0.6 - 0.3 = 0.3 \\
p_X(3) &= F_X(3) - F_X(2) = 1.0 - 0.6 = 0.4
\end{align}$$

**Result:**
$$p_X(x) = \begin{cases}
0.3 & \text{if } x = 1 \\
0.3 & \text{if } x = 2 \\
0.4 & \text{if } x = 3 \\
0 & \text{otherwise}
\end{cases}$$

**Verification:** $0.3 + 0.3 + 0.4 = 1.0$ ✓

---

## 💼 Data Science Applications

### Application 1: Customer Churn Prediction

**Problem:** Predict number of customers who will cancel subscription next month.

**Random Variable:** $X$ = number of cancellations out of 100 customers

**Historical PMF:**
- $p_X(0) = 0.10$ (10% chance no cancellations)
- $p_X(1) = 0.20$
- $p_X(2) = 0.30$
- $p_X(3) = 0.25$
- $p_X(4+) = 0.15$

**Business Question:** What's the probability of losing at most 2 customers?

**Solution:**
$$P(X \leq 2) = F_X(2) = 0.10 + 0.20 + 0.30 = 0.60$$

**Interpretation:** 60% confidence that churn will be 2 or fewer customers. Company can plan resources accordingly.

### Application 2: A/B Testing Conversions

**Problem:** Website redesign A/B test with 50 users shown new design.

**Random Variable:** $X$ = number of users who convert (make purchase)

**Null hypothesis:** Conversion rate unchanged at 10%

**Expected PMF (under null):** Binomial distribution (covered next week)

**Data science use:**
- Compare observed $X$ to expected PMF
- If observed value has low probability under null, reject null hypothesis
- Conclude redesign significantly affects conversions

### Application 3: Server Request Modeling

**Problem:** Model number of API requests per second to scale infrastructure.

**Random Variable:** $X$ = requests per second

**Observed PMF from logs:**

| $x$ | 0-10 | 11-20 | 21-30 | 31-40 | 41+ |
|-----|------|-------|-------|-------|-----|
| $p_X(x)$ | 0.05 | 0.15 | 0.40 | 0.30 | 0.10 |

**Infrastructure Planning:**
- $P(X \leq 30) = F_X(30) = 0.60$ → 60% of time, load is manageable
- $P(X > 40) = 0.10$ → Need burst capacity for 10% of time

**Decision:** Provision servers for 30 req/s baseline + auto-scaling for peaks.

### Application 4: Model Confidence Scores

**Problem:** Binary classifier outputs confidence score for predictions.

**Random Variable:** $X$ = confidence score (0-10 discrete levels)

**Ideal PMF for accurate model:**
- High confidence for correct predictions
- Low confidence for incorrect predictions

**Use CDF:**
$$F_X(5) = P(\text{confidence} \leq 5)$$

If $F_X(5) = 0.90$ → Model is highly confident (90% of predictions above 5)

---

## ⚠️ Common Pitfalls

### ❌ Pitfall 1: Confusing $P(X = x)$ with $P(X \leq x)$

**Wrong:**
"The probability of rolling at most 4 is $p_X(4) = \frac{1}{6}$."

**Right:**
"The probability of rolling at most 4 is $F_X(4) = P(X \leq 4) = \frac{4}{6}$."

**Remember:**
- PMF: $p_X(x) = P(X = x)$ (equals exactly $x$)
- CDF: $F_X(x) = P(X \leq x)$ (at most $x$)

### ❌ Pitfall 2: PMF Not Summing to 1

**Wrong:**
$$p_X(x) = \begin{cases}
0.3 & \text{if } x = 1 \\
0.4 & \text{if } x = 2 \\
0.5 & \text{if } x = 3
\end{cases}$$

Sum: $0.3 + 0.4 + 0.5 = 1.2 \neq 1$ ❌

**Right:** Normalize:
$$p_X(x) = \begin{cases}
0.3/1.2 = 0.25 & \text{if } x = 1 \\
0.4/1.2 = 0.33 & \text{if } x = 2 \\
0.5/1.2 = 0.42 & \text{if } x = 3
\end{cases}$$

### ❌ Pitfall 3: CDF Decreasing

**Wrong:**
$$F_X(x) = \begin{cases}
0.5 & \text{if } x < 1 \\
0.3 & \text{if } x \geq 1
\end{cases}$$

CDF decreases from 0.5 to 0.3 ❌ (violates monotonicity)

**Right:** CDF must be non-decreasing.

### ❌ Pitfall 4: Treating Discrete as Continuous

**Wrong:**
"For discrete $X$: $P(X = 3) = 0$" (treating as continuous)

**Right:**
"For discrete $X$: $P(X = 3) = p_X(3)$ which can be non-zero."

**Remember:** Only continuous random variables have $P(X = x) = 0$ for specific $x$.

### ❌ Pitfall 5: Incorrect Interval Probability

**Wrong:**
"$P(2 \leq X \leq 5) = F_X(5) - F_X(2)$"

**Right:**
"$P(2 \leq X \leq 5) = F_X(5) - F_X(1)$" (since we want to include $X = 2$)

Or: "$P(2 < X \leq 5) = F_X(5) - F_X(2)$" (exclude $X = 2$)

**Formula:**
- $P(a < X \leq b) = F_X(b) - F_X(a)$
- $P(a \leq X \leq b) = F_X(b) - F_X(a-1)$ (for discrete $X$)

---

## 💻 Python Implementation

### Implementation 1: Custom PMF

```python
import numpy as np
import matplotlib.pyplot as plt

# Define custom PMF for die roll
def die_pmf(x):
    """PMF for fair six-sided die"""
    if x in [1, 2, 3, 4, 5, 6]:
        return 1/6
    else:
        return 0

# Calculate probabilities
outcomes = range(0, 8)  # Include beyond range to show zero probability
probabilities = [die_pmf(x) for x in outcomes]

# Visualize PMF
plt.figure(figsize=(10, 5))
plt.stem(outcomes, probabilities, basefmt=' ')
plt.xlabel('Outcome (x)')
plt.ylabel('Probability P(X = x)')
plt.title('PMF of Fair Die Roll')
plt.xticks(outcomes)
plt.ylim(0, 0.25)
plt.grid(True, alpha=0.3)
plt.show()

# Verify normalization
print(f"Sum of probabilities: {sum(probabilities):.4f}")
# Output: Sum of probabilities: 1.0000
```

### Implementation 2: CDF from PMF

```python
def die_cdf(x):
    """CDF for fair six-sided die"""
    if x < 1:
        return 0
    elif x >= 6:
        return 1
    else:
        return int(x) / 6

# Calculate CDF values
x_values = np.linspace(0, 7, 1000)
cdf_values = [die_cdf(x) for x in x_values]

# Visualize CDF (step function)
plt.figure(figsize=(10, 5))
plt.plot(x_values, cdf_values, 'b-', linewidth=2)
plt.xlabel('x')
plt.ylabel('F(x) = P(X ≤ x)')
plt.title('CDF of Fair Die Roll')
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)

# Add horizontal lines to emphasize steps
for i in range(1, 7):
    plt.hlines(i/6, i, i+1, colors='blue', alpha=0.3, linestyle='--')
    plt.plot(i, i/6, 'bo', markersize=8)

plt.show()
```

### Implementation 3: Using scipy.stats

```python
from scipy import stats

# Create discrete random variable for die roll
die_values = [1, 2, 3, 4, 5, 6]
die_probabilities = [1/6, 1/6, 1/6, 1/6, 1/6, 1/6]
die_rv = stats.rv_discrete(values=(die_values, die_probabilities))

# PMF
print(f"P(X = 3) = {die_rv.pmf(3):.4f}")  # 0.1667

# CDF
print(f"P(X ≤ 4) = {die_rv.cdf(4):.4f}")  # 0.6667

# Probability of interval
print(f"P(2 ≤ X ≤ 5) = {die_rv.cdf(5) - die_rv.cdf(1):.4f}")  # 0.6667

# Generate random samples
samples = die_rv.rvs(size=1000)
print(f"Mean of 1000 samples: {np.mean(samples):.2f}")  # ≈ 3.5
print(f"Empirical PMF at x=3: {np.sum(samples == 3) / 1000:.3f}")  # ≈ 0.167
```

### Implementation 4: Sum of Two Dice

```python
# Generate PMF for sum of two dice
sum_values = range(2, 13)  # Possible sums: 2 through 12
sum_counts = []

for s in sum_values:
    count = 0
    for die1 in range(1, 7):
        for die2 in range(1, 7):
            if die1 + die2 == s:
                count += 1
    sum_counts.append(count)

sum_probabilities = [c / 36 for c in sum_counts]

# Create discrete RV
sum_rv = stats.rv_discrete(values=(sum_values, sum_probabilities))

# Visualize PMF
plt.figure(figsize=(12, 5))
plt.bar(sum_values, sum_probabilities, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Sum of Two Dice')
plt.ylabel('Probability')
plt.title('PMF of Sum of Two Fair Dice')
plt.xticks(sum_values)
plt.grid(True, axis='y', alpha=0.3)
plt.show()

# Calculate probabilities
print(f"P(X = 7) = {sum_rv.pmf(7):.4f}")  # 0.1667 (6/36)
print(f"P(X ≤ 7) = {sum_rv.cdf(7):.4f}")  # 0.5833 (21/36)
print(f"P(X > 9) = {1 - sum_rv.cdf(9):.4f}")  # 0.1667 (6/36)
```

### Implementation 5: Custom Business PMF

```python
# Tech support calls example
calls_values = [0, 1, 2, 3, 4]
calls_probabilities = [0.05, 0.20, 0.35, 0.25, 0.15]
calls_rv = stats.rv_discrete(values=(calls_values, calls_probabilities))

# Create side-by-side PMF and CDF plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# PMF
ax1.stem(calls_values, calls_probabilities, basefmt=' ')
ax1.set_xlabel('Number of Calls')
ax1.set_ylabel('Probability P(X = x)')
ax1.set_title('PMF: Tech Support Calls (10 min window)')
ax1.set_xticks(calls_values)
ax1.grid(True, alpha=0.3)

# CDF
cdf_values = [calls_rv.cdf(x) for x in calls_values]
ax2.step(calls_values, cdf_values, where='post', linewidth=2)
ax2.plot(calls_values, cdf_values, 'bo', markersize=8)
ax2.set_xlabel('Number of Calls')
ax2.set_ylabel('Cumulative Probability F(x)')
ax2.set_title('CDF: Tech Support Calls')
ax2.set_xticks(calls_values)
ax2.set_ylim(0, 1.1)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Business insights
print(f"P(at most 2 calls) = {calls_rv.cdf(2):.2f}")  # 0.60
print(f"P(more than 2 calls) = {1 - calls_rv.cdf(2):.2f}")  # 0.40
print(f"P(1 or 2 calls) = {calls_rv.pmf(1) + calls_rv.pmf(2):.2f}")  # 0.55
```

---

## 📝 Practice Problems

### Basic Level

**Problem 1:** A random variable $X$ has the following PMF:

$$p_X(x) = \begin{cases}
0.2 & \text{if } x = 1 \\
0.3 & \text{if } x = 2 \\
0.5 & \text{if } x = 3 \\
0 & \text{otherwise}
\end{cases}$$

(a) Verify this is a valid PMF.
(b) Find $P(X \geq 2)$.
(c) Compute the CDF $F_X(x)$ for all $x$.

<details>
<summary>Solution</summary>

(a) Check properties:
- Non-negativity: All probabilities ≥ 0 ✓
- Normalization: $0.2 + 0.3 + 0.5 = 1.0$ ✓

Valid PMF!

(b) $P(X \geq 2) = p_X(2) + p_X(3) = 0.3 + 0.5 = 0.8$

(c) CDF:
$$F_X(x) = \begin{cases}
0 & \text{if } x < 1 \\
0.2 & \text{if } 1 \leq x < 2 \\
0.5 & \text{if } 2 \leq x < 3 \\
1.0 & \text{if } x \geq 3
\end{cases}$$
</details>

**Problem 2:** For the random variable in Problem 1, find:

(a) $P(X < 3)$
(b) $P(1 < X \leq 3)$
(c) $P(X = 2.5)$

<details>
<summary>Solution</summary>

(a) $P(X < 3) = F_X(2) = 0.5$ (or $p_X(1) + p_X(2) = 0.2 + 0.3 = 0.5$)

(b) $P(1 < X \leq 3) = F_X(3) - F_X(1) = 1.0 - 0.2 = 0.8$

(c) $P(X = 2.5) = 0$ (2.5 is not in the support of $X$)
</details>

**Problem 3:** A random variable has CDF:

$$F_X(x) = \begin{cases}
0 & \text{if } x < 0 \\
0.4 & \text{if } 0 \leq x < 1 \\
0.7 & \text{if } 1 \leq x < 2 \\
1.0 & \text{if } x \geq 2
\end{cases}$$

Find the PMF $p_X(x)$.

<details>
<summary>Solution</summary>

$$\begin{align}
p_X(0) &= F_X(0) - F_X(0^-) = 0.4 - 0 = 0.4 \\
p_X(1) &= F_X(1) - F_X(0) = 0.7 - 0.4 = 0.3 \\
p_X(2) &= F_X(2) - F_X(1) = 1.0 - 0.7 = 0.3
\end{align}$$

PMF:
$$p_X(x) = \begin{cases}
0.4 & \text{if } x = 0 \\
0.3 & \text{if } x = 1 \\
0.3 & \text{if } x = 2 \\
0 & \text{otherwise}
\end{cases}$$

Verification: $0.4 + 0.3 + 0.3 = 1.0$ ✓
</details>

### Intermediate Level

**Problem 4:** A discrete random variable $X$ has PMF $p_X(x) = cx^2$ for $x \in \{1, 2, 3\}$ and 0 otherwise.

(a) Find the value of $c$.
(b) Calculate $P(X \geq 2)$.
(c) Find the CDF and evaluate $F_X(2.5)$.

<details>
<summary>Solution</summary>

(a) Normalization condition:
$$\sum_{x=1}^{3} cx^2 = 1$$
$$c(1^2 + 2^2 + 3^2) = c(1 + 4 + 9) = 14c = 1$$
$$c = \frac{1}{14}$$

(b) $$P(X \geq 2) = p_X(2) + p_X(3) = \frac{4}{14} + \frac{9}{14} = \frac{13}{14} \approx 0.929$$

(c) CDF:
$$F_X(x) = \begin{cases}
0 & \text{if } x < 1 \\
\frac{1}{14} & \text{if } 1 \leq x < 2 \\
\frac{5}{14} & \text{if } 2 \leq x < 3 \\
1 & \text{if } x \geq 3
\end{cases}$$

$$F_X(2.5) = \frac{5}{14} \approx 0.357$$
</details>

**Problem 5:** Two independent fair coins are tossed. Let $X$ = number of heads.

(a) Define the sample space and the random variable.
(b) Find the PMF of $X$.
(c) Calculate $P(X \leq 1)$ and verify using the CDF.

<details>
<summary>Solution</summary>

(a) Sample space: $S = \{HH, HT, TH, TT\}$

Random variable:
- $X(HH) = 2$
- $X(HT) = 1$
- $X(TH) = 1$
- $X(TT) = 0$

(b) PMF:
$$\begin{align}
p_X(0) &= P(\{TT\}) = \frac{1}{4} \\
p_X(1) &= P(\{HT, TH\}) = \frac{2}{4} = \frac{1}{2} \\
p_X(2) &= P(\{HH\}) = \frac{1}{4}
\end{align}$$

(c) Direct: $P(X \leq 1) = p_X(0) + p_X(1) = \frac{1}{4} + \frac{1}{2} = \frac{3}{4}$

CDF:
$$F_X(x) = \begin{cases}
0 & \text{if } x < 0 \\
\frac{1}{4} & \text{if } 0 \leq x < 1 \\
\frac{3}{4} & \text{if } 1 \leq x < 2 \\
1 & \text{if } x \geq 2
\end{cases}$$

$F_X(1) = \frac{3}{4}$ ✓ (matches)
</details>

### Advanced Level

**Problem 6:** A data scientist models the number of server crashes per week as a random variable $X$ with PMF:

$$p_X(x) = \frac{c}{x!} \quad \text{for } x \in \{0, 1, 2, 3\}$$

(a) Find $c$ (hint: $0! = 1, 1! = 1, 2! = 2, 3! = 6$).
(b) What is the probability of at least one crash?
(c) If each crash costs $\$1000$, what's the probability the weekly cost exceeds $\$2000$?

<details>
<summary>Solution</summary>

(a) Normalization:
$$\sum_{x=0}^{3} \frac{c}{x!} = 1$$
$$c\left(\frac{1}{0!} + \frac{1}{1!} + \frac{1}{2!} + \frac{1}{3!}\right) = 1$$
$$c\left(1 + 1 + 0.5 + 0.167\right) = c(2.667) = 1$$
$$c \approx 0.375$$

(b) $P(X \geq 1) = 1 - P(X = 0) = 1 - \frac{c}{0!} = 1 - c = 1 - 0.375 = 0.625$

(c) Cost exceeds $\$2000$ means $X \geq 3$ (3+ crashes):
$$P(X \geq 3) = p_X(3) = \frac{c}{3!} = \frac{0.375}{6} \approx 0.0625$$

About 6.25% chance of exceeding $\$2000$ weekly cost.
</details>

**Problem 7:** Consider a random variable $X$ representing the number of defects in a manufactured item, with PMF:

| $x$ | 0 | 1 | 2 | 3 |
|-----|---|---|---|---|
| $p_X(x)$ | 0.6 | 0.3 | 0.08 | 0.02 |

An item is accepted if defects ≤ 1, inspected if defects = 2, and rejected if defects ≥ 3.

(a) What proportion of items are accepted?
(b) What proportion require inspection or rejection?
(c) Among items with at least one defect, what's the probability of rejection?

<details>
<summary>Solution</summary>

(a) Accepted: $X \leq 1$
$$P(X \leq 1) = p_X(0) + p_X(1) = 0.6 + 0.3 = 0.9$$

90% of items accepted.

(b) Inspection or rejection: $X \geq 2$
$$P(X \geq 2) = p_X(2) + p_X(3) = 0.08 + 0.02 = 0.10$$

10% require inspection or rejection.

(c) Conditional probability: rejection given at least one defect
$$P(X \geq 3 | X \geq 1) = \frac{P(X \geq 3 \text{ and } X \geq 1)}{P(X \geq 1)} = \frac{P(X \geq 3)}{P(X \geq 1)}$$

$$P(X \geq 1) = 1 - p_X(0) = 1 - 0.6 = 0.4$$
$$P(X \geq 3) = 0.02$$

$$P(X \geq 3 | X \geq 1) = \frac{0.02}{0.4} = 0.05$$

Among defective items, 5% are rejected.
</details>

---

## 🎓 Self-Assessment Questions

After completing this week's material, you should be able to answer:

**Conceptual Understanding:**
- [ ] Can you explain the difference between a sample space and a random variable?
- [ ] Why do we need random variables instead of working directly with outcomes?
- [ ] What is the fundamental difference between discrete and continuous random variables?
- [ ] How does a PMF differ from a CDF conceptually?

**Technical Skills:**
- [ ] Can you verify if a given function is a valid PMF?
- [ ] Can you construct the CDF from a given PMF?
- [ ] Can you extract the PMF from a given CDF?
- [ ] Can you calculate probabilities using both PMF and CDF?

**Application:**
- [ ] Can you model a real-world problem using a discrete random variable?
- [ ] Can you implement custom PMF and CDF functions in Python?
- [ ] Can you use scipy.stats to work with discrete distributions?
- [ ] Can you interpret PMF and CDF plots in the context of a business problem?

---

## 📚 Quick Reference Summary

### Key Definitions

| Concept | Formula | Purpose |
|---------|---------|---------|
| Random Variable | $X: S \to \mathbb{R}$ | Maps outcomes to numbers |
| PMF (discrete) | $p_X(x) = P(X = x)$ | Probability at specific value |
| CDF | $F_X(x) = P(X \leq x)$ | Cumulative probability |
| PMF → CDF | $F_X(x) = \sum_{t \leq x} p_X(t)$ | Cumulative sum |
| CDF → PMF | $p_X(x_i) = F_X(x_i) - F_X(x_{i-1})$ | Jump size |

### PMF Properties

1. $p_X(x) \geq 0$ for all $x$
2. $\sum_{\text{all } x} p_X(x) = 1$

### CDF Properties

1. $0 \leq F_X(x) \leq 1$
2. $F_X$ is non-decreasing
3. $\lim_{x \to -\infty} F_X(x) = 0$, $\lim_{x \to \infty} F_X(x) = 1$
4. Right-continuous

### Python Quick Reference

```python
from scipy import stats

# Custom discrete RV
rv = stats.rv_discrete(values=([1,2,3], [0.2, 0.5, 0.3]))

# PMF: P(X = x)
rv.pmf(2)  # 0.5

# CDF: P(X ≤ x)
rv.cdf(2)  # 0.7

# Inverse CDF: value at percentile
rv.ppf(0.5)  # median

# Random samples
rv.rvs(size=100)  # 100 random samples
```

---

## 🔜 Next Week Preview

**Week 10: Expectation and Variance**

Building on random variables, we'll learn:
- **Expected value** $E[X]$ - the "average" or "center" of a distribution
- **Variance** $\text{Var}(X)$ - measuring spread around the mean
- Properties of expectation and variance
- Applications to decision-making under uncertainty

**Connection:** PMF and CDF provide the complete probability structure. Expectation and variance summarize this structure with just two numbers!

---

**End of Week 9 Notes**
