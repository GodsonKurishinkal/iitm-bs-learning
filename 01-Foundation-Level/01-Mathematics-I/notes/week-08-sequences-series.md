---
Date: 2025-11-21
Course: BSMA1001 - Mathematics for Data Science I
Level: Foundation
Week: 8 of 12
Source: IIT Madras Mathematics for Data Science I Week 8
Topic Area: Mathematics
Tags: #BSMA1001 #Sequences #Series #Week8 #Foundation
---

# Week 8: Sequences and Series

## Overview

Sequences and series are fundamental concepts in mathematics with extensive applications in data science, particularly in algorithm analysis, numerical methods, probability theory, and machine learning. Understanding convergence, summation, and infinite series is essential for working with iterative algorithms, Taylor series approximations, and asymptotic analysis.

**Learning Objectives:**
- Understand sequences and their convergence properties
- Work with arithmetic and geometric sequences
- Master summation notation and properties
- Analyze infinite series and convergence tests
- Apply sequences and series to algorithm analysis
- Understand Taylor series and approximations

**Key Concepts:** Sequences, limits, convergence, arithmetic/geometric progressions, summation, infinite series, convergence tests, Taylor series

---

## 1. Sequences

### 1.1 Definition

A **sequence** is an ordered list of numbers following a specific pattern or rule.

**Notation**: $\{a_n\}$ or $\{a_1, a_2, a_3, \ldots\}$ or $(a_n)_{n=1}^{\infty}$

**General Term**: $a_n = f(n)$ where $n \in \mathbb{N}$

**Example 1.1: Sequences**

**a) Sequence**: $\{2, 4, 6, 8, 10, \ldots\}$
- General term: $a_n = 2n$
- Pattern: Even positive integers

**b) Sequence**: $\{1, \frac{1}{2}, \frac{1}{3}, \frac{1}{4}, \ldots\}$
- General term: $a_n = \frac{1}{n}$
- Pattern: Reciprocals

**c) Sequence**: $\{1, -1, 1, -1, 1, \ldots\}$
- General term: $a_n = (-1)^{n+1}$
- Pattern: Alternating

### 1.2 Types of Sequences

**Increasing**: $a_{n+1} > a_n$ for all $n$
- Example: $\{1, 2, 3, 4, \ldots\}$

**Decreasing**: $a_{n+1} < a_n$ for all $n$
- Example: $\{10, 9, 8, 7, \ldots\}$

**Bounded Above**: $a_n \leq M$ for some constant $M$ and all $n$

**Bounded Below**: $a_n \geq m$ for some constant $m$ and all $n$

**Bounded**: Both bounded above and below

**Example 1.2: Classifying Sequences**

**a)** $a_n = \frac{n}{n+1} = \frac{1}{1 + 1/n}$
- Increasing: $a_{n+1} = \frac{n+1}{n+2} > \frac{n}{n+1} = a_n$ ✓
- Bounded: $0 < a_n < 1$ for all $n$ ✓

**b)** $a_n = (-1)^n$
- Not monotonic (alternates)
- Bounded: $-1 \leq a_n \leq 1$ ✓

### 1.3 Limit of a Sequence

**Definition**: A sequence $\{a_n\}$ **converges** to limit $L$ if:

$$\lim_{n \to \infty} a_n = L$$

Meaning: $a_n$ gets arbitrarily close to $L$ as $n$ increases.

**Notation**: $a_n \to L$ as $n \to \infty$

If no such $L$ exists, the sequence **diverges**.

**Example 1.3: Finding Limits**

**a)** $\lim_{n \to \infty} \frac{1}{n} = 0$

As $n$ grows, $\frac{1}{n}$ approaches 0.

**b)** $\lim_{n \to \infty} \frac{n}{n+1}$

Divide numerator and denominator by $n$:
$$\lim_{n \to \infty} \frac{1}{1 + 1/n} = \frac{1}{1 + 0} = 1$$

**c)** $\lim_{n \to \infty} \frac{2n^2 + 3n}{n^2 - 1}$

Divide by $n^2$:
$$\lim_{n \to \infty} \frac{2 + 3/n}{1 - 1/n^2} = \frac{2 + 0}{1 - 0} = 2$$

**d)** $\lim_{n \to \infty} (-1)^n$ does not exist (oscillates between -1 and 1)

### 1.4 Limit Laws for Sequences

If $\lim_{n \to \infty} a_n = L$ and $\lim_{n \to \infty} b_n = M$, then:

1. **Sum**: $\lim_{n \to \infty} (a_n + b_n) = L + M$

2. **Difference**: $\lim_{n \to \infty} (a_n - b_n) = L - M$

3. **Product**: $\lim_{n \to \infty} (a_n \cdot b_n) = L \cdot M$

4. **Quotient**: $\lim_{n \to \infty} \frac{a_n}{b_n} = \frac{L}{M}$ (if $M \neq 0$)

5. **Constant Multiple**: $\lim_{n \to \infty} (c \cdot a_n) = c \cdot L$

6. **Power**: $\lim_{n \to \infty} (a_n)^p = L^p$

---

## 2. Arithmetic Sequences

### 2.1 Definition

An **arithmetic sequence** has a constant difference between consecutive terms.

**General Form**: $a_n = a_1 + (n-1)d$

Where:
- $a_1$: First term
- $d$: Common difference
- $n$: Term number

**Recursive Form**: $a_{n+1} = a_n + d$

**Example 2.1: Arithmetic Sequences**

**a) Sequence**: $3, 7, 11, 15, 19, \ldots$
- $a_1 = 3$, $d = 4$
- General term: $a_n = 3 + (n-1) \cdot 4 = 4n - 1$
- 10th term: $a_{10} = 4(10) - 1 = 39$

**b) Sequence**: $100, 95, 90, 85, \ldots$
- $a_1 = 100$, $d = -5$
- General term: $a_n = 100 + (n-1)(-5) = 105 - 5n$
- When is $a_n = 0$? $105 - 5n = 0 \implies n = 21$

### 2.2 Sum of Arithmetic Sequence

**Sum of first $n$ terms**:

$$S_n = \frac{n}{2}(a_1 + a_n) = \frac{n}{2}(2a_1 + (n-1)d)$$

**Derivation**: Write sum forward and backward, then add:
\begin{align*}
S_n &= a_1 + (a_1+d) + (a_1+2d) + \cdots + a_n \\
S_n &= a_n + (a_n-d) + (a_n-2d) + \cdots + a_1 \\
2S_n &= (a_1+a_n) + (a_1+a_n) + \cdots + (a_1+a_n) \quad [n \text{ times}] \\
2S_n &= n(a_1 + a_n) \\
S_n &= \frac{n}{2}(a_1 + a_n)
\end{align*}

**Example 2.2: Sum of Arithmetic Sequence**

Find the sum: $2 + 5 + 8 + 11 + \cdots + 50$

**Solution:**

First, find $n$ (number of terms):
$$a_n = a_1 + (n-1)d$$
$$50 = 2 + (n-1) \cdot 3$$
$$48 = 3(n-1)$$
$$n = 17$$

Now find sum:
$$S_{17} = \frac{17}{2}(2 + 50) = \frac{17 \cdot 52}{2} = 442$$

---

## 3. Geometric Sequences

### 3.1 Definition

A **geometric sequence** has a constant ratio between consecutive terms.

**General Form**: $a_n = a_1 \cdot r^{n-1}$

Where:
- $a_1$: First term
- $r$: Common ratio
- $n$: Term number

**Recursive Form**: $a_{n+1} = r \cdot a_n$

**Example 3.1: Geometric Sequences**

**a) Sequence**: $2, 6, 18, 54, \ldots$
- $a_1 = 2$, $r = 3$
- General term: $a_n = 2 \cdot 3^{n-1}$
- 8th term: $a_8 = 2 \cdot 3^7 = 4374$

**b) Sequence**: $100, 50, 25, 12.5, \ldots$
- $a_1 = 100$, $r = 0.5$
- General term: $a_n = 100 \cdot (0.5)^{n-1}$
- Limit: $\lim_{n \to \infty} a_n = 0$ (converges to 0)

### 3.2 Sum of Geometric Sequence

**Finite sum** (first $n$ terms):

$$S_n = a_1 \frac{1 - r^n}{1 - r} = \frac{a_1(1-r^n)}{1-r} \quad (r \neq 1)$$

If $r = 1$: $S_n = n \cdot a_1$

**Infinite sum** (if $|r| < 1$):

$$S_{\infty} = \sum_{n=1}^{\infty} a_1 r^{n-1} = \frac{a_1}{1-r}$$

**Convergence**: The infinite series converges only if $|r| < 1$

**Example 3.2: Finite Geometric Sum**

Find the sum: $3 + 6 + 12 + 24 + \cdots + 3072$

**Solution:**

$a_1 = 3$, $r = 2$

Find $n$:
$$a_n = 3 \cdot 2^{n-1} = 3072$$
$$2^{n-1} = 1024 = 2^{10}$$
$$n = 11$$

Sum:
$$S_{11} = 3 \cdot \frac{1 - 2^{11}}{1 - 2} = 3 \cdot \frac{1 - 2048}{-1} = 3 \cdot 2047 = 6141$$

**Example 3.3: Infinite Geometric Sum**

Find the sum: $1 + \frac{1}{2} + \frac{1}{4} + \frac{1}{8} + \cdots$

**Solution:**

$a_1 = 1$, $r = \frac{1}{2}$

Since $|r| = \frac{1}{2} < 1$, series converges:

$$S_{\infty} = \frac{1}{1 - 1/2} = \frac{1}{1/2} = 2$$

---

## 4. Summation Notation

### 4.1 Sigma Notation

**Definition**: $\sum_{i=m}^{n} a_i = a_m + a_{m+1} + \cdots + a_n$

Where:
- $\sum$: Sigma (summation symbol)
- $i$: Index of summation
- $m$: Lower limit
- $n$: Upper limit
- $a_i$: General term

**Example 4.1: Evaluating Summations**

**a)** $\sum_{i=1}^{5} i^2 = 1^2 + 2^2 + 3^2 + 4^2 + 5^2 = 1 + 4 + 9 + 16 + 25 = 55$

**b)** $\sum_{k=3}^{6} (2k+1) = 7 + 9 + 11 + 13 = 40$

**c)** $\sum_{n=0}^{4} 3^n = 1 + 3 + 9 + 27 + 81 = 121$

### 4.2 Summation Properties

1. **Constant Multiple**: $\sum_{i=m}^{n} c \cdot a_i = c \sum_{i=m}^{n} a_i$

2. **Sum**: $\sum_{i=m}^{n} (a_i + b_i) = \sum_{i=m}^{n} a_i + \sum_{i=m}^{n} b_i$

3. **Difference**: $\sum_{i=m}^{n} (a_i - b_i) = \sum_{i=m}^{n} a_i - \sum_{i=m}^{n} b_i$

4. **Index Shift**: $\sum_{i=m}^{n} a_i = \sum_{j=m+k}^{n+k} a_{j-k}$

### 4.3 Common Summation Formulas

| Sum | Formula |
|-----|---------|
| $\sum_{i=1}^{n} 1$ | $n$ |
| $\sum_{i=1}^{n} i$ | $\frac{n(n+1)}{2}$ |
| $\sum_{i=1}^{n} i^2$ | $\frac{n(n+1)(2n+1)}{6}$ |
| $\sum_{i=1}^{n} i^3$ | $\left[\frac{n(n+1)}{2}\right]^2$ |
| $\sum_{i=0}^{n} r^i$ | $\frac{1-r^{n+1}}{1-r}$ (geometric) |

**Example 4.2: Using Formulas**

Evaluate $\sum_{i=1}^{100} (3i^2 - 2i + 5)$

**Solution:**

Split into separate sums:
$$\sum_{i=1}^{100} (3i^2 - 2i + 5) = 3\sum_{i=1}^{100} i^2 - 2\sum_{i=1}^{100} i + \sum_{i=1}^{100} 5$$

Apply formulas:
\begin{align*}
&= 3 \cdot \frac{100 \cdot 101 \cdot 201}{6} - 2 \cdot \frac{100 \cdot 101}{2} + 5 \cdot 100 \\
&= 3 \cdot 338350 - 2 \cdot 5050 + 500 \\
&= 1015050 - 10100 + 500 \\
&= 1005450
\end{align*}

---

## 5. Infinite Series

### 5.1 Definition

An **infinite series** is the sum of infinitely many terms:

$$\sum_{n=1}^{\infty} a_n = a_1 + a_2 + a_3 + \cdots$$

**Partial Sum**: $S_n = \sum_{i=1}^{n} a_i$ (sum of first $n$ terms)

**Convergence**: If $\lim_{n \to \infty} S_n = S$ (a finite number), the series **converges** to $S$.

**Divergence**: If the limit doesn't exist or is infinite, the series **diverges**.

### 5.2 Convergence Tests

#### Test 1: Divergence Test (nth Term Test)

**If** $\lim_{n \to \infty} a_n \neq 0$, **then** $\sum a_n$ diverges.

**Note**: If $\lim_{n \to \infty} a_n = 0$, the test is **inconclusive** (may converge or diverge).

**Example 5.1: Divergence Test**

**a)** $\sum_{n=1}^{\infty} \frac{n}{2n+1}$

$$\lim_{n \to \infty} \frac{n}{2n+1} = \lim_{n \to \infty} \frac{1}{2 + 1/n} = \frac{1}{2} \neq 0$$

Series **diverges** by divergence test.

**b)** $\sum_{n=1}^{\infty} \frac{1}{n^2}$

$$\lim_{n \to \infty} \frac{1}{n^2} = 0$$

Test is inconclusive (but this series actually converges).

#### Test 2: Geometric Series Test

$$\sum_{n=0}^{\infty} ar^n \text{ converges to } \frac{a}{1-r} \text{ if } |r| < 1$$

$$\sum_{n=0}^{\infty} ar^n \text{ diverges if } |r| \geq 1$$

**Example 5.2: Geometric Series**

**a)** $\sum_{n=0}^{\infty} \frac{1}{3^n}$

$r = \frac{1}{3}$, $|r| < 1$, so converges to $\frac{1}{1-1/3} = \frac{3}{2}$

**b)** $\sum_{n=0}^{\infty} (-2)^n$

$r = -2$, $|r| = 2 > 1$, so diverges

#### Test 3: p-Series Test

$$\sum_{n=1}^{\infty} \frac{1}{n^p} \begin{cases}
\text{converges if } p > 1 \\
\text{diverges if } p \leq 1
\end{cases}$$

**Special Cases**:
- **Harmonic series** ($p=1$): $\sum_{n=1}^{\infty} \frac{1}{n}$ diverges
- **p=2**: $\sum_{n=1}^{\infty} \frac{1}{n^2} = \frac{\pi^2}{6}$ converges

#### Test 4: Comparison Test

Given two series $\sum a_n$ and $\sum b_n$ with $a_n, b_n > 0$:

**If** $a_n \leq b_n$ for all $n$:
- If $\sum b_n$ converges, then $\sum a_n$ converges
- If $\sum a_n$ diverges, then $\sum b_n$ diverges

**Example 5.3: Comparison Test**

Does $\sum_{n=1}^{\infty} \frac{1}{n^2 + 1}$ converge?

**Solution:**

For $n \geq 1$: $n^2 + 1 > n^2$, so $\frac{1}{n^2+1} < \frac{1}{n^2}$

We know $\sum \frac{1}{n^2}$ converges (p-series with $p=2$).

By comparison test, $\sum \frac{1}{n^2+1}$ **converges**.

#### Test 5: Ratio Test

For $\sum a_n$ with $a_n > 0$, let $L = \lim_{n \to \infty} \frac{a_{n+1}}{a_n}$

- If $L < 1$: Series converges
- If $L > 1$ (or $L = \infty$): Series diverges
- If $L = 1$: Test inconclusive

**Example 5.4: Ratio Test**

Test convergence of $\sum_{n=1}^{\infty} \frac{n!}{n^n}$

**Solution:**

$$L = \lim_{n \to \infty} \frac{(n+1)!/(n+1)^{n+1}}{n!/n^n}$$

$$= \lim_{n \to \infty} \frac{(n+1) \cdot n!}{(n+1)^{n+1}} \cdot \frac{n^n}{n!}$$

$$= \lim_{n \to \infty} \frac{(n+1) \cdot n^n}{(n+1)^{n+1}}$$

$$= \lim_{n \to \infty} \frac{n^n}{(n+1)^n}$$

$$= \lim_{n \to \infty} \left(\frac{n}{n+1}\right)^n$$

$$= \lim_{n \to \infty} \left(\frac{1}{1+1/n}\right)^n$$

$$= \frac{1}{e} < 1$$

Series **converges** by ratio test.

---

## 6. Taylor Series

### 6.1 Power Series

A **power series** centered at $a$ is:

$$\sum_{n=0}^{\infty} c_n(x-a)^n = c_0 + c_1(x-a) + c_2(x-a)^2 + \cdots$$

**Taylor Series** of $f(x)$ at $x = a$:

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n$$

Where $f^{(n)}(a)$ is the $n$-th derivative of $f$ evaluated at $a$.

**Maclaurin Series** (Taylor series at $a = 0$):

$$f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(0)}{n!}x^n$$

### 6.2 Common Taylor Series

**Exponential**:
$$e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!} = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \cdots$$

**Sine**:
$$\sin(x) = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n+1}}{(2n+1)!} = x - \frac{x^3}{3!} + \frac{x^5}{5!} - \frac{x^7}{7!} + \cdots$$

**Cosine**:
$$\cos(x) = \sum_{n=0}^{\infty} \frac{(-1)^n x^{2n}}{(2n)!} = 1 - \frac{x^2}{2!} + \frac{x^4}{4!} - \frac{x^6}{6!} + \cdots$$

**Natural Logarithm** (for $|x| < 1$):
$$\ln(1+x) = \sum_{n=1}^{\infty} \frac{(-1)^{n+1} x^n}{n} = x - \frac{x^2}{2} + \frac{x^3}{3} - \frac{x^4}{4} + \cdots$$

**Geometric Series**:
$$\frac{1}{1-x} = \sum_{n=0}^{\infty} x^n = 1 + x + x^2 + x^3 + \cdots \quad (|x| < 1)$$

**Example 6.1: Using Taylor Series**

Approximate $e^{0.5}$ using the first 5 terms of the series.

**Solution:**

$$e^x = 1 + x + \frac{x^2}{2!} + \frac{x^3}{3!} + \frac{x^4}{4!} + \cdots$$

For $x = 0.5$:
\begin{align*}
e^{0.5} &\approx 1 + 0.5 + \frac{(0.5)^2}{2} + \frac{(0.5)^3}{6} + \frac{(0.5)^4}{24} \\
&= 1 + 0.5 + 0.125 + 0.0208\overline{3} + 0.0026041\overline{6} \\
&\approx 1.6484
\end{align*}

Actual value: $e^{0.5} \approx 1.6487$ (error < 0.0003)

### 6.3 Applications in Computing

**Approximation**: Computers use Taylor series to compute transcendental functions (sin, cos, exp, log).

**Machine Learning**: Taylor series used in:
- Gradient descent optimization
- Second-order methods (Newton's method)
- Function approximation in neural networks

---

## 7. Applications in Data Science

### 7.1 Algorithm Analysis

**Time Complexity** often involves series:

**Loop example**:
```python
for i in range(n):
    for j in range(i):
        # O(1) operation
```

Total operations: $\sum_{i=1}^{n} i = \frac{n(n+1)}{2} = O(n^2)$

**Binary search**: Operations at each level form geometric sequence
- Level 0: $n$ elements
- Level 1: $n/2$ elements
- Level 2: $n/4$ elements
- Total levels: $\log_2(n)$

Total work: $n + n/2 + n/4 + \cdots = n(1 + 1/2 + 1/4 + \cdots) = 2n = O(n)$

### 7.2 Discounting and Present Value

**Future value with compound interest**:

$$FV = PV(1 + r)^n$$

**Present value** of future cashflows:

$$PV = \sum_{t=1}^{n} \frac{CF_t}{(1+r)^t}$$

This is a geometric series!

**Example 7.1: Perpetuity**

An investment pays $\$1000$ annually forever, with discount rate 5%. What's the present value?

$$PV = \sum_{t=1}^{\infty} \frac{1000}{(1.05)^t} = 1000 \sum_{t=1}^{\infty} \left(\frac{1}{1.05}\right)^t$$

This is geometric series with $r = \frac{1}{1.05} \approx 0.952$:

$$PV = 1000 \cdot \frac{1/1.05}{1 - 1/1.05} = 1000 \cdot \frac{1}{0.05} = \$20,000$$

### 7.3 Markov Chains and Steady State

**Transition matrix powers**: $P^n$ as $n \to \infty$

Involves infinite series of matrix multiplications!

### 7.4 Gradient Descent

**Taylor approximation** in optimization:

$$f(x + \Delta x) \approx f(x) + f'(x)\Delta x + \frac{f''(x)}{2}(\Delta x)^2$$

First-order (gradient descent): Use only first two terms
Second-order (Newton's method): Include second derivative term

### 7.5 Probabilistic Models

**Geometric distribution** (number of trials until first success):

$$P(X = k) = (1-p)^{k-1} p$$

Expected value:
$$E[X] = \sum_{k=1}^{\infty} k(1-p)^{k-1}p = \frac{1}{p}$$

(Derived using infinite series!)

---

## 8. Worked Examples

### Example 8.1: Complex Summation

Evaluate: $\sum_{k=1}^{n} k(k+1)$

**Solution:**

Expand: $k(k+1) = k^2 + k$

$$\sum_{k=1}^{n} k(k+1) = \sum_{k=1}^{n} k^2 + \sum_{k=1}^{n} k$$

Apply formulas:
$$= \frac{n(n+1)(2n+1)}{6} + \frac{n(n+1)}{2}$$

Factor out $\frac{n(n+1)}{6}$:
$$= \frac{n(n+1)}{6}[2n+1 + 3] = \frac{n(n+1)(2n+4)}{6}$$

Simplify:
$$= \frac{n(n+1) \cdot 2(n+2)}{6} = \frac{n(n+1)(n+2)}{3}$$

### Example 8.2: Convergence Analysis

Determine if $\sum_{n=1}^{\infty} \frac{n^2}{2^n}$ converges.

**Solution:**

Use ratio test:
$$L = \lim_{n \to \infty} \frac{(n+1)^2/2^{n+1}}{n^2/2^n}$$

$$= \lim_{n \to \infty} \frac{(n+1)^2}{n^2} \cdot \frac{2^n}{2^{n+1}}$$

$$= \lim_{n \to \infty} \frac{(n+1)^2}{n^2} \cdot \frac{1}{2}$$

$$= \frac{1}{2} \lim_{n \to \infty} \left(\frac{n+1}{n}\right)^2$$

$$= \frac{1}{2} \cdot 1^2 = \frac{1}{2} < 1$$

Series **converges** by ratio test.

### Example 8.3: Finding Taylor Series

Find the first 4 non-zero terms of the Taylor series for $f(x) = e^x \sin(x)$ at $x = 0$.

**Solution:**

**Method 1**: Multiply series for $e^x$ and $\sin(x)$

$$e^x = 1 + x + \frac{x^2}{2} + \frac{x^3}{6} + \cdots$$

$$\sin(x) = x - \frac{x^3}{6} + \frac{x^5}{120} + \cdots$$

Multiply (collect terms up to $x^3$):
\begin{align*}
e^x \sin(x) &= \left(1 + x + \frac{x^2}{2} + \frac{x^3}{6}\right)\left(x - \frac{x^3}{6}\right) \\
&= x + x^2 + \frac{x^3}{2} + \frac{x^4}{6} - \frac{x^3}{6} + \text{higher order} \\
&= x + x^2 + \left(\frac{1}{2} - \frac{1}{6}\right)x^3 + \cdots \\
&= x + x^2 + \frac{x^3}{3} + \cdots
\end{align*}

**Method 2**: Compute derivatives

$f(x) = e^x \sin(x)$

$f(0) = 0$

$f'(x) = e^x \sin(x) + e^x \cos(x) = e^x(\sin x + \cos x)$

$f'(0) = 1$

Continue for $f''(0), f'''(0), \ldots$

---

## 9. Practice Problems

### Basic Level

**Problem 1**: Find the general term for the sequence: $5, 8, 11, 14, 17, \ldots$

**Problem 2**: Find the 15th term of the arithmetic sequence with $a_1 = 7$ and $d = -3$

**Problem 3**: Find the sum: $\sum_{i=1}^{50} (2i - 1)$

**Problem 4**: Determine if the sequence $a_n = \frac{n^2}{n^2 + 1}$ converges, and if so, find the limit

**Problem 5**: Find the sum of the geometric series: $2 + 6 + 18 + 54 + 162$

### Intermediate Level

**Problem 6**: Find the sum: $\sum_{k=1}^{n} (3k^2 - k + 2)$

**Problem 7**: Determine if $\sum_{n=1}^{\infty} \frac{1}{n(n+1)}$ converges, and if so, find the sum

**Problem 8**: Test convergence: $\sum_{n=1}^{\infty} \frac{2^n}{n!}$

**Problem 9**: Find the infinite sum: $\sum_{n=0}^{\infty} \frac{3}{4^n}$

**Problem 10**: How many terms of the series $\sum_{n=1}^{\infty} \frac{1}{n^2}$ are needed to approximate the sum within 0.01?

### Advanced Level

**Problem 11**: Prove that $\sum_{k=1}^{n} k^3 = \left[\frac{n(n+1)}{2}\right]^2$ using induction

**Problem 12**: Find the interval of convergence for the power series $\sum_{n=0}^{\infty} \frac{(x-2)^n}{n+1}$

**Problem 13**: Use the Taylor series for $\ln(1+x)$ to approximate $\ln(1.1)$ to 4 decimal places

**Problem 14**: A ball is dropped from 10 meters. Each bounce reaches 80% of the previous height. Find the total distance traveled.

**Problem 15**: Analyze the time complexity:
```python
for i in range(n):
    for j in range(i, n):
        for k in range(j, n):
            # O(1) operation
```

---

## Summary and Key Takeaways

### Core Concepts Mastered

1. **Sequences**
   - Definition and notation
   - Types: increasing, decreasing, bounded
   - Convergence and limits
   - Limit laws

2. **Arithmetic Sequences**
   - General term: $a_n = a_1 + (n-1)d$
   - Sum formula: $S_n = \frac{n}{2}(a_1 + a_n)$

3. **Geometric Sequences**
   - General term: $a_n = a_1 r^{n-1}$
   - Finite sum: $S_n = a_1\frac{1-r^n}{1-r}$
   - Infinite sum (if $|r|<1$): $S_\infty = \frac{a_1}{1-r}$

4. **Summation Notation**
   - Sigma notation: $\sum_{i=m}^{n} a_i$
   - Properties: linearity, index shifting
   - Common formulas: $\sum i$, $\sum i^2$, $\sum i^3$

5. **Infinite Series**
   - Convergence vs divergence
   - Tests: divergence, geometric, p-series, comparison, ratio

6. **Taylor Series**
   - Power series representations
   - Common series: $e^x$, $\sin x$, $\cos x$, $\ln(1+x)$
   - Applications in approximation

7. **Applications**
   - Algorithm complexity analysis
   - Financial calculations (present value)
   - Optimization (gradient descent)
   - Probability distributions

### Essential Formulas

| Concept | Formula |
|---------|---------|
| **Arithmetic Sum** | $S_n = \frac{n}{2}(a_1 + a_n)$ |
| **Geometric Sum (finite)** | $S_n = a_1\frac{1-r^n}{1-r}$ |
| **Geometric Sum (infinite)** | $S_\infty = \frac{a_1}{1-r}$ if $\|r\|<1$ |
| **Sum of integers** | $\sum_{i=1}^{n} i = \frac{n(n+1)}{2}$ |
| **Sum of squares** | $\sum_{i=1}^{n} i^2 = \frac{n(n+1)(2n+1)}{6}$ |
| **Taylor series** | $f(x) = \sum_{n=0}^{\infty} \frac{f^{(n)}(a)}{n!}(x-a)^n$ |

### Connections to Other Topics

- **Week 6 (Exponentials/Logs)**: Geometric sequences are exponential functions
- **Week 7 (Trigonometry)**: Taylor series for sin and cos
- **Future (Calculus)**: Derivatives and integrals as limits
- **Future (Limits)**: Formal definition of sequence convergence
- **Algorithm Analysis**: Big-O notation and growth rates
- **Machine Learning**: Optimization algorithms use series expansions
- **Probability**: Distributions and expected values

### Study Checklist

- [ ] Understand sequence notation and convergence
- [ ] Can find general term of arithmetic/geometric sequences
- [ ] Master summation formulas and properties
- [ ] Can apply convergence tests to infinite series
- [ ] Understand geometric series convergence condition
- [ ] Know common Taylor series expansions
- [ ] Can use series for approximation
- [ ] Recognize series in algorithm analysis
- [ ] Understand applications in data science
- [ ] Avoid common pitfalls (mixing finite/infinite sums)

---

## Additional Resources

### Recommended Reading

1. **Textbook**: IIT Madras BSMA1001 Week 8 materials
2. **Practice**: Khan Academy - Sequences and Series
3. **Visualization**: 3Blue1Brown - Taylor Series
4. **Applications**: "Introduction to Algorithms" - CLRS

### Online Tools

- **WolframAlpha**: Evaluate sums and test convergence
- **Desmos**: Visualize partial sums
- **SymPy**: Symbolic series manipulation

### Next Week Preview

**Week 9: Limits and Continuity**
- Formal definition of limits
- One-sided limits
- Continuity and discontinuities
- Intermediate Value Theorem
- Applications to optimization

---

**End of Week 8 Notes**

*These notes are part of the IIT Madras BS in Data Science Foundation Level coursework.*
