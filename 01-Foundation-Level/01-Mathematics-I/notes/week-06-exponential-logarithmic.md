---
Date: 2025-11-21
Course: BSMA1001 - Mathematics for Data Science I
Level: Foundation
Week: 6 of 12
Source: IIT Madras Mathematics for Data Science I Week 6
Topic Area: Mathematics
Tags: #BSMA1001 #Exponential #Logarithm #Week6 #Foundation
---

# Week 6: Exponential and Logarithmic Functions

## Overview

Exponential and logarithmic functions are among the most important functions in mathematics and data science. They appear everywhere: population growth, radioactive decay, compound interest, information theory, machine learning algorithms, and complexity analysis. This week provides a comprehensive exploration of these functions, their properties, and their extensive applications in data science.

**Learning Objectives:**
- Understand exponential functions and their properties
- Master logarithmic functions and logarithm laws
- Work with natural logarithms and the number $e$
- Solve exponential and logarithmic equations
- Apply these functions to real-world data science problems
- Understand logarithmic scales and transformations

**Key Concepts:** Exponential growth/decay, base $e$, natural logarithm, logarithm laws, change of base, logistic function, information entropy

---

## 1. Exponential Functions

### 1.1 Definition and Basic Properties

**Definition:** An **exponential function** with base $a > 0$, $a \neq 1$ is defined as:

$$f(x) = a^x$$

where $x$ can be any real number.

**Key Properties:**
1. **Domain**: All real numbers $\mathbb{R}$
2. **Range**: $(0, \infty)$ (always positive, never zero)
3. **$y$-intercept**: $(0, 1)$ since $a^0 = 1$
4. **Horizontal asymptote**: $y = 0$ (x-axis)
5. **One-to-one**: Passes horizontal line test (invertible)

**Behavior:**
- If $a > 1$: **Exponential growth** (increasing function)
- If $0 < a < 1$: **Exponential decay** (decreasing function)

**Example 1.1: Basic Exponential Functions**

Consider $f(x) = 2^x$ and $g(x) = \left(\frac{1}{2}\right)^x$:

| $x$ | $f(x) = 2^x$ | $g(x) = \left(\frac{1}{2}\right)^x$ |
|-----|--------------|--------------------------------------|
| -2  | $\frac{1}{4} = 0.25$ | $4$ |
| -1  | $\frac{1}{2} = 0.5$ | $2$ |
| 0   | $1$ | $1$ |
| 1   | $2$ | $0.5$ |
| 2   | $4$ | $0.25$ |
| 3   | $8$ | $0.125$ |

**Observation**: $g(x) = \left(\frac{1}{2}\right)^x = 2^{-x} = f(-x)$ (reflection of $f$ across y-axis)

### 1.2 Laws of Exponents

For $a, b > 0$ and $x, y \in \mathbb{R}$:

| Law | Formula | Example |
|-----|---------|---------|
| **Product Rule** | $a^x \cdot a^y = a^{x+y}$ | $2^3 \cdot 2^5 = 2^8$ |
| **Quotient Rule** | $\frac{a^x}{a^y} = a^{x-y}$ | $\frac{3^7}{3^4} = 3^3$ |
| **Power Rule** | $(a^x)^y = a^{xy}$ | $(5^2)^3 = 5^6$ |
| **Product of Bases** | $(ab)^x = a^x b^x$ | $(2 \cdot 3)^4 = 2^4 \cdot 3^4$ |
| **Quotient of Bases** | $\left(\frac{a}{b}\right)^x = \frac{a^x}{b^x}$ | $\left(\frac{4}{3}\right)^2 = \frac{16}{9}$ |
| **Zero Exponent** | $a^0 = 1$ | $100^0 = 1$ |
| **Negative Exponent** | $a^{-x} = \frac{1}{a^x}$ | $2^{-3} = \frac{1}{8}$ |

**Example 1.2: Applying Exponent Laws**

Simplify: $\frac{(2^3)^2 \cdot 2^5}{2^7}$

**Solution:**
\begin{align*}
\frac{(2^3)^2 \cdot 2^5}{2^7} &= \frac{2^{3 \cdot 2} \cdot 2^5}{2^7} && \text{(Power rule)} \\
&= \frac{2^6 \cdot 2^5}{2^7} && \text{(Simplify)} \\
&= \frac{2^{6+5}}{2^7} && \text{(Product rule)} \\
&= \frac{2^{11}}{2^7} && \text{(Simplify)} \\
&= 2^{11-7} && \text{(Quotient rule)} \\
&= 2^4 = 16
\end{align*}

### 1.3 The Natural Exponential Function

**The Number $e$**: Euler's number, approximately $e \approx 2.71828...$

**Definition**: $e$ is defined as:
$$e = \lim_{n \to \infty} \left(1 + \frac{1}{n}\right)^n$$

Or equivalently:
$$e = \sum_{n=0}^{\infty} \frac{1}{n!} = 1 + 1 + \frac{1}{2} + \frac{1}{6} + \frac{1}{24} + \cdots$$

**Natural Exponential Function**: $f(x) = e^x$ (also written as $\exp(x)$)

**Why $e$ is Special:**
- Most "natural" base for exponential functions
- Derivative of $e^x$ is $e^x$ (unique property!)
- Appears in continuous growth/decay models
- Foundation of complex analysis and differential equations

**Example 1.3: Computing with $e$**

Evaluate: $e^2$, $e^{-1}$, $e^0$

**Solution:**
- $e^2 \approx (2.71828)^2 \approx 7.389$
- $e^{-1} = \frac{1}{e} \approx 0.368$
- $e^0 = 1$

---

## 2. Logarithmic Functions

### 2.1 Definition

**Definition:** The **logarithm** with base $a$ (where $a > 0$, $a \neq 1$) is the inverse of the exponential function:

$$y = \log_a(x) \iff a^y = x$$

**Read as**: "logarithm base $a$ of $x$" or "log base $a$ of $x$"

**Meaning**: $\log_a(x)$ answers the question: "To what power must we raise $a$ to get $x$?"

**Special Logarithms:**
- **Common logarithm**: $\log_{10}(x)$ (often written as $\log(x)$ in applied contexts)
- **Natural logarithm**: $\log_e(x) = \ln(x)$ (most important in mathematics/data science)
- **Binary logarithm**: $\log_2(x)$ (used in computer science and information theory)

### 2.2 Basic Properties

**Key Properties:**
1. **Domain**: $(0, \infty)$ (only positive numbers)
2. **Range**: All real numbers $\mathbb{R}$
3. **$x$-intercept**: $(1, 0)$ since $\log_a(1) = 0$
4. **Vertical asymptote**: $x = 0$ (y-axis)
5. **One-to-one**: Invertible
6. **Inverse relationship**: $\log_a(a^x) = x$ and $a^{\log_a(x)} = x$

**Example 2.1: Basic Logarithm Evaluations**

Evaluate without a calculator:

**a)** $\log_2(8)$

**Solution**: Ask: "2 to what power gives 8?"
$$2^? = 8 \implies 2^3 = 8$$
$$\log_2(8) = 3$$

**b)** $\log_{10}(1000)$

**Solution**: $10^3 = 1000$
$$\log_{10}(1000) = 3$$

**c)** $\log_5(1)$

**Solution**: $5^0 = 1$
$$\log_5(1) = 0$$

**d)** $\log_3\left(\frac{1}{27}\right)$

**Solution**: $3^{-3} = \frac{1}{27}$
$$\log_3\left(\frac{1}{27}\right) = -3$$

### 2.3 Logarithm Laws

For $a, x, y > 0$ with $a \neq 1$, and $r \in \mathbb{R}$:

| Law | Formula | Name |
|-----|---------|------|
| **Product Rule** | $\log_a(xy) = \log_a(x) + \log_a(y)$ | Sum of logs |
| **Quotient Rule** | $\log_a\left(\frac{x}{y}\right) = \log_a(x) - \log_a(y)$ | Difference of logs |
| **Power Rule** | $\log_a(x^r) = r \log_a(x)$ | Coefficient becomes exponent |
| **Change of Base** | $\log_a(x) = \frac{\log_b(x)}{\log_b(a)}$ | Convert to different base |
| **Special Values** | $\log_a(1) = 0$ and $\log_a(a) = 1$ | Identity values |
| **Inverse Property** | $\log_a(a^x) = x$ and $a^{\log_a(x)} = x$ | Undo each other |

**Example 2.2: Applying Logarithm Laws**

Simplify: $\log_2(32) + \log_2(4) - \log_2(8)$

**Solution:**
\begin{align*}
\log_2(32) + \log_2(4) - \log_2(8) &= \log_2\left(\frac{32 \cdot 4}{8}\right) && \text{(Product and quotient rules)} \\
&= \log_2\left(\frac{128}{8}\right) \\
&= \log_2(16) \\
&= \log_2(2^4) \\
&= 4
\end{align*}

**Example 2.3: Expanding Logarithmic Expressions**

Expand: $\ln\left(\frac{x^3\sqrt{y}}{z^2}\right)$

**Solution:**
\begin{align*}
\ln\left(\frac{x^3\sqrt{y}}{z^2}\right) &= \ln(x^3\sqrt{y}) - \ln(z^2) && \text{(Quotient rule)} \\
&= \ln(x^3) + \ln(\sqrt{y}) - \ln(z^2) && \text{(Product rule)} \\
&= \ln(x^3) + \ln(y^{1/2}) - \ln(z^2) && \text{(Rewrite root)} \\
&= 3\ln(x) + \frac{1}{2}\ln(y) - 2\ln(z) && \text{(Power rule)}
\end{align*}

### 2.4 Change of Base Formula

**Formula**:
$$\log_a(x) = \frac{\log_b(x)}{\log_b(a)} = \frac{\ln(x)}{\ln(a)}$$

**Purpose**: Convert logarithms to a base we can compute (usually base 10 or $e$)

**Example 2.4: Change of Base**

Evaluate $\log_5(20)$ using natural logarithms.

**Solution:**
$$\log_5(20) = \frac{\ln(20)}{\ln(5)} = \frac{2.9957...}{1.6094...} \approx 1.861$$

**Verification**: $5^{1.861} \approx 20$ ✓

---

## 3. Natural Logarithm ($\ln$)

### 3.1 Properties of $\ln(x)$

The natural logarithm $\ln(x) = \log_e(x)$ has special importance:

**Key Properties:**
- $\ln(e) = 1$
- $\ln(1) = 0$
- $\ln(e^x) = x$
- $e^{\ln(x)} = x$
- $\frac{d}{dx}[\ln(x)] = \frac{1}{x}$ (simple derivative)
- $\int \frac{1}{x}dx = \ln|x| + C$ (natural antiderivative)

**Example 3.1: Natural Logarithm Calculations**

**a)** Simplify $\ln(e^5)$

**Solution**: $\ln(e^5) = 5$

**b)** Simplify $e^{3\ln(2)}$

**Solution**:
$$e^{3\ln(2)} = e^{\ln(2^3)} = e^{\ln(8)} = 8$$

**c)** Solve $\ln(x) = 2$

**Solution**: Exponentiate both sides:
$$e^{\ln(x)} = e^2$$
$$x = e^2 \approx 7.389$$

### 3.2 Relationship Between $\log$ and $\ln$

**Conversion formulas:**
$$\ln(x) = \log_e(x) = \frac{\log_{10}(x)}{\log_{10}(e)} \approx 2.303 \cdot \log_{10}(x)$$

$$\log_{10}(x) = \frac{\ln(x)}{\ln(10)} \approx 0.434 \cdot \ln(x)$$

---

## 4. Solving Exponential and Logarithmic Equations

### 4.1 Exponential Equations

**Strategy**: Take logarithm of both sides

**Example 4.1: Solving Exponential Equations**

Solve $2^x = 50$

**Solution:**
\begin{align*}
2^x &= 50 \\
\ln(2^x) &= \ln(50) && \text{(Take ln of both sides)} \\
x \ln(2) &= \ln(50) && \text{(Power rule)} \\
x &= \frac{\ln(50)}{\ln(2)} && \text{(Divide by } \ln(2)\text{)} \\
x &= \frac{3.912}{0.693} \\
x &\approx 5.644
\end{align*}

**Verification**: $2^{5.644} \approx 50$ ✓

**Example 4.2: Equations with $e$**

Solve $3e^{2x} - 7 = 20$

**Solution:**
\begin{align*}
3e^{2x} - 7 &= 20 \\
3e^{2x} &= 27 && \text{(Add 7)} \\
e^{2x} &= 9 && \text{(Divide by 3)} \\
\ln(e^{2x}) &= \ln(9) && \text{(Take ln)} \\
2x &= \ln(9) && \text{(Simplify)} \\
x &= \frac{\ln(9)}{2} \\
x &= \frac{2.197}{2} \approx 1.099
\end{align*}

### 4.2 Logarithmic Equations

**Strategy**: Exponentiate both sides or use logarithm properties

**Example 4.3: Solving Logarithmic Equations**

Solve $\log_2(x) + \log_2(x-3) = 2$

**Solution:**
\begin{align*}
\log_2(x) + \log_2(x-3) &= 2 \\
\log_2(x(x-3)) &= 2 && \text{(Product rule)} \\
2^2 &= x(x-3) && \text{(Exponentiate base 2)} \\
4 &= x^2 - 3x \\
0 &= x^2 - 3x - 4 \\
0 &= (x-4)(x+1) && \text{(Factor)} \\
x &= 4 \text{ or } x = -1
\end{align*}

**Check domain**: $x > 0$ and $x - 3 > 0 \implies x > 3$

Only $x = 4$ satisfies the domain restriction.

**Answer**: $x = 4$

**Example 4.4: Change of Base in Equations**

Solve $\ln(x) = 5\log_{10}(x)$

**Solution:**

Convert $\log_{10}(x)$ to natural log:
$$\ln(x) = 5 \cdot \frac{\ln(x)}{\ln(10)}$$

Let $y = \ln(x)$:
$$y = \frac{5y}{\ln(10)}$$
$$y \ln(10) = 5y$$
$$y(\ln(10) - 5) = 0$$

Either $y = 0$ or $\ln(10) - 5 = 0$

Since $\ln(10) \approx 2.303 \neq 5$:
$$y = 0$$
$$\ln(x) = 0$$
$$x = e^0 = 1$$

**Answer**: $x = 1$

---

## 5. Graphs and Transformations

### 5.1 Graphical Relationship

**Key Observation**: $y = a^x$ and $y = \log_a(x)$ are **inverse functions**

Their graphs are reflections of each other across the line $y = x$.

**Comparing $y = 2^x$ and $y = \log_2(x)$:**

| Feature | $y = 2^x$ | $y = \log_2(x)$ |
|---------|-----------|-----------------|
| Domain | $\mathbb{R}$ | $(0, \infty)$ |
| Range | $(0, \infty)$ | $\mathbb{R}$ |
| Intercept | $(0, 1)$ | $(1, 0)$ |
| Asymptote | Horizontal: $y = 0$ | Vertical: $x = 0$ |
| Behavior | Increases rapidly | Increases slowly |

### 5.2 Transformations

**Exponential**: $f(x) = a \cdot b^{k(x-h)} + c$
- $a$: Vertical stretch/reflection
- $b$: Base (growth/decay rate)
- $k$: Horizontal stretch/compression
- $h$: Horizontal shift
- $c$: Vertical shift

**Logarithmic**: $g(x) = a \cdot \log_b(k(x-h)) + c$
- Similar transformation effects

**Example 5.1: Identifying Transformations**

Describe the transformations from $f(x) = 2^x$ to $g(x) = -2^{x+1} + 3$:

1. Shift left 1 unit: $2^{x+1}$
2. Reflect about x-axis: $-2^{x+1}$
3. Shift up 3 units: $-2^{x+1} + 3$

**New asymptote**: $y = 3$

---

## 6. Applications in Data Science

### 6.1 Exponential Growth and Decay

**General Model**: $N(t) = N_0 e^{kt}$

Where:
- $N(t)$: Quantity at time $t$
- $N_0$: Initial quantity
- $k$: Growth rate ($k > 0$) or decay rate ($k < 0$)
- $t$: Time

**Example 6.1: Population Growth**

A bacteria population grows exponentially. Initially there are 1000 bacteria, and after 3 hours there are 8000.

**a) Find the growth model**

**Solution:**

Use $N(t) = N_0 e^{kt}$ with $N_0 = 1000$:
$$8000 = 1000 e^{3k}$$
$$8 = e^{3k}$$
$$\ln(8) = 3k$$
$$k = \frac{\ln(8)}{3} \approx 0.693$$

**Model**: $N(t) = 1000e^{0.693t}$

**b) When will the population reach 50,000?**

**Solution:**
$$50000 = 1000e^{0.693t}$$
$$50 = e^{0.693t}$$
$$\ln(50) = 0.693t$$
$$t = \frac{\ln(50)}{0.693} \approx 5.65 \text{ hours}$$

### 6.2 Compound Interest

**Formula**: $A = P\left(1 + \frac{r}{n}\right)^{nt}$

Where:
- $A$: Final amount
- $P$: Principal (initial investment)
- $r$: Annual interest rate (decimal)
- $n$: Compounding frequency per year
- $t$: Time in years

**Continuous Compounding**: $A = Pe^{rt}$

**Example 6.2: Investment Growth**

Invest $5000 at 6% annual interest, compounded quarterly. Find the value after 10 years.

**Solution:**
$$A = 5000\left(1 + \frac{0.06}{4}\right)^{4 \cdot 10}$$
$$A = 5000(1.015)^{40}$$
$$A \approx 5000(1.8140)$$
$$A \approx \$9070$$

**With continuous compounding**:
$$A = 5000e^{0.06 \cdot 10} = 5000e^{0.6} \approx \$9110$$

### 6.3 Logarithmic Scales

**Purpose**: Display data spanning many orders of magnitude

**Applications**:
- **Richter scale** (earthquakes): $M = \log_{10}\left(\frac{I}{I_0}\right)$
- **pH scale** (acidity): $\text{pH} = -\log_{10}[\text{H}^+]$
- **Decibels** (sound): $dB = 10\log_{10}\left(\frac{I}{I_0}\right)$

**Example 6.3: Earthquake Magnitude**

An earthquake measuring 7.0 on the Richter scale is how many times more intense than one measuring 5.0?

**Solution:**

Let $I_7$ and $I_5$ be the intensities:
$$7 = \log_{10}\left(\frac{I_7}{I_0}\right) \implies I_7 = 10^7 I_0$$
$$5 = \log_{10}\left(\frac{I_5}{I_0}\right) \implies I_5 = 10^5 I_0$$

Ratio:
$$\frac{I_7}{I_5} = \frac{10^7 I_0}{10^5 I_0} = 10^2 = 100$$

**Answer**: The 7.0 earthquake is **100 times more intense**.

### 6.4 Information Theory and Entropy

**Shannon Entropy**: Measures information content

$$H(X) = -\sum_{i=1}^{n} p_i \log_2(p_i)$$

Where $p_i$ is the probability of outcome $i$.

**Units**: Bits (when using $\log_2$)

**Example 6.4: Entropy of a Coin Flip**

**Fair coin** ($p(\text{H}) = p(\text{T}) = 0.5$):
$$H = -[0.5\log_2(0.5) + 0.5\log_2(0.5)]$$
$$H = -[0.5(-1) + 0.5(-1)] = 1 \text{ bit}$$

**Biased coin** ($p(\text{H}) = 0.9$, $p(\text{T}) = 0.1$):
$$H = -[0.9\log_2(0.9) + 0.1\log_2(0.1)]$$
$$H \approx 0.469 \text{ bits}$$

**Interpretation**: Fair coin has maximum entropy (most uncertainty).

### 6.5 Logistic Regression

**Logistic Function**: $\sigma(x) = \frac{1}{1 + e^{-x}}$

**Properties**:
- S-shaped curve
- Range: $(0, 1)$ (perfect for probabilities)
- Derivative: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$

**Binary Classification**:
$$P(y=1|x) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x)}}$$

**Log-Odds (Logit)**:
$$\text{logit}(p) = \ln\left(\frac{p}{1-p}\right) = \beta_0 + \beta_1 x$$

### 6.6 Log Transformations in Data Analysis

**When to use**:
1. Data is right-skewed
2. Relationships are multiplicative rather than additive
3. Variance increases with mean
4. Need to stabilize variance

**Example 6.5: Log Transform for Linear Regression**

Original relationship: $y = ab^x$ (exponential)

Taking logarithm:
$$\ln(y) = \ln(a) + x\ln(b)$$

Let $Y = \ln(y)$, $A = \ln(a)$, $B = \ln(b)$:
$$Y = A + Bx$$

Now it's linear! Can use standard linear regression on $(x, Y)$.

### 6.7 Complexity Analysis (Big-O Notation)

**Logarithmic complexity**: $O(\log n)$

**Examples**:
- **Binary search**: Divides search space in half each step
- **Balanced tree operations**: Height grows as $\log_2(n)$

**Why it's efficient**: $\log_2(1,000,000) \approx 20$ (only 20 operations!)

**Comparison**:
- $O(1)$: Constant (best)
- $O(\log n)$: Logarithmic (excellent)
- $O(n)$: Linear (good)
- $O(n\log n)$: Linearithmic (acceptable)
- $O(n^2)$: Quadratic (poor for large $n$)
- $O(2^n)$: Exponential (intractable for large $n$)

---

## 7. Common Pitfalls and Misconceptions

### 7.1 Logarithm Domain Errors

❌ **Wrong**: $\log(-5)$ or $\ln(0)$
✅ **Correct**: Logarithms only defined for positive numbers

**Domain of $\log_a(x)$**: $x > 0$ only

### 7.2 Logarithm Property Misapplication

❌ **Wrong**: $\log(x + y) = \log(x) + \log(y)$
✅ **Correct**: $\log(xy) = \log(x) + \log(y)$

❌ **Wrong**: $\log(x^2) = (\log x)^2$
✅ **Correct**: $\log(x^2) = 2\log(x)$

### 7.3 Base Confusion

❌ **Wrong**: Assuming $\log$ always means $\log_{10}$
✅ **Correct**: Check context
- Mathematics: $\log = \log_{10}$ or $\log_e = \ln$
- Computer Science: $\log = \log_2$
- Data Science: Usually $\ln$ (natural log)

### 7.4 Inverse Misunderstanding

❌ **Wrong**: $(\log x)^{-1} = \log(1/x)$
✅ **Correct**:
- $(\log x)^{-1} = \frac{1}{\log x}$ (reciprocal)
- Inverse function: $\log^{-1}(x) = 10^x$ or $e^x$

### 7.5 Exponential Growth Underestimation

❌ **Wrong**: "Exponential just means 'really fast'"
✅ **Correct**: Exponential means proportional to current value

**Example**: Doubling time
- Linear: $1, 2, 3, 4, 5, ...$
- Exponential: $1, 2, 4, 8, 16, 32, ...$

Exponential quickly dwarfs any polynomial!

---

## 8. Worked Examples

### Example 8.1: Comprehensive Problem

**Problem**: Solve $2^{x+1} = 3^{x-1}$ for $x$.

**Solution:**

Take natural log of both sides:
$$\ln(2^{x+1}) = \ln(3^{x-1})$$
$$(x+1)\ln(2) = (x-1)\ln(3)$$
$$x\ln(2) + \ln(2) = x\ln(3) - \ln(3)$$
$$x\ln(2) - x\ln(3) = -\ln(3) - \ln(2)$$
$$x[\ln(2) - \ln(3)] = -[\ln(3) + \ln(2)]$$
$$x = \frac{-[\ln(3) + \ln(2)]}{\ln(2) - \ln(3)}$$
$$x = \frac{\ln(3) + \ln(2)}{\ln(3) - \ln(2)}$$
$$x = \frac{\ln(6)}{\ln(3/2)}$$
$$x \approx \frac{1.7918}{0.4055} \approx 4.42$$

### Example 8.2: Application Problem

**Problem**: Carbon-14 dating uses exponential decay. If a sample has 60% of its original C-14 (half-life = 5730 years), how old is it?

**Solution:**

Decay model: $N(t) = N_0 e^{-kt}$

**Step 1**: Find decay constant $k$ using half-life:
$$0.5N_0 = N_0 e^{-k(5730)}$$
$$0.5 = e^{-5730k}$$
$$\ln(0.5) = -5730k$$
$$k = \frac{-\ln(0.5)}{5730} = \frac{\ln(2)}{5730} \approx 0.000121$$

**Step 2**: Find age when 60% remains:
$$0.6N_0 = N_0 e^{-0.000121t}$$
$$0.6 = e^{-0.000121t}$$
$$\ln(0.6) = -0.000121t$$
$$t = \frac{-\ln(0.6)}{0.000121} = \frac{0.5108}{0.000121} \approx 4221 \text{ years}$$

### Example 8.3: Logarithmic Scale

**Problem**: The pH of lemon juice is 2 and the pH of tomato juice is 4. How many times more acidic is lemon juice?

**Solution:**

pH formula: $\text{pH} = -\log_{10}[\text{H}^+]$

For lemon juice:
$$2 = -\log_{10}[\text{H}^+]_L$$
$$[\text{H}^+]_L = 10^{-2}$$

For tomato juice:
$$4 = -\log_{10}[\text{H}^+]_T$$
$$[\text{H}^+]_T = 10^{-4}$$

Ratio:
$$\frac{[\text{H}^+]_L}{[\text{H}^+]_T} = \frac{10^{-2}}{10^{-4}} = 10^2 = 100$$

**Answer**: Lemon juice is **100 times more acidic** than tomato juice.

---

## 9. Practice Problems

### Basic Level

**Problem 1**: Evaluate without calculator: $\log_3(81)$, $\ln(e^4)$, $\log_2(1/16)$

**Problem 2**: Simplify: $e^{2\ln(x)}$

**Problem 3**: Solve for $x$: $2^x = 32$

**Problem 4**: Expand using logarithm laws: $\log_5(x^2y^3)$

**Problem 5**: Convert to exponential form: $\log_7(49) = 2$

### Intermediate Level

**Problem 6**: Solve: $\ln(x) + \ln(x-2) = \ln(3)$

**Problem 7**: Find the inverse function of $f(x) = 2^{x-1} + 3$

**Problem 8**: Simplify: $\frac{\log(100) + \ln(e^3)}{\log(10)}$

**Problem 9**: A population grows from 5000 to 20000 in 10 years. Find the exponential growth rate.

**Problem 10**: If $\log_2(x) = 3\log_2(y) - \log_2(z)$, express $x$ in terms of $y$ and $z$.

### Advanced Level

**Problem 11**: Solve the system:
$$\begin{cases}
2^x \cdot 3^y = 72 \\
x + y = 5
\end{cases}$$

**Problem 12**: Prove that $\log_a(b) \cdot \log_b(c) \cdot \log_c(a) = 1$

**Problem 13**: Find all values of $x$ satisfying: $\log_4(x^2 - 5x + 6) = \log_4(x - 2)$

**Problem 14**: A radioactive substance has a half-life of 12 years. What percentage remains after 30 years?

**Problem 15**: Design a log transformation experiment: Generate data from $y = 3 \cdot 2^x + \epsilon$ and show how log transformation linearizes the relationship. Fit both untransformed and transformed models and compare.

---

## Summary and Key Takeaways

### Core Concepts Mastered

1. **Exponential Functions**
   - Definition: $f(x) = a^x$
   - Properties: Domain $\mathbb{R}$, Range $(0, \infty)$
   - Growth $(a>1)$ vs Decay $(0<a<1)$
   - Natural exponential: $e^x$

2. **Logarithmic Functions**
   - Definition: Inverse of exponential
   - $y = \log_a(x) \iff a^y = x$
   - Properties: Domain $(0,\infty)$, Range $\mathbb{R}$
   - Natural logarithm: $\ln(x) = \log_e(x)$

3. **Fundamental Laws**
   - **Exponents**: Product, quotient, power rules
   - **Logarithms**: Product, quotient, power, change of base
   - **Inverse property**: $\log_a(a^x) = x$ and $a^{\log_a(x)} = x$

4. **Solving Equations**
   - Exponential: Take logarithm
   - Logarithmic: Exponentiate or combine logs

5. **Applications**
   - Growth/decay models
   - Compound interest
   - Logarithmic scales (Richter, pH, decibels)
   - Information entropy
   - Logistic regression
   - Complexity analysis
   - Data transformations

### Essential Formulas

| Formula | Name/Use |
|---------|----------|
| $N(t) = N_0 e^{kt}$ | Exponential growth/decay |
| $A = Pe^{rt}$ | Continuous compound interest |
| $\log_a(x) = \frac{\ln(x)}{\ln(a)}$ | Change of base |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | Logistic function |
| $H = -\sum p_i \log_2(p_i)$ | Shannon entropy |

### Connections to Other Topics

- **Week 4 (Polynomials)**: Exponentials grow faster than any polynomial
- **Week 5 (Functions)**: Exponential and logarithm are inverse functions
- **Future (Calculus)**: $\frac{d}{dx}[e^x] = e^x$, $\frac{d}{dx}[\ln x] = \frac{1}{x}$
- **Statistics**: Normal distribution, log-normal distribution, exponential distribution
- **Machine Learning**: Logistic regression, softmax, cross-entropy loss
- **Algorithms**: Time complexity, divide-and-conquer analysis

### Study Checklist

- [ ] Understand exponential function properties
- [ ] Master logarithm definition and laws
- [ ] Can solve exponential and logarithmic equations
- [ ] Understand relationship between $e$ and $\ln$
- [ ] Apply change of base formula
- [ ] Recognize exponential growth/decay patterns
- [ ] Understand logarithmic scales
- [ ] Connect to data science applications (logistic regression, entropy)
- [ ] Can apply log transformations to data
- [ ] Avoid common pitfalls (domain errors, property misuse)

---

## Additional Resources

### Recommended Reading

1. **Textbook**: IIT Madras BSMA1001 Week 6 materials
2. **Practice**: Khan Academy - Exponential and Logarithmic Functions
3. **Visualization**: Desmos - Interactive exponential/log graphing
4. **Applications**: "Introduction to Statistical Learning" - Logistic Regression

### Online Tools

- **Desmos**: Graph and compare exponential/logarithmic functions
- **WolframAlpha**: Solve complex exponential/logarithmic equations
- **Calculator**: Log and exponential calculations

### Next Week Preview

**Week 7: Sequences and Series**
- Arithmetic and geometric sequences
- Summation notation
- Convergence and divergence
- Applications to algorithms and data analysis

---

**End of Week 6 Notes**

*These notes are part of the IIT Madras BS in Data Science Foundation Level coursework.*
