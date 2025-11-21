---
Date: 2025-11-21
Course: BSMA1001 - Mathematics for Data Science I
Level: Foundation
Week: 4 of 12
Source: IIT Madras Mathematics for Data Science I Week 4
Topic Area: Mathematics
Tags: #BSMA1001 #Polynomials #Algebra #Week4 #Foundation
---

# Week 4: Algebra of Polynomials

## Overview

Polynomials are fundamental mathematical objects that form the backbone of algebraic manipulation and serve as essential tools in data science applications. This week explores the structure, operations, and properties of polynomials, building a foundation for understanding more complex mathematical concepts used in machine learning algorithms, curve fitting, and approximation theory.

**Learning Objectives:**
- Understand polynomial structure and terminology
- Master polynomial operations (addition, subtraction, multiplication, division)
- Apply the division algorithm and remainder theorem
- Factor polynomials using various techniques
- Solve polynomial equations
- Connect polynomial concepts to data science applications

**Key Concepts:** Polynomial degree, coefficients, roots, division algorithm, remainder theorem, factor theorem, synthetic division, polynomial factorization

---

## 1. Polynomial Fundamentals

### 1.1 Definition and Terminology

**Definition:** A polynomial in variable $x$ is an expression of the form:

$$P(x) = a_n x^n + a_{n-1} x^{n-1} + \cdots + a_2 x^2 + a_1 x + a_0$$

where:
- $a_0, a_1, \ldots, a_n$ are **coefficients** (real or complex numbers)
- $n$ is a non-negative integer
- $a_n \neq 0$ (the leading coefficient)
- $n$ is the **degree** of the polynomial, denoted $\deg(P)$

**Special Terminology:**
- **Constant term**: $a_0$ (the term with no variable)
- **Linear term**: $a_1 x$ (degree 1)
- **Quadratic term**: $a_2 x^2$ (degree 2)
- **Leading term**: $a_n x^n$ (highest degree term)
- **Monic polynomial**: A polynomial with leading coefficient $a_n = 1$

**Classification by Degree:**

| Degree | Name | General Form | Example |
|--------|------|--------------|---------|
| 0 | Constant | $a_0$ | $5$ |
| 1 | Linear | $a_1 x + a_0$ | $3x - 2$ |
| 2 | Quadratic | $a_2 x^2 + a_1 x + a_0$ | $x^2 - 4x + 4$ |
| 3 | Cubic | $a_3 x^3 + a_2 x^2 + a_1 x + a_0$ | $2x^3 + x - 1$ |
| 4 | Quartic | $a_4 x^4 + \cdots$ | $x^4 - 16$ |
| 5 | Quintic | $a_5 x^5 + \cdots$ | $x^5 - 1$ |

### 1.2 Polynomial Equality

**Definition:** Two polynomials are equal if and only if all their corresponding coefficients are equal.

If $P(x) = a_n x^n + \cdots + a_0$ and $Q(x) = b_m x^m + \cdots + b_0$, then:

$$P(x) = Q(x) \iff n = m \text{ and } a_i = b_i \text{ for all } i$$

**Example 1.1: Testing Polynomial Equality**

Are $P(x) = 2x^2 + 3x - 1$ and $Q(x) = 2x^2 + 3x - 1$ equal?

**Solution:**
Comparing coefficients:
- Coefficient of $x^2$: $2 = 2$ ✓
- Coefficient of $x$: $3 = 3$ ✓
- Constant term: $-1 = -1$ ✓

All coefficients match, so $P(x) = Q(x)$.

---

## 2. Polynomial Operations

### 2.1 Addition and Subtraction

**Rule:** Add or subtract corresponding coefficients.

**Algebraic Definition:** If $P(x) = \sum_{i=0}^{n} a_i x^i$ and $Q(x) = \sum_{i=0}^{m} b_i x^i$, then:

$$(P + Q)(x) = \sum_{i=0}^{\max(n,m)} (a_i + b_i) x^i$$

**Example 2.1: Polynomial Addition**

Add $P(x) = 3x^3 + 2x^2 - x + 5$ and $Q(x) = x^3 - 4x^2 + 3x - 2$.

**Solution:**
\begin{align*}
P(x) + Q(x) &= (3x^3 + 2x^2 - x + 5) + (x^3 - 4x^2 + 3x - 2) \\
&= (3 + 1)x^3 + (2 - 4)x^2 + (-1 + 3)x + (5 - 2) \\
&= 4x^3 - 2x^2 + 2x + 3
\end{align*}

**Example 2.2: Polynomial Subtraction**

Subtract $Q(x) = 2x^2 + 3x - 1$ from $P(x) = 5x^2 - x + 4$.

**Solution:**
\begin{align*}
P(x) - Q(x) &= (5x^2 - x + 4) - (2x^2 + 3x - 1) \\
&= 5x^2 - x + 4 - 2x^2 - 3x + 1 \\
&= (5 - 2)x^2 + (-1 - 3)x + (4 + 1) \\
&= 3x^2 - 4x + 5
\end{align*}

### 2.2 Polynomial Multiplication

**Rule:** Use the distributive property and combine like terms.

**Degree Property:** If $P(x)$ has degree $n$ and $Q(x)$ has degree $m$, then:

$$\deg(P \cdot Q) = \deg(P) + \deg(Q) = n + m$$

**Example 2.3: Multiplying Linear Polynomials**

Multiply $(2x + 3)(x - 4)$.

**Solution:**
\begin{align*}
(2x + 3)(x - 4) &= 2x \cdot x + 2x \cdot (-4) + 3 \cdot x + 3 \cdot (-4) \\
&= 2x^2 - 8x + 3x - 12 \\
&= 2x^2 - 5x - 12
\end{align*}

**Example 2.4: Multiplying Higher-Degree Polynomials**

Multiply $P(x) = x^2 + 2x - 1$ by $Q(x) = x + 3$.

**Solution:**
\begin{align*}
(x^2 + 2x - 1)(x + 3) &= x^2 \cdot x + x^2 \cdot 3 + 2x \cdot x + 2x \cdot 3 - 1 \cdot x - 1 \cdot 3 \\
&= x^3 + 3x^2 + 2x^2 + 6x - x - 3 \\
&= x^3 + 5x^2 + 5x - 3
\end{align*}

**Verification:** $\deg(P \cdot Q) = 2 + 1 = 3$ ✓

### 2.3 Special Product Formulas

These formulas are frequently used in polynomial manipulation:

1. **Square of a Binomial:**
   $$(a + b)^2 = a^2 + 2ab + b^2$$
   $$(a - b)^2 = a^2 - 2ab + b^2$$

2. **Difference of Squares:**
   $$a^2 - b^2 = (a + b)(a - b)$$

3. **Cube of a Binomial:**
   $$(a + b)^3 = a^3 + 3a^2b + 3ab^2 + b^3$$
   $$(a - b)^3 = a^3 - 3a^2b + 3ab^2 - b^3$$

4. **Sum and Difference of Cubes:**
   $$a^3 + b^3 = (a + b)(a^2 - ab + b^2)$$
   $$a^3 - b^3 = (a - b)(a^2 + ab + b^2)$$

**Example 2.5: Using Special Products**

Expand $(3x - 2)^2$ and $(x^2 + 4)(x^2 - 4)$.

**Solution:**

For $(3x - 2)^2$:
$$(3x - 2)^2 = (3x)^2 - 2(3x)(2) + 2^2 = 9x^2 - 12x + 4$$

For $(x^2 + 4)(x^2 - 4)$:
$$(x^2 + 4)(x^2 - 4) = (x^2)^2 - 4^2 = x^4 - 16$$

---

## 3. Polynomial Division

### 3.1 Division Algorithm

**Theorem (Division Algorithm):** For any polynomials $P(x)$ and $D(x)$ with $D(x) \neq 0$, there exist unique polynomials $Q(x)$ (quotient) and $R(x)$ (remainder) such that:

$$P(x) = Q(x) \cdot D(x) + R(x)$$

where either $R(x) = 0$ or $\deg(R) < \deg(D)$.

**Key Relationships:**
- $P(x)$ = **Dividend** (polynomial being divided)
- $D(x)$ = **Divisor** (polynomial we're dividing by)
- $Q(x)$ = **Quotient** (result of division)
- $R(x)$ = **Remainder** (what's left over)

### 3.2 Long Division Method

**Example 3.1: Polynomial Long Division**

Divide $P(x) = 2x^3 + 5x^2 - 3x + 7$ by $D(x) = x + 2$.

**Solution:**

```
           2x² + x - 5
        ________________
x + 2 | 2x³ + 5x² - 3x + 7
        2x³ + 4x²
        __________
              x² - 3x
              x² + 2x
              _______
                 -5x + 7
                 -5x - 10
                 ________
                      17
```

**Step-by-step:**
1. Divide leading term: $2x^3 \div x = 2x^2$
2. Multiply: $2x^2(x + 2) = 2x^3 + 4x^2$
3. Subtract: $(2x^3 + 5x^2) - (2x^3 + 4x^2) = x^2$
4. Bring down: $x^2 - 3x$
5. Repeat: $x^2 \div x = x$
6. Continue until degree of remainder < degree of divisor

**Result:**
$$2x^3 + 5x^2 - 3x + 7 = (x + 2)(2x^2 + x - 5) + 17$$

where $Q(x) = 2x^2 + x - 5$ and $R(x) = 17$.

### 3.3 Synthetic Division

**Purpose:** A shortcut method for dividing by linear polynomials of the form $(x - c)$.

**Example 3.2: Synthetic Division**

Divide $P(x) = 3x^3 - 2x^2 + 5x - 7$ by $(x - 2)$ using synthetic division.

**Solution:**

For divisor $(x - 2)$, use $c = 2$:

```
2 |  3  -2   5  -7
  |     6   8  26
  |________________
     3   4  13  19
```

**Steps:**
1. Write coefficients: $3, -2, 5, -7$
2. Bring down first coefficient: $3$
3. Multiply by $c = 2$: $3 \times 2 = 6$
4. Add to next coefficient: $-2 + 6 = 4$
5. Repeat: $4 \times 2 = 8$, then $5 + 8 = 13$
6. Repeat: $13 \times 2 = 26$, then $-7 + 26 = 19$

**Result:**
- Quotient: $Q(x) = 3x^2 + 4x + 13$
- Remainder: $R(x) = 19$

$$P(x) = (x - 2)(3x^2 + 4x + 13) + 19$$

---

## 4. Remainder and Factor Theorems

### 4.1 Remainder Theorem

**Theorem:** When a polynomial $P(x)$ is divided by $(x - c)$, the remainder is $P(c)$.

**Proof:**
By the division algorithm:
$$P(x) = Q(x)(x - c) + R$$

where $R$ is a constant (since $\deg(R) < \deg(x - c) = 1$).

Substituting $x = c$:
$$P(c) = Q(c)(c - c) + R = 0 + R = R$$

**Example 4.1: Using the Remainder Theorem**

Find the remainder when $P(x) = x^3 - 4x^2 + 6x - 8$ is divided by $(x - 3)$.

**Solution:**
By the Remainder Theorem, remainder = $P(3)$:
\begin{align*}
P(3) &= 3^3 - 4(3)^2 + 6(3) - 8 \\
&= 27 - 36 + 18 - 8 \\
&= 1
\end{align*}

The remainder is $1$.

### 4.2 Factor Theorem

**Theorem:** $(x - c)$ is a factor of $P(x)$ if and only if $P(c) = 0$.

**Proof:**
- If $(x - c)$ is a factor, then $P(x) = Q(x)(x - c)$ for some $Q(x)$
- Substituting $x = c$: $P(c) = Q(c)(c - c) = 0$
- Conversely, if $P(c) = 0$, then by Remainder Theorem, remainder = 0, so $(x - c)$ divides $P(x)$

**Example 4.2: Applying the Factor Theorem**

Is $(x - 2)$ a factor of $P(x) = x^3 - 6x^2 + 11x - 6$?

**Solution:**
Test if $P(2) = 0$:
\begin{align*}
P(2) &= 2^3 - 6(2)^2 + 11(2) - 6 \\
&= 8 - 24 + 22 - 6 \\
&= 0
\end{align*}

Since $P(2) = 0$, by the Factor Theorem, $(x - 2)$ is a factor of $P(x)$.

**Example 4.3: Finding Unknown Coefficients**

If $(x - 1)$ is a factor of $P(x) = x^3 + ax^2 + bx - 6$, and $P(2) = 0$, find $a$ and $b$.

**Solution:**

From $(x - 1)$ being a factor: $P(1) = 0$
$$1 + a + b - 6 = 0$$
$$a + b = 5 \quad \text{...(1)}$$

From $P(2) = 0$:
$$8 + 4a + 2b - 6 = 0$$
$$4a + 2b = -2$$
$$2a + b = -1 \quad \text{...(2)}$$

Solving the system:
From (1): $b = 5 - a$
Substitute into (2): $2a + (5 - a) = -1$
$$a + 5 = -1$$
$$a = -6$$

Therefore: $b = 5 - (-6) = 11$

**Answer:** $a = -6$, $b = 11$

---

## 5. Polynomial Factorization

### 5.1 Factoring Techniques

**Purpose:** Express a polynomial as a product of lower-degree polynomials.

**Common Techniques:**

1. **Common Factor Extraction**
   $$6x^3 + 9x^2 = 3x^2(2x + 3)$$

2. **Grouping**
   $$x^3 + 2x^2 + 3x + 6 = x^2(x + 2) + 3(x + 2) = (x + 2)(x^2 + 3)$$

3. **Quadratic Patterns**
   - $a^2 - b^2 = (a + b)(a - b)$
   - $a^2 + 2ab + b^2 = (a + b)^2$
   - $a^2 - 2ab + b^2 = (a - b)^2$

4. **Rational Root Theorem + Factor Theorem**

**Example 5.1: Factoring by Grouping**

Factor $P(x) = x^3 - 2x^2 - 9x + 18$.

**Solution:**
Group terms:
\begin{align*}
P(x) &= (x^3 - 2x^2) + (-9x + 18) \\
&= x^2(x - 2) - 9(x - 2) \\
&= (x - 2)(x^2 - 9) \\
&= (x - 2)(x + 3)(x - 3)
\end{align*}

### 5.2 Rational Root Theorem

**Theorem:** If a polynomial $P(x) = a_n x^n + \cdots + a_1 x + a_0$ with integer coefficients has a rational root $\frac{p}{q}$ (in lowest terms), then:
- $p$ divides $a_0$ (constant term)
- $q$ divides $a_n$ (leading coefficient)

**Example 5.2: Using Rational Root Theorem**

Factor $P(x) = 2x^3 - x^2 - 13x - 6$ completely.

**Solution:**

**Step 1:** Find possible rational roots
- Factors of $a_0 = -6$: $\pm 1, \pm 2, \pm 3, \pm 6$
- Factors of $a_n = 2$: $\pm 1, \pm 2$
- Possible rational roots: $\pm 1, \pm 2, \pm 3, \pm 6, \pm \frac{1}{2}, \pm \frac{3}{2}$

**Step 2:** Test candidates using Factor Theorem

Test $x = -1$:
$$P(-1) = 2(-1)^3 - (-1)^2 - 13(-1) - 6 = -2 - 1 + 13 - 6 = 4 \neq 0$$

Test $x = 3$:
$$P(3) = 2(3)^3 - (3)^2 - 13(3) - 6 = 54 - 9 - 39 - 6 = 0$$ ✓

**Step 3:** Divide by $(x - 3)$ using synthetic division

```
3 |  2  -1  -13  -6
  |     6   15   6
  |_________________
     2   5    2   0
```

Quotient: $2x^2 + 5x + 2$

**Step 4:** Factor the quotient
$$2x^2 + 5x + 2 = (2x + 1)(x + 2)$$

**Final Factorization:**
$$P(x) = (x - 3)(2x + 1)(x + 2)$$

**Example 5.3: Complete Factorization**

Factor $P(x) = x^4 - 5x^2 + 4$ completely.

**Solution:**

This is a quadratic in disguise. Let $u = x^2$:
$$P(x) = u^2 - 5u + 4 = (u - 4)(u - 1)$$

Substituting back:
$$P(x) = (x^2 - 4)(x^2 - 1) = (x + 2)(x - 2)(x + 1)(x - 1)$$

---

## 6. Roots and Zeros of Polynomials

### 6.1 Definitions

**Definition:** A **root** (or **zero**) of a polynomial $P(x)$ is a value $c$ such that $P(c) = 0$.

**Terminology:**
- If $(x - c)$ appears $k$ times as a factor, then $c$ is a root of **multiplicity** $k$
- Multiplicity 1: **simple root**
- Multiplicity 2: **double root**
- Multiplicity 3: **triple root**

**Example 6.1: Finding Roots**

Find all roots of $P(x) = x^3 - 6x^2 + 11x - 6$.

**Solution:**

Using Rational Root Theorem, possible roots: $\pm 1, \pm 2, \pm 3, \pm 6$

Test $x = 1$:
$$P(1) = 1 - 6 + 11 - 6 = 0$$ ✓

Divide by $(x - 1)$ using synthetic division:
```
1 |  1  -6  11  -6
  |     1  -5   6
  |________________
     1  -5   6   0
```

Quotient: $x^2 - 5x + 6 = (x - 2)(x - 3)$

**Roots:** $x = 1, 2, 3$ (all simple roots)

### 6.2 Fundamental Theorem of Algebra

**Theorem:** Every polynomial of degree $n \geq 1$ with complex coefficients has exactly $n$ roots (counting multiplicities) in the complex numbers.

**Consequence:** A polynomial of degree $n$ can be written as:
$$P(x) = a_n(x - r_1)(x - r_2) \cdots (x - r_n)$$

where $r_1, r_2, \ldots, r_n$ are the roots (possibly repeated, possibly complex).

**Example 6.2: Multiplicity of Roots**

Analyze the roots of $P(x) = (x - 2)^3(x + 1)^2(x - 5)$.

**Solution:**
- Root $x = 2$ with multiplicity 3
- Root $x = -1$ with multiplicity 2
- Root $x = 5$ with multiplicity 1
- Total number of roots (counting multiplicities): $3 + 2 + 1 = 6$
- Degree of polynomial: 6 ✓

---

## 7. Data Science Applications

### 7.1 Polynomial Regression

**Purpose:** Fit a polynomial curve to data points for prediction.

**Model:** For data points $(x_1, y_1), (x_2, y_2), \ldots, (x_n, y_n)$, fit:
$$y = \beta_0 + \beta_1 x + \beta_2 x^2 + \cdots + \beta_d x^d$$

**Use Case:** When relationship between variables is nonlinear.

**Example Application:**
- Predicting housing prices based on size (quadratic relationship)
- Modeling temperature variations over time
- Approximating complex functions

### 7.2 Feature Engineering

**Polynomial Features:** Create new features by computing powers and interactions:

For features $x_1, x_2$, degree-2 polynomial features:
$$\{1, x_1, x_2, x_1^2, x_1 x_2, x_2^2\}$$

**Example:** In scikit-learn:
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)
```

### 7.3 Taylor Series Approximation

**Purpose:** Approximate complex functions using polynomials.

**Taylor Polynomial:** The $n$-th degree Taylor polynomial for $f(x)$ centered at $a$:
$$P_n(x) = f(a) + f'(a)(x-a) + \frac{f''(a)}{2!}(x-a)^2 + \cdots + \frac{f^{(n)}(a)}{n!}(x-a)^n$$

**Application:** Numerical computation of functions (sin, cos, exp) in calculators and computers.

### 7.4 Error Correcting Codes

**Reed-Solomon Codes:** Use polynomial evaluation and interpolation over finite fields for:
- QR codes
- CD/DVD error correction
- Deep space communication

**Key Idea:** Encode data as coefficients of a polynomial, evaluate at multiple points, and use redundancy to recover from errors.

### 7.5 Interpolation

**Problem:** Given $n$ points, find a polynomial of degree at most $n-1$ passing through all points.

**Lagrange Interpolation:** Construct polynomial explicitly:
$$P(x) = \sum_{i=1}^{n} y_i \prod_{j \neq i} \frac{x - x_j}{x_i - x_j}$$

**Application:** Data imputation, signal processing, computer graphics.

---

## 8. Common Pitfalls and Misconceptions

### 8.1 Degree Misconceptions

❌ **Wrong:** "The degree of $(x^2 + 1)(x - 1)$ is $2 + 1 = 3$"
✅ **Correct:** Must expand to verify: $(x^2 + 1)(x - 1) = x^3 - x^2 + x - 1$, so degree is 3.

**Rule:** $\deg(P \cdot Q) = \deg(P) + \deg(Q)$ only when both are non-zero.

### 8.2 Division Errors

❌ **Wrong:** Forgetting to check $\deg(R) < \deg(D)$
✅ **Correct:** Always verify remainder has smaller degree than divisor.

❌ **Wrong:** Confusing divisor $(x - c)$ with $(x + c)$ in synthetic division
✅ **Correct:** For $(x - c)$, use $+c$ in synthetic division; for $(x + c)$, use $-c$.

### 8.3 Factorization Mistakes

❌ **Wrong:** $x^2 + 4 = (x + 2)^2$
✅ **Correct:** $x^2 + 4$ cannot be factored over real numbers (it's $(x + 2i)(x - 2i)$ over complex numbers).

❌ **Wrong:** Assuming all polynomials factor nicely
✅ **Correct:** Some polynomials are **irreducible** (cannot be factored into lower-degree polynomials with real coefficients).

### 8.4 Root Finding Errors

❌ **Wrong:** "If $P(x)$ has degree 3, it has 3 real roots"
✅ **Correct:** It has 3 roots counting multiplicities, but some may be complex.

**Example:** $P(x) = x^3 + x$ has roots $0, i, -i$ (only one real root).

---

## 9. Worked Examples

### Example 9.1: Complete Problem Solving

**Problem:** Find all values of $k$ such that $(x - 2)$ is a factor of $P(x) = x^3 - kx^2 + 4x - 8$.

**Solution:**

By Factor Theorem, $(x - 2)$ is a factor ⟺ $P(2) = 0$

Calculate $P(2)$:
\begin{align*}
P(2) &= 2^3 - k(2)^2 + 4(2) - 8 \\
&= 8 - 4k + 8 - 8 \\
&= 8 - 4k
\end{align*}

Set $P(2) = 0$:
$$8 - 4k = 0$$
$$k = 2$$

**Answer:** $k = 2$

**Verification:** When $k = 2$, $P(x) = x^3 - 2x^2 + 4x - 8$
Factor by grouping:
$$P(x) = x^2(x - 2) + 4(x - 2) = (x - 2)(x^2 + 4)$$

Indeed, $(x - 2)$ is a factor. ✓

### Example 9.2: Complex Factorization

**Problem:** Factor $P(x) = x^4 + x^3 - 7x^2 - x + 6$ completely.

**Solution:**

**Step 1:** Use Rational Root Theorem
Possible rational roots: $\pm 1, \pm 2, \pm 3, \pm 6$

Test $x = 1$:
$$P(1) = 1 + 1 - 7 - 1 + 6 = 0$$ ✓

**Step 2:** Synthetic division by $(x - 1)$
```
1 |  1   1  -7  -1   6
  |      1   2  -5  -6
  |____________________
     1   2  -5  -6   0
```

Quotient: $Q(x) = x^3 + 2x^2 - 5x - 6$

**Step 3:** Factor $Q(x)$
Test $x = -1$:
$$Q(-1) = -1 + 2 + 5 - 6 = 0$$ ✓

Synthetic division by $(x + 1)$:
```
-1 |  1   2  -5  -6
   |     -1  -1   6
   |________________
      1   1  -6   0
```

Quotient: $R(x) = x^2 + x - 6$

**Step 4:** Factor $R(x)$
$$x^2 + x - 6 = (x + 3)(x - 2)$$

**Final Answer:**
$$P(x) = (x - 1)(x + 1)(x + 3)(x - 2)$$

**Roots:** $x = 1, -1, -3, 2$ (all simple roots)

### Example 9.3: Application Problem

**Problem:** A data scientist is fitting a cubic model to predict sales based on advertising spend. The model is:
$$S(x) = ax^3 + bx^2 + cx + d$$

where $S$ is sales (in $1000s) and $x$ is ad spend (in $1000s).

Given:
- When no advertising ($x = 0$), baseline sales are $10$ (i.e., $S(0) = 10$)
- The model has critical points at $x = 1$ and $x = 3$
- At $x = 2$, sales are $26$

Find the polynomial model.

**Solution:**

From $S(0) = 10$: $d = 10$

Critical points occur where $S'(x) = 0$:
$$S'(x) = 3ax^2 + 2bx + c$$

Critical points at $x = 1$ and $x = 3$ means:
$$S'(x) = 3a(x - 1)(x - 3) = 3a(x^2 - 4x + 3) = 3ax^2 - 12ax + 9a$$

Comparing coefficients:
- $3a = 3a$ ✓
- $2b = -12a \Rightarrow b = -6a$
- $c = 9a$

From $S(2) = 26$:
$$a(2)^3 + b(2)^2 + c(2) + 10 = 26$$
$$8a + 4b + 2c + 10 = 26$$
$$8a + 4(-6a) + 2(9a) = 16$$
$$8a - 24a + 18a = 16$$
$$2a = 16$$
$$a = 8$$

Therefore: $b = -6(8) = -48$, $c = 9(8) = 72$, $d = 10$

**Final Model:**
$$S(x) = 8x^3 - 48x^2 + 72x + 10$$

---

## 10. Practice Problems

### Basic Level

**Problem 1:** Add $(3x^2 - 2x + 5)$ and $(x^2 + 4x - 3)$.

**Problem 2:** Multiply $(x + 5)(x - 3)$.

**Problem 3:** Divide $x^3 + 2x^2 - 5x + 12$ by $(x + 3)$ using synthetic division.

**Problem 4:** Use the Remainder Theorem to find the remainder when $P(x) = 2x^3 - 3x^2 + x - 5$ is divided by $(x - 2)$.

**Problem 5:** Is $(x + 1)$ a factor of $x^3 + 2x^2 - x - 2$? Use the Factor Theorem.

### Intermediate Level

**Problem 6:** Factor completely: $x^3 - 3x^2 - 10x + 24$

**Problem 7:** Find all rational roots of $P(x) = 2x^3 - 9x^2 + 10x - 3$ and factor completely.

**Problem 8:** If $(x - 1)$ and $(x + 2)$ are factors of $x^3 + ax^2 + bx - 4$, find $a$ and $b$.

**Problem 9:** Determine the multiplicity of each root of $P(x) = x^4 - 5x^3 + 6x^2$.

**Problem 10:** Divide $2x^4 - 3x^3 + x - 5$ by $x^2 - 2x + 1$ using long division.

### Advanced Level

**Problem 11:** Prove that if $P(x)$ is a polynomial with integer coefficients and $P(0)$ and $P(1)$ are both odd integers, then $P(x)$ has no integer roots.

**Problem 12:** Find a cubic polynomial with roots $1, -2, 3$ such that $P(0) = -12$.

**Problem 13:** If $\alpha$ and $\beta$ are roots of $x^2 - px + q = 0$, express $\alpha^3 + \beta^3$ in terms of $p$ and $q$.

**Problem 14:** A polynomial $P(x)$ of degree 4 satisfies $P(x) - P(-x) = 0$ for all $x$. What can you conclude about $P(x)$?

**Problem 15:** Design a polynomial regression experiment: Generate 20 data points from $y = x^2 + 2x + 1 + \epsilon$ (where $\epsilon$ is random noise) and fit polynomials of degrees 1, 2, and 5. Discuss overfitting and underfitting.

---

## Summary and Key Takeaways

### Core Concepts Mastered

1. **Polynomial Structure**
   - Definition, degree, coefficients, terminology
   - Classification by degree
   - Polynomial equality

2. **Operations**
   - Addition/subtraction: combine like terms
   - Multiplication: distributive property, degree addition
   - Division: division algorithm, long division, synthetic division

3. **Fundamental Theorems**
   - **Division Algorithm**: $P(x) = Q(x)D(x) + R(x)$
   - **Remainder Theorem**: Remainder = $P(c)$
   - **Factor Theorem**: $(x - c)$ is factor ⟺ $P(c) = 0$
   - **Fundamental Theorem of Algebra**: Degree $n$ ⟹ $n$ complex roots

4. **Factorization Techniques**
   - Common factors
   - Grouping
   - Special products
   - Rational Root Theorem
   - Complete factorization

5. **Applications in Data Science**
   - Polynomial regression
   - Feature engineering
   - Function approximation
   - Error correction
   - Interpolation

### Essential Formulas

| Concept | Formula |
|---------|---------|
| Degree of product | $\deg(P \cdot Q) = \deg(P) + \deg(Q)$ |
| Division Algorithm | $P(x) = Q(x)D(x) + R(x)$ |
| Remainder Theorem | $R = P(c)$ when dividing by $(x - c)$ |
| Factor Theorem | $(x - c)$ factor ⟺ $P(c) = 0$ |
| Difference of squares | $a^2 - b^2 = (a+b)(a-b)$ |
| Sum of cubes | $a^3 + b^3 = (a+b)(a^2-ab+b^2)$ |

### Connections to Other Topics

- **Week 3 (Quadratic Functions)**: Special case of polynomials (degree 2)
- **Week 5 (Functions)**: Polynomials as a class of functions
- **Week 6 (Exponential Functions)**: Approximation via Taylor polynomials
- **Future (Calculus)**: Derivatives and integrals of polynomials
- **Statistics**: Polynomial regression models
- **Machine Learning**: Feature transformations

### Study Checklist

- [ ] Can perform polynomial operations (add, subtract, multiply, divide)
- [ ] Understand and apply the Division Algorithm
- [ ] Can use both long division and synthetic division
- [ ] Apply Remainder and Factor Theorems correctly
- [ ] Factor polynomials using multiple techniques
- [ ] Find all roots (including complex) of polynomials
- [ ] Understand multiplicity of roots
- [ ] Connect polynomial concepts to data science applications
- [ ] Avoid common pitfalls in polynomial manipulation

---

## Additional Resources

### Recommended Reading

1. **Textbook Sections**: IIT Madras BSMA1001 Week 4 materials
2. **Practice**: Khan Academy - Polynomial Arithmetic
3. **Visualization**: Desmos polynomial graphing
4. **Applications**: "Introduction to Statistical Learning" - Chapter 7 (Polynomial Regression)

### Online Tools

- **Symbolab**: Step-by-step polynomial division and factorization
- **WolframAlpha**: Verify factorizations and find roots
- **SciPy**: `numpy.polynomial` module for computational work

### Next Week Preview

**Week 5: Functions - Deep Dive**
- Formal definition of functions
- Domain, range, and graphs
- Function composition and inverses
- Special types of functions
- Building on polynomial functions as examples

---

**End of Week 4 Notes**

*These notes are part of the IIT Madras BS in Data Science Foundation Level coursework.*
