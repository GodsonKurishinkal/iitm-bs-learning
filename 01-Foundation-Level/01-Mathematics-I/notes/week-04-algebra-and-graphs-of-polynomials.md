# Week 04: Algebra and Graphs of Polynomials

**Course**: BSMA1001 - Mathematics I
**Level**: Foundation

---

## 1. Polynomial Arithmetic

### 1.1 Theory

Polynomials are expressions with variables raised to non-negative integer powers. Operations on polynomials follow standard algebraic rules.

### 1.2 Mathematical Definition

A polynomial of degree $n$:
$$P(x) = a_nx^n + a_{n-1}x^{n-1} + ... + a_1x + a_0$$

Where:
- $a_n$ is the **leading coefficient** (must be non-zero)
- $a_0$ is the **constant term**
- $n$ is the **degree** of the polynomial

### 1.3 Polynomial Operations

| Operation | Rule |
|-----------|------|
| Addition | Combine like terms |
| Subtraction | Distribute negative, combine like terms |
| Multiplication | FOIL or distribute each term |
| Division | Long division or synthetic division |

### 1.4 Division Algorithm

For polynomials $P(x)$ (dividend) and $D(x)$ (divisor):
$$P(x) = D(x) \cdot Q(x) + R(x)$$

Where:
- $Q(x)$ = quotient
- $R(x)$ = remainder
- $\deg(R) < \deg(D)$

### 1.5 Remainder Theorem

When $P(x)$ is divided by $(x - c)$, the remainder is $P(c)$.

### 1.6 Supply Chain Application

**Retail Context**: Polynomial models capture complex non-linear relationships like:
- Diminishing returns on advertising spend
- Seasonal demand patterns with multiple peaks
- Cost functions with multiple inflection points

---

## 2. X-Intercepts and Multiplicities

### 2.1 Theory

X-intercepts (roots) are values where the polynomial equals zero. The multiplicity of a root affects how the graph behaves at that point.

### 2.2 Mathematical Definition

**Factor Theorem**: If $P(r) = 0$, then $(x - r)$ is a factor of $P(x)$.

**Fundamental Theorem of Algebra**: A polynomial of degree $n$ has exactly $n$ roots (counting multiplicities, including complex roots).

### 2.3 Multiplicity and Graph Behavior

| Multiplicity | Type | Graph Behavior at Root |
|--------------|------|----------------------|
| 1 | Odd | Crosses the x-axis |
| 2 | Even | Touches and bounces off |
| 3 | Odd | Crosses with inflection |
| 4 | Even | Touches with flattening |

**General Rule**:
- **Odd multiplicity**: Graph crosses the x-axis
- **Even multiplicity**: Graph touches and bounces off the x-axis

### 2.4 Finding Roots

Methods for finding polynomial roots:
1. **Factoring** (when possible)
2. **Rational Root Theorem**: Possible rational roots are $\pm \frac{p}{q}$ where $p$ divides $a_0$ and $q$ divides $a_n$
3. **Synthetic Division**: Test potential roots efficiently
4. **Numerical Methods**: For higher-degree polynomials

### 2.5 Supply Chain Application

**Retail Context**: Roots of profit or cost functions identify critical thresholds:
- Price points where profit becomes zero (break-even)
- Inventory levels where carrying costs equal ordering costs
- Demand levels for capacity planning

---

## 3. End Behavior and Turning Points

### 3.1 Theory

End behavior describes what happens to the polynomial as $x$ approaches infinity. Turning points are local maxima and minima.

### 3.2 End Behavior Rules

For polynomial of degree $n$ with leading coefficient $a_n$:

| Degree | Leading Coefficient | Left End ($x \to -\infty$) | Right End ($x \to +\infty$) |
|--------|--------------------|-----------------------------|------------------------------|
| Even | $a_n > 0$ | $\uparrow$ (up) | $\uparrow$ (up) |
| Even | $a_n < 0$ | $\downarrow$ (down) | $\downarrow$ (down) |
| Odd | $a_n > 0$ | $\downarrow$ (down) | $\uparrow$ (up) |
| Odd | $a_n < 0$ | $\uparrow$ (up) | $\downarrow$ (down) |

### 3.3 Turning Points

**Maximum number of turning points** = $n - 1$ (for degree $n$ polynomial)

- A turning point is where the function changes from increasing to decreasing (local max) or vice versa (local min)
- The actual number of turning points may be less than $n - 1$

### 3.4 Sketching Polynomial Graphs

Steps to sketch a polynomial:
1. Find the **y-intercept**: $P(0) = a_0$
2. Find the **x-intercepts** (roots) and their multiplicities
3. Determine **end behavior** from degree and leading coefficient
4. Plot additional points if needed
5. Connect smoothly through all points

### 3.5 Supply Chain Application

**Retail Context**: Turning points in demand curves identify:
- Seasonal peaks and troughs
- Optimal timing for inventory buildup
- Markdown timing decisions
- Capacity planning inflection points

---

## 4. Polynomial Construction

### 4.1 Theory

We can construct a polynomial if we know its roots (with multiplicities) and one additional point to determine the leading coefficient.

### 4.2 Mathematical Definition

Given roots $r_1, r_2, ..., r_k$ with multiplicities $m_1, m_2, ..., m_k$:

$$P(x) = a(x - r_1)^{m_1}(x - r_2)^{m_2}...(x - r_k)^{m_k}$$

Where:
- $a$ is a constant (leading coefficient scaled)
- Total degree = $m_1 + m_2 + ... + m_k$

### 4.3 Construction Process

1. Write the polynomial in factored form using known roots
2. Use an additional known point $(x_0, y_0)$ to solve for $a$:
   $$a = \frac{y_0}{(x_0 - r_1)^{m_1}(x_0 - r_2)^{m_2}...}$$
3. Expand if standard form is needed

### 4.4 Supply Chain Application

**Retail Context**: Constructing polynomials to fit observed patterns allows:
- Creating predictive models for complex seasonal patterns
- Modeling cyclical demand with multiple peaks
- Fitting historical data for forecasting

---

## Summary

| Concept | Key Formula/Rule |
|---------|-----------------|
| Polynomial Form | $P(x) = a_nx^n + a_{n-1}x^{n-1} + ... + a_0$ |
| Division Algorithm | $P(x) = D(x) \cdot Q(x) + R(x)$ |
| Remainder Theorem | $P(x) \div (x-c)$ has remainder $P(c)$ |
| Factor Theorem | $P(r) = 0 \Leftrightarrow (x-r)$ is a factor |
| Max Turning Points | $n - 1$ for degree $n$ |
| Odd Multiplicity | Graph crosses x-axis |
| Even Multiplicity | Graph bounces off x-axis |

## Key Takeaways

1. **Polynomial operations** follow algebraic rules with the division algorithm

2. **Roots have multiplicities** affecting graph behavior (cross vs. bounce)

3. **End behavior** is determined by degree and leading coefficient sign

4. **Polynomials can be constructed** from known roots and multiplicities

5. **Maximum turning points** = degree minus 1

---

## Next Week Preview

Week 5 covers **Functions and Transformations** - we'll explore function tests, exponential functions, and function composition.

---

*IIT Madras BS Degree in Data Science*
