# Week 03: Functions of Random Variables

**Course**: BSST1002 - Statistics II
**Level**: Foundation

---

## Learning Objectives
- Understand transformations of random variables
- Master the Jacobian method for finding new distributions
- Apply LOTUS (Law of the Unconscious Statistician)

---

## 1. Transformations of Random Variables

### 1.1 Theory

When we apply functions to random variables, we get **new random variables** with different distributions.

If $X$ is a random variable and $g$ is a function, then:

$$Y = g(X)$$

is also a random variable.

### 1.2 Common Transformations

| Transformation | Formula | Example |
|----------------|---------|---------|
| **Linear** | $Y = aX + b$ | Revenue = Price × Quantity + Fee |
| **Logarithmic** | $Y = \log(X)$ | Log-transformed demand |
| **Exponential** | $Y = e^X$ | Growth models |
| **Square** | $Y = X^2$ | Squared error |
| **Power** | $Y = X^n$ | Non-linear effects |

### 1.3 Effect on Distribution Parameters (Linear Case)

For $Y = aX + b$:

| Parameter | Transformation |
|-----------|----------------|
| **Mean** | $E[Y] = aE[X] + b$ |
| **Variance** | $\text{Var}(Y) = a^2 \text{Var}(X)$ |
| **Std Dev** | $\sigma_Y = |a| \sigma_X$ |

---

## 2. Finding Distributions of Transformations

### 2.1 Discrete Case

For discrete $X$ with PMF $p_X(x)$ and $Y = g(X)$:

$$p_Y(y) = \sum_{x: g(x) = y} p_X(x)$$

Sum probabilities of all $x$ values that map to the same $y$.

### 2.2 Continuous Case: Jacobian Method

For continuous $X$ with PDF $f_X(x)$ and $Y = g(X)$ where $g$ is **monotonic**:

$$f_Y(y) = f_X(g^{-1}(y)) \cdot \left|\frac{d}{dy}g^{-1}(y)\right|$$

Where:
- $g^{-1}(y)$ is the inverse function
- $\left|\frac{d}{dy}g^{-1}(y)\right|$ is the **Jacobian** (absolute value of derivative)

### 2.3 Jacobian Method Steps

1. Find the inverse: $x = g^{-1}(y)$
2. Compute the derivative: $\frac{dx}{dy}$
3. Take absolute value: $|dx/dy|$
4. Substitute into formula: $f_Y(y) = f_X(g^{-1}(y)) \cdot |dx/dy|$

### 2.4 Example: Linear Transformation

If $X \sim N(\mu, \sigma^2)$ and $Y = aX + b$:

Then $Y \sim N(a\mu + b, a^2\sigma^2)$

---

## 3. Law of the Unconscious Statistician (LOTUS)

### 3.1 Theory

**LOTUS** allows us to find expected values of functions **without deriving the full distribution** of the transformation.

### 3.2 Mathematical Definition

For $Y = g(X)$:

| Type | Formula |
|------|---------|
| **Discrete** | $E[g(X)] = \sum_x g(x) \cdot p_X(x)$ |
| **Continuous** | $E[g(X)] = \int_{-\infty}^{\infty} g(x) \cdot f_X(x) \, dx$ |

### 3.3 Why "Unconscious"?

The formula looks like we're computing $E[Y]$ using the distribution of $X$ (not $Y$), as if we're "unconscious" of the transformation.

### 3.4 Common Applications of LOTUS

| Quantity | LOTUS Formula |
|----------|---------------|
| **$E[X^2]$** | $\sum x^2 p(x)$ or $\int x^2 f(x) dx$ |
| **$E[e^X]$** | $\sum e^x p(x)$ or $\int e^x f(x) dx$ |
| **Variance** | $\text{Var}(X) = E[X^2] - (E[X])^2$ |

### 3.5 Supply Chain Application

**Retail Context**:
- **Expected profit**: $E[\text{Revenue} - \text{Cost}]$
- **Expected holding cost** under uncertain demand
- **Expected service level** given demand distribution
- **Expected squared error** of forecasts

---

## 4. Special Transformations

### 4.1 Sum of Random Variables

If $Z = X + Y$:
- $E[Z] = E[X] + E[Y]$ (always)
- $\text{Var}(Z) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$

### 4.2 Product of Random Variables

If $Z = XY$:
- $E[Z] = E[X]E[Y] + \text{Cov}(X,Y)$
- If independent: $E[XY] = E[X] \cdot E[Y]$

### 4.3 Ratio of Random Variables

If $Z = X/Y$:
- Generally complex; no simple formula
- Often use simulation or approximation

---

## 5. Moment Generating Functions (Preview)

### 5.1 Definition

The **MGF** of $X$ is:

$$M_X(t) = E[e^{tX}]$$

### 5.2 Why Useful

- MGF uniquely determines the distribution
- Moments can be found by differentiation: $E[X^n] = M_X^{(n)}(0)$
- MGF of sum = product of MGFs (if independent)

---

## Summary Table

| Concept | Formula | Application |
|---------|---------|-------------|
| **Transformation** | $Y = g(X)$ | New random variable |
| **Jacobian** | $f_Y(y) = f_X(g^{-1}(y)) \cdot \|dg^{-1}/dy\|$ | Find PDF of transformation |
| **LOTUS** | $E[g(X)] = \sum g(x)p(x)$ | Expected value directly |
| **Linear E[Y]** | $E[aX+b] = aE[X] + b$ | Revenue calculations |
| **Linear Var(Y)** | $\text{Var}(aX+b) = a^2\text{Var}(X)$ | Scaled variability |

---

## Key Takeaways

1. **Functions of RVs** create new RVs with transformed distributions
2. **Jacobian formula** handles monotonic continuous transformations
3. **LOTUS**: Find $E[g(X)]$ directly without deriving distribution of $g(X)$
4. These tools enable **profit, loss, and cost analysis** under uncertainty

---

## Next Week Preview

Week 4 covers **Continuous Joint Distributions** - bivariate Normal and more.

---

*IIT Madras BS Degree in Data Science*
