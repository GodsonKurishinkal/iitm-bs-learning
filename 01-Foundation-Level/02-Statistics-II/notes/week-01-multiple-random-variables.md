# Week 01: Multiple Random Variables

**Course**: BSST1002 - Statistics II
**Level**: Foundation

---

## Learning Objectives
- Understand joint distributions for multiple variables
- Master marginal and conditional distributions
- Calculate and interpret covariance and correlation

---

## 1. Joint Distributions

### 1.1 Theory

When analyzing **multiple random variables together**, we need joint distributions that describe their combined behavior.

### 1.2 Mathematical Definitions

| Type | Definition |
|------|------------|
| **Joint PMF** (discrete) | $p_{X,Y}(x,y) = P(X = x, Y = y)$ |
| **Joint PDF** (continuous) | $f_{X,Y}(x,y)$ where $P((X,Y) \in A) = \iint_A f(x,y) \, dx \, dy$ |

### 1.3 Properties

| Property | Discrete | Continuous |
|----------|----------|------------|
| **Non-negative** | $p(x,y) \geq 0$ | $f(x,y) \geq 0$ |
| **Sums/Integrates to 1** | $\sum_x \sum_y p(x,y) = 1$ | $\iint f(x,y) \, dx \, dy = 1$ |

### 1.4 Joint CDF

$$F_{X,Y}(x,y) = P(X \leq x, Y \leq y)$$

### 1.5 Supply Chain Application

**Retail Context**:
- **Joint demand** across multiple products
- **Price and quantity sold** — understanding elasticity
- **Lead time and demand during lead time** — safety stock calculation

---

## 2. Marginal Distributions

### 2.1 Theory

**Marginal distributions** recover individual variable behavior from the joint distribution by summing (or integrating) over the other variable.

### 2.2 Mathematical Definition

| Type | Formula |
|------|---------|
| **Discrete** | $p_X(x) = \sum_y p_{X,Y}(x,y)$ |
| **Continuous** | $f_X(x) = \int_{-\infty}^{\infty} f_{X,Y}(x,y) \, dy$ |

### 2.3 Interpretation

- The marginal distribution of $X$ ignores $Y$
- Sum across all possible values of the other variable
- Recovers the individual variable's distribution

### 2.4 Supply Chain Application

**Retail Context**:
- From joint product demand, get **individual product demand**
- From joint (store, product) sales, get **total product sales** across stores

---

## 3. Conditional Distributions

### 3.1 Theory

**Conditional distributions** describe one variable given a specific value of another.

### 3.2 Mathematical Definition

| Type | Formula |
|------|---------|
| **Discrete** | $p_{Y|X}(y|x) = \frac{p_{X,Y}(x,y)}{p_X(x)}$ |
| **Continuous** | $f_{Y|X}(y|x) = \frac{f_{X,Y}(x,y)}{f_X(x)}$ |

### 3.3 Properties

- Conditional distribution is a valid probability distribution
- Sums/integrates to 1 over $y$ for fixed $x$

### 3.4 Supply Chain Application

**Retail Context**:
- **Demand given promotion status** — demand | promo = Yes
- **Lead time given supplier** — lead time | supplier = A
- **Conversion rate given traffic source** — conversion | source = organic

---

## 4. Covariance

### 4.1 Theory

**Covariance** measures how two variables move together.
- **Positive covariance**: Variables increase together
- **Negative covariance**: One increases as other decreases
- **Zero covariance**: No linear relationship

### 4.2 Mathematical Definition

$$\text{Cov}(X, Y) = E[(X - \mu_X)(Y - \mu_Y)]$$

**Computational formula**:

$$\text{Cov}(X, Y) = E[XY] - E[X]E[Y]$$

### 4.3 Properties of Covariance

| Property | Formula |
|----------|---------|
| **Symmetry** | $\text{Cov}(X, Y) = \text{Cov}(Y, X)$ |
| **Variance** | $\text{Cov}(X, X) = \text{Var}(X)$ |
| **Linearity** | $\text{Cov}(aX + b, Y) = a \cdot \text{Cov}(X, Y)$ |
| **Addition** | $\text{Cov}(X + Y, Z) = \text{Cov}(X, Z) + \text{Cov}(Y, Z)$ |
| **Independence** | If independent: $\text{Cov}(X, Y) = 0$ |

### 4.4 Variance of Sum

$$\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X, Y)$$

---

## 5. Correlation

### 5.1 Theory

**Correlation** standardizes covariance to $[-1, 1]$, making it interpretable regardless of units.

### 5.2 Mathematical Definition

$$\rho_{XY} = \frac{\text{Cov}(X, Y)}{\sigma_X \sigma_Y}$$

### 5.3 Interpretation

| $\rho$ Value | Interpretation |
|--------------|----------------|
| $\rho = 1$ | Perfect positive linear relationship |
| $\rho = -1$ | Perfect negative linear relationship |
| $\rho = 0$ | No linear relationship |
| $0 < |\rho| < 0.3$ | Weak |
| $0.3 < |\rho| < 0.7$ | Moderate |
| $0.7 < |\rho| < 1$ | Strong |

### 5.4 Covariance vs Correlation

| Aspect | Covariance | Correlation |
|--------|------------|-------------|
| **Range** | $(-\infty, \infty)$ | $[-1, 1]$ |
| **Units** | Product of X and Y units | Unitless |
| **Interpretability** | Scale-dependent | Scale-independent |

---

## Summary Table

| Concept | Definition | Formula | Supply Chain Application |
|---------|------------|---------|--------------------------|
| **Joint Distribution** | Combined behavior of X and Y | $p_{X,Y}(x,y)$ | Multi-product demand |
| **Marginal** | Individual distribution from joint | $p_X(x) = \sum_y p(x,y)$ | Total product demand |
| **Conditional** | Distribution given other variable | $p_{Y|X} = p_{X,Y}/p_X$ | Demand given promo |
| **Covariance** | How variables move together | $E[XY] - E[X]E[Y]$ | Product demand correlation |
| **Correlation** | Standardized covariance | $\text{Cov}/(\sigma_X \sigma_Y)$ | Portfolio risk |

---

## Key Takeaways

1. **Joint distributions** describe multiple variables together
2. **Marginals**: Sum/integrate over other variables to recover individual distributions
3. **Conditionals**: Fix one variable, renormalize
4. **Covariance** measures linear relationship; **correlation** standardizes to $[-1, 1]$

---

## Next Week Preview

Week 2 covers **Independence and Expected Values** for multiple variables.

---

*IIT Madras BS Degree in Data Science*
