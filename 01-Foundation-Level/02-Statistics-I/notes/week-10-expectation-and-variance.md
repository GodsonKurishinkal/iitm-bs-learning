# Week 10: Expectation and Variance

**Course**: BSST1001 - Statistics I
**Level**: Foundation

---

## Learning Objectives
- Master expected value calculations
- Understand variance and standard deviation
- Apply coefficient of variation for comparisons

---

## 1. Expected Value (Mean)

### 1.1 Theory

**Expected value** is the probability-weighted average of all possible values. It represents the "long-run average" of a random variable.

### 1.2 Mathematical Definition

| Type | Formula |
|------|---------|
| **Discrete** | $E[X] = \sum_x x \cdot p(x)$ |
| **Continuous** | $E[X] = \int_{-\infty}^{\infty} x \cdot f(x) \, dx$ |

### 1.3 Properties of Expected Value

| Property | Formula |
|----------|---------|
| **Linearity** | $E[aX + b] = aE[X] + b$ |
| **Addition** | $E[X + Y] = E[X] + E[Y]$ (always) |
| **Multiplication** | $E[XY] = E[X] \cdot E[Y]$ (only if independent) |
| **Constant** | $E[c] = c$ |

### 1.4 Interpretation

- **Center of distribution** — where the probability "balances"
- **Long-run average** — average over many repetitions
- **Fair value** — expected payoff in games/decisions

### 1.5 Supply Chain Application

**Retail Context**:
- **Expected daily demand** — base for inventory planning
- **Expected profit** — guides pricing decisions
- **Expected lead time** — determines reorder timing

---

## 2. Variance and Standard Deviation

### 2.1 Theory

**Variance** measures the spread of values around the mean. **Standard deviation** is in the same units as the original variable.

### 2.2 Mathematical Definition

$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

$$\sigma = SD(X) = \sqrt{\text{Var}(X)}$$

### 2.3 Alternative Formulas

| Approach | Formula |
|----------|---------|
| **Definition** | $\text{Var}(X) = E[(X - \mu)^2]$ |
| **Computational** | $\text{Var}(X) = E[X^2] - \mu^2$ |
| **Discrete** | $\text{Var}(X) = \sum_x (x - \mu)^2 \cdot p(x)$ |
| **Continuous** | $\text{Var}(X) = \int (x - \mu)^2 f(x) \, dx$ |

### 2.4 Properties of Variance

| Property | Formula |
|----------|---------|
| **Scaling** | $\text{Var}(aX) = a^2 \text{Var}(X)$ |
| **Shift invariance** | $\text{Var}(X + b) = \text{Var}(X)$ |
| **Combined** | $\text{Var}(aX + b) = a^2 \text{Var}(X)$ |
| **Sum (independent)** | $\text{Var}(X + Y) = \text{Var}(X) + \text{Var}(Y)$ |
| **Non-negative** | $\text{Var}(X) \geq 0$ |

### 2.5 Standard Deviation Properties

| Property | Formula |
|----------|---------|
| **Scaling** | $SD(aX) = |a| \cdot SD(X)$ |
| **Shift invariance** | $SD(X + b) = SD(X)$ |

### 2.6 Supply Chain Application

**Retail Context**:

**Safety Stock Formula**:
$$SS = z \cdot \sigma_D \cdot \sqrt{L}$$

Where:
- $z$ = service level z-score
- $\sigma_D$ = standard deviation of demand
- $L$ = lead time

**Key insight**: Higher demand variability → more safety stock needed.

---

## 3. Coefficient of Variation (CV)

### 3.1 Theory

**Coefficient of Variation** normalizes standard deviation by the mean, enabling comparison across different scales.

### 3.2 Mathematical Definition

$$CV = \frac{\sigma}{\mu}$$

Often expressed as a percentage: $CV\% = \frac{\sigma}{\mu} \times 100\%$

### 3.3 Interpretation

| CV Value | Interpretation |
|----------|---------------|
| **Low CV** | Consistent, predictable |
| **High CV** | Variable, harder to forecast |

### 3.4 Why Use CV?

| Scenario | Why CV Helps |
|----------|--------------|
| **Different scales** | Compare variability of $100 revenue vs $10,000 revenue |
| **Different units** | Compare demand (units) vs lead time (days) |
| **SKU classification** | Identify high-variability products |

### 3.5 Supply Chain Application

**Retail Context**:
- **Compare SKUs**: Which products have most variable demand?
- **ABC-XYZ Analysis**: X (low CV), Y (medium CV), Z (high CV)
- **Forecast difficulty**: High CV → harder to forecast

---

## 4. Summary of Formulas

### 4.1 Key Formulas

| Measure | Symbol | Formula |
|---------|--------|---------|
| **Expected Value** | $\mu$ or $E[X]$ | $\sum x \cdot p(x)$ |
| **Variance** | $\sigma^2$ or $\text{Var}(X)$ | $E[X^2] - (E[X])^2$ |
| **Standard Deviation** | $\sigma$ | $\sqrt{\text{Var}(X)}$ |
| **Coefficient of Variation** | $CV$ | $\sigma / \mu$ |

### 4.2 Calculation Steps

1. Calculate $E[X] = \sum x \cdot p(x)$
2. Calculate $E[X^2] = \sum x^2 \cdot p(x)$
3. Calculate $\text{Var}(X) = E[X^2] - (E[X])^2$
4. Calculate $SD(X) = \sqrt{\text{Var}(X)}$
5. Calculate $CV = SD(X) / E[X]$

---

## Summary Table

| Concept | Definition | Formula | Supply Chain Application |
|---------|------------|---------|--------------------------|
| **Expected Value** | Probability-weighted average | $E[X] = \sum x \cdot p(x)$ | Average demand |
| **Variance** | Squared spread around mean | $\text{Var}(X) = E[X^2] - \mu^2$ | Demand variability |
| **Std Deviation** | Spread in original units | $\sigma = \sqrt{\text{Var}(X)}$ | Safety stock calculation |
| **CV** | Relative variability | $CV = \sigma / \mu$ | SKU comparison |

---

## Key Takeaways

1. **$E[X]$**: Probability-weighted average — the "center" of distribution
2. **$\text{Var}(X)$**: Expected squared deviation from mean — measures spread
3. **$\sigma$**: Standard deviation — spread in original units
4. **$CV$**: Relative variability — enables cross-scale comparison

---

## Next Week Preview

Week 11 covers **Binomial and Poisson Distributions** - key discrete distributions.

---

*IIT Madras BS Degree in Data Science*
