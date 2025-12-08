# Week 04: Continuous Joint Distributions

**Course**: BSST1002 - Statistics II
**Level**: Foundation

---

## Learning Objectives
- Master the bivariate Normal distribution
- Understand conditional distributions for continuous variables
- Apply correlation in multivariate settings

---

## 1. Bivariate Normal Distribution

### 1.1 Theory

The **bivariate Normal** extends the Normal distribution to two dimensions with correlation. It is fundamental for multivariate analysis.

### 1.2 Notation

$$(X, Y) \sim N(\mu_X, \mu_Y, \sigma_X^2, \sigma_Y^2, \rho)$$

Where:
- $\mu_X, \mu_Y$ = means
- $\sigma_X^2, \sigma_Y^2$ = variances
- $\rho$ = correlation coefficient

### 1.3 Properties

| Property | Formula |
|----------|---------|
| **5 Parameters** | $\mu_X, \mu_Y, \sigma_X, \sigma_Y, \rho$ |
| **Correlation** | $-1 \leq \rho \leq 1$ |
| **Contours** | Ellipses centered at $(\mu_X, \mu_Y)$ |

### 1.4 Marginal Distributions

The marginals of a bivariate Normal are **univariate Normal**:

$$X \sim N(\mu_X, \sigma_X^2)$$
$$Y \sim N(\mu_Y, \sigma_Y^2)$$

**Important**: Marginals ignore the correlation!

### 1.5 Conditional Distribution

The conditional distribution of $Y$ given $X = x$ is also Normal:

$$Y | X = x \sim N\left(\mu_{Y|X}, \sigma_{Y|X}^2\right)$$

Where:

| Parameter | Formula |
|-----------|---------|
| **Conditional Mean** | $\mu_{Y|X} = \mu_Y + \rho \frac{\sigma_Y}{\sigma_X}(x - \mu_X)$ |
| **Conditional Variance** | $\sigma_{Y|X}^2 = \sigma_Y^2(1 - \rho^2)$ |

### 1.6 Key Insights from Conditional Distribution

| Observation | Interpretation |
|-------------|----------------|
| Conditional mean is linear in $x$ | Regression line |
| $\rho > 0$: higher $x$ → higher $E[Y\|x]$ | Positive relationship |
| $\rho < 0$: higher $x$ → lower $E[Y\|x]$ | Negative relationship |
| Conditional variance < marginal variance | Knowing $X$ reduces uncertainty in $Y$ |
| $\rho = 0$: conditional = marginal | Independence |

### 1.7 Variance Reduction

$$\sigma_{Y|X}^2 = \sigma_Y^2(1 - \rho^2)$$

| $\|\rho\|$ | Variance Reduction |
|-----------|-------------------|
| 0 | 0% (no reduction) |
| 0.5 | 25% |
| 0.7 | 51% |
| 0.9 | 81% |
| 1.0 | 100% (perfect prediction) |

### 1.8 Supply Chain Application

**Retail Context**:
- **Correlated demands** for substitute products (Coke vs Pepsi)
- **Lead time and demand** joint distribution
- **Price-quantity relationship** modeling

---

## 2. Conditional Distributions (General Continuous)

### 2.1 Theory

**Conditional distributions** update our beliefs when we observe one variable.

### 2.2 Mathematical Definition

$$f_{Y|X}(y|x) = \frac{f_{X,Y}(x,y)}{f_X(x)}$$

Where:
- $f_{X,Y}(x,y)$ = joint PDF
- $f_X(x)$ = marginal PDF of $X$

### 2.3 Properties

| Property | Description |
|----------|-------------|
| Valid PDF | $\int f_{Y|X}(y|x) \, dy = 1$ |
| Conditional expectation | $E[Y|X=x] = \int y \cdot f_{Y|X}(y|x) \, dy$ |
| Conditional variance | $\text{Var}(Y|X=x) = E[Y^2|X=x] - (E[Y|X=x])^2$ |

### 2.4 Supply Chain Application

**Retail Context**:
- **Expected demand given observed lead time**
- **Forecast update** given partial period sales
- **Conditional safety stock** based on known factors

---

## 3. Independence in Continuous Case

### 3.1 Definition

$X$ and $Y$ are independent if:

$$f_{X,Y}(x,y) = f_X(x) \cdot f_Y(y) \quad \forall x, y$$

### 3.2 For Bivariate Normal

$X$ and $Y$ are independent if and only if:

$$\rho = 0$$

**Special property**: For bivariate Normal, zero correlation implies independence (not true in general!).

---

## 4. Bivariate Normal in Practice

### 4.1 Generating Correlated Normal Variables

Given independent $Z_1, Z_2 \sim N(0,1)$:

$$X = \mu_X + \sigma_X Z_1$$
$$Y = \mu_Y + \sigma_Y (\rho Z_1 + \sqrt{1-\rho^2} Z_2)$$

This produces $(X, Y)$ with correlation $\rho$.

### 4.2 Python Implementation

```python
from scipy import stats
import numpy as np

# Define bivariate Normal
mean = [mu_X, mu_Y]
cov = [[sigma_X**2, rho*sigma_X*sigma_Y],
       [rho*sigma_X*sigma_Y, sigma_Y**2]]

# Generate samples
samples = np.random.multivariate_normal(mean, cov, size=1000)
```

---

## Summary Table

| Concept | Formula | Key Property |
|---------|---------|--------------|
| **Bivariate Normal** | $(X,Y) \sim N(\mu_X, \mu_Y, \sigma_X^2, \sigma_Y^2, \rho)$ | 5 parameters |
| **Marginal** | $X \sim N(\mu_X, \sigma_X^2)$ | Ignores correlation |
| **Conditional Mean** | $\mu_Y + \rho\frac{\sigma_Y}{\sigma_X}(x - \mu_X)$ | Linear in $x$ |
| **Conditional Variance** | $\sigma_Y^2(1-\rho^2)$ | Reduced by correlation |
| **Independence** | $\rho = 0$ | For bivariate Normal only |

---

## Key Takeaways

1. **Bivariate Normal**: Two correlated Normal variables with elliptical contours
2. **Marginals are Normal**: Each variable individually is Normal
3. **Conditionals are Normal**: With updated mean and reduced variance
4. **Correlation affects** joint behavior and conditional predictions

---

## Next Week Preview

Week 5 covers **Limit Theorems** - Law of Large Numbers and Central Limit Theorem.

---

*IIT Madras BS Degree in Data Science*
