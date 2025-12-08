# Week 10: Simple Regression

**Course**: BSST1002 - Statistics II
**Level**: Foundation Level

## Learning Objectives
- Understand the simple linear regression model and its components
- Estimate regression coefficients using the least squares method
- Interpret coefficients and R² (coefficient of determination)
- Perform residual diagnostics to validate model assumptions

---

## 1. The Linear Regression Model

### 1.1 Theory

**Simple linear regression** models the relationship between a predictor variable X (independent) and a response variable Y (dependent) as a straight line.

**Goal**: Find the best-fitting line that describes how Y changes as X changes.

### 1.2 Mathematical Definition

**Population Model**:

$$Y = \beta_0 + \beta_1 X + \epsilon$$

Where:
- $Y$ = response (dependent) variable
- $X$ = predictor (independent) variable
- $\beta_0$ = intercept (value of Y when X = 0)
- $\beta_1$ = slope (change in Y for one-unit increase in X)
- $\epsilon$ = error term (random noise), assumed $\epsilon \sim N(0, \sigma^2)$

**Fitted Model**:

$$\hat{Y} = \hat{\beta}_0 + \hat{\beta}_1 X$$

### 1.3 Least Squares Estimation

The **Ordinary Least Squares (OLS)** method minimizes the sum of squared residuals:

$$\text{Minimize: } \sum_{i=1}^{n}(y_i - \hat{y}_i)^2 = \sum_{i=1}^{n}(y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2$$

**Coefficient Estimates**:

$$\hat{\beta}_1 = \frac{\sum_{i=1}^{n}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n}(x_i - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}$$

$$\hat{\beta}_0 = \bar{y} - \hat{\beta}_1 \bar{x}$$

**Alternative Formula for Slope**:

$$\hat{\beta}_1 = r \cdot \frac{s_y}{s_x}$$

Where $r$ is the Pearson correlation coefficient.

### 1.4 Supply Chain Application

**Retail Context**:
- **Price-Demand Relationship**: How quantity demanded changes with price
- **Advertising-Sales Relationship**: Impact of ad spend on revenue
- **Store Size-Revenue Relationship**: How floor area affects sales
- **Lead Time-Order Size**: Relationship between order quantity and delivery time

---

## 2. Coefficient Interpretation and R²

### 2.1 Interpreting Coefficients

**Slope ($\hat{\beta}_1$)**:
- The expected change in Y for a **one-unit increase** in X
- Positive slope: Y increases as X increases
- Negative slope: Y decreases as X increases

**Intercept ($\hat{\beta}_0$)**:
- The expected value of Y when X = 0
- May not have practical meaning if X = 0 is outside the data range

### 2.2 Coefficient of Determination (R²)

**R²** measures the proportion of variance in Y explained by the model:

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum(y_i - \hat{y}_i)^2}{\sum(y_i - \bar{y})^2}$$

Where:
- $SS_{res}$ = Residual Sum of Squares (unexplained variance)
- $SS_{tot}$ = Total Sum of Squares (total variance)

**Properties of R²**:
- Range: $0 \leq R^2 \leq 1$
- $R^2 = 0$: Model explains none of the variance
- $R^2 = 1$: Model explains all of the variance (perfect fit)
- For simple regression: $R^2 = r^2$ (squared correlation)

### 2.3 Interpretation Guidelines

| R² Value | Interpretation |
|----------|----------------|
| 0.00 - 0.25 | Weak relationship |
| 0.25 - 0.50 | Moderate relationship |
| 0.50 - 0.75 | Strong relationship |
| 0.75 - 1.00 | Very strong relationship |

> **Caution**: High R² doesn't imply causation, and what's "good" depends on the field.

---

## 3. Predictions and Residual Diagnostics

### 3.1 Making Predictions

**Point Prediction**: Substitute new X value into fitted equation:

$$\hat{Y}_{new} = \hat{\beta}_0 + \hat{\beta}_1 X_{new}$$

**Caution - Extrapolation**: Predictions outside the range of observed X values are unreliable.

### 3.2 Residuals

**Residual**: The difference between observed and predicted values:

$$e_i = y_i - \hat{y}_i$$

**Properties**:
- Sum of residuals = 0: $\sum e_i = 0$
- Residuals are uncorrelated with X: $\sum e_i x_i = 0$

### 3.3 Assumptions of Linear Regression

| Assumption | Description | Diagnostic |
|------------|-------------|------------|
| **Linearity** | Relationship between X and Y is linear | Residual vs. fitted plot |
| **Independence** | Observations are independent | Study design, Durbin-Watson test |
| **Homoscedasticity** | Constant variance of residuals | Residual vs. fitted plot |
| **Normality** | Residuals are normally distributed | Q-Q plot, histogram of residuals |

### 3.4 Residual Plots

**Residuals vs. Fitted Values**:
- Should show random scatter around zero
- Patterns indicate violated assumptions:
  - Curved pattern → non-linearity
  - Funnel shape → heteroscedasticity

**Normal Q-Q Plot**:
- Points should fall along the diagonal line
- Deviations indicate non-normality

### 3.5 Standard Error of the Estimate

$$s_e = \sqrt{\frac{\sum(y_i - \hat{y}_i)^2}{n-2}} = \sqrt{\frac{SS_{res}}{n-2}}$$

Measures the typical size of prediction errors.

---

## 4. Inference for Regression Coefficients

### 4.1 Hypothesis Testing for Slope

**Testing if there's a significant relationship**:

$$H_0: \beta_1 = 0 \quad \text{(no linear relationship)}$$
$$H_1: \beta_1 \neq 0 \quad \text{(linear relationship exists)}$$

**Test Statistic**:

$$t = \frac{\hat{\beta}_1}{SE(\hat{\beta}_1)}$$

With $df = n - 2$

### 4.2 Confidence Interval for Slope

$$\hat{\beta}_1 \pm t_{\alpha/2, n-2} \cdot SE(\hat{\beta}_1)$$

---

## Summary

| Concept | Formula | Interpretation |
|---------|---------|----------------|
| Regression Model | $Y = \beta_0 + \beta_1 X + \epsilon$ | Linear relationship with error |
| Slope Estimate | $\hat{\beta}_1 = \frac{S_{xy}}{S_{xx}}$ | Change in Y per unit X |
| Intercept Estimate | $\hat{\beta}_0 = \bar{y} - \hat{\beta}_1\bar{x}$ | Y when X = 0 |
| R² | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained |
| Residual | $e_i = y_i - \hat{y}_i$ | Prediction error |

## Key Takeaways
- Simple regression models linear relationship: Y = β₀ + β₁X + ε
- The slope β₁ represents the change in Y for a one-unit increase in X
- R² measures how well the model explains variability in Y (0 to 1)
- Always check assumptions through residual analysis before trusting results

## Next Week Preview
Week 11 covers **Multiple Regression** - extending to multiple predictor variables.

---
*IIT Madras BS Degree in Data Science*
