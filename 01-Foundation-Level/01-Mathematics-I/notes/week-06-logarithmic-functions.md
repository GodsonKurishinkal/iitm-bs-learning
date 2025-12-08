# Week 06: Logarithmic Functions

**Course**: BSMA1001 - Mathematics I
**Level**: Foundation

---

## 1. Logarithm Properties

### 1.1 Theory

Logarithms are inverses of exponential functions. They answer the question: "To what power must we raise the base to get this number?"

### 1.2 Mathematical Definition

$$\log_b(x) = y \iff b^y = x$$

Where:
- $b$ = base (must be positive, $b \neq 1$)
- $x$ = argument (must be positive)
- $y$ = logarithm (the exponent)

### 1.3 Common Logarithms

| Notation | Name | Base |
|----------|------|------|
| $\log(x)$ | Common logarithm | 10 |
| $\ln(x)$ | Natural logarithm | $e \approx 2.718$ |
| $\log_2(x)$ | Binary logarithm | 2 |

### 1.4 Fundamental Properties

| Property | Formula | Description |
|----------|---------|-------------|
| Product Rule | $\log_b(xy) = \log_b(x) + \log_b(y)$ | Log of product = sum of logs |
| Quotient Rule | $\log_b\left(\frac{x}{y}\right) = \log_b(x) - \log_b(y)$ | Log of quotient = difference of logs |
| Power Rule | $\log_b(x^n) = n \cdot \log_b(x)$ | Log of power = power times log |
| Change of Base | $\log_b(x) = \frac{\ln(x)}{\ln(b)} = \frac{\log(x)}{\log(b)}$ | Convert between bases |

### 1.5 Special Values

- $\log_b(1) = 0$ (because $b^0 = 1$)
- $\log_b(b) = 1$ (because $b^1 = b$)
- $\log_b(b^n) = n$
- $b^{\log_b(x)} = x$

### 1.6 Supply Chain Application

**Retail Context**: Logarithms appear in:
- Diminishing returns (log-response to marketing spend)
- Information theory (bits of information)
- pH-like scales for measuring phenomena spanning many orders of magnitude
- Pareto analysis and power law distributions

---

## 2. Graphs of Logarithmic Functions

### 2.1 Theory

Logarithmic graphs are reflections of exponential graphs across the line $y = x$. They grow slowly and have vertical asymptotes.

### 2.2 Key Characteristics

For $f(x) = \log_b(x)$:

| Property | Value |
|----------|-------|
| Domain | $(0, \infty)$ — only positive numbers |
| Range | $(-\infty, \infty)$ — all real numbers |
| Vertical Asymptote | $x = 0$ (y-axis) |
| x-intercept | $(1, 0)$ — always passes through this point |
| Behavior | Increases slowly for $b > 1$ |

### 2.3 Comparing Bases

- **Larger base** ($b > 1$): Graph rises more slowly
- **Smaller base** (closer to 1): Graph rises more quickly
- All logarithmic functions pass through $(1, 0)$

### 2.4 Transformations

For $f(x) = a \cdot \log_b(x - h) + k$:
- $h$ = horizontal shift (asymptote moves to $x = h$)
- $k$ = vertical shift
- $a$ = vertical stretch/compression (negative reflects over x-axis)

### 2.5 Supply Chain Application

**Retail Context**: Log scales help visualize data spanning many magnitudes:
- Comparing small boutique sales to warehouse volumes
- Visualizing SKUs with vastly different velocities
- Plotting demand across product categories with wide range

---

## 3. Exponential Equations

### 3.1 Theory

When the variable is in the exponent, we use logarithms to solve. Taking the log of both sides "brings down" the exponent.

### 3.2 Solving Method

To solve $b^x = c$:

$$x = \log_b(c) = \frac{\ln(c)}{\ln(b)}$$

### 3.3 Common Forms

**Simple exponential**:
$$b^x = c \implies x = \frac{\ln(c)}{\ln(b)}$$

**Compound growth**:
$$A = P(1 + r)^t \implies t = \frac{\ln(A/P)}{\ln(1+r)}$$

**Continuous growth**:
$$A = Pe^{kt} \implies t = \frac{\ln(A/P)}{k}$$

### 3.4 Examples

| Equation | Solution |
|----------|----------|
| $2^x = 8$ | $x = \log_2(8) = 3$ |
| $e^{2x} = 10$ | $x = \frac{\ln(10)}{2} \approx 1.15$ |
| $5^{x-1} = 25$ | $x - 1 = 2 \implies x = 3$ |

### 3.5 Supply Chain Application

**Retail Context**: Solving questions like:
- "When will inventory reach the reorder threshold?"
- "How long until we double our customer base?"
- "When will demand exceed capacity?"

---

## 4. Logarithmic Equations

### 4.1 Theory

Logarithmic equations have the variable inside the logarithm. We solve by converting to exponential form or using logarithm properties.

### 4.2 Solving Method

To solve $\log_b(x) = c$:

$$x = b^c$$

### 4.3 Strategies for Solving

1. **Convert to exponential form**: $\log_b(x) = c \implies x = b^c$

2. **Combine logs first**: Use properties to combine multiple logs, then convert

3. **Isolate the logarithm**: Get the log term alone before converting

### 4.4 Important: Domain Constraints

**Always verify solutions!** The argument of a logarithm must be positive.

For $\log_b(f(x)) = c$, check that $f(x) > 0$ for each solution.

### 4.5 Examples

| Equation | Solution | Check |
|----------|----------|-------|
| $\log_2(x) = 5$ | $x = 2^5 = 32$ | $32 > 0$ ✓ |
| $\ln(x-3) = 2$ | $x - 3 = e^2 \implies x = e^2 + 3 \approx 10.39$ | $10.39 - 3 > 0$ ✓ |
| $\log(x) + \log(x+3) = 1$ | $\log(x(x+3)) = 1 \implies x^2 + 3x = 10$ | Solve quadratic, check $x > 0$ |

### 4.6 Supply Chain Application

**Retail Context**: If marketing effectiveness follows:
$$\text{response} = a \cdot \log(\text{spend}) + b$$

We can solve for the required spend to achieve a target response level.

---

## Summary

| Concept | Key Formula |
|---------|-------------|
| Definition | $\log_b(x) = y \iff b^y = x$ |
| Product Rule | $\log_b(xy) = \log_b(x) + \log_b(y)$ |
| Quotient Rule | $\log_b(x/y) = \log_b(x) - \log_b(y)$ |
| Power Rule | $\log_b(x^n) = n \cdot \log_b(x)$ |
| Change of Base | $\log_b(x) = \frac{\ln(x)}{\ln(b)}$ |
| Solve $b^x = c$ | $x = \frac{\ln(c)}{\ln(b)}$ |
| Solve $\log_b(x) = c$ | $x = b^c$ |

## Key Takeaways

1. **Logarithms are inverses of exponentials**: $\log_b(x) = y \iff b^y = x$

2. **Key properties**: Product → sum, quotient → difference, power → multiply

3. **Logarithmic graphs** have vertical asymptote at $x = 0$ and pass through $(1, 0)$

4. **Exponential equations** are solved by taking logarithms of both sides

5. **Logarithmic equations** are solved by converting to exponential form

6. **Always check domain constraints** — logarithm arguments must be positive

---

## Next Week Preview

Week 7 covers **Sequences, Limits, and Continuity** — fundamental concepts for calculus and mathematical analysis.

---

*IIT Madras BS Degree in Data Science*
