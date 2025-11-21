# Week 11: Integration - Comprehensive Study Notes

---
**Date**: 2025-11-22
**Course**: BSMA1001 - Mathematics for Data Science I
**Level**: Foundation
**Week**: 11 of 12
**Source**: IIT Madras Mathematics I Week 11
**Topic Area**: Calculus - Integration
**Tags**: #BSMA1001 #Integration #Calculus #Week11 #Foundation
---

## Overview

Integration is the **reverse operation of differentiation** and one of the two fundamental operations in calculus. While derivatives measure instantaneous rates of change, integrals measure **accumulation** - the total quantity accumulated over an interval. In data science, integration appears in probability distributions, loss function analysis, signal processing, and Bayesian statistics.

**Key Concepts This Week:**
- Antiderivatives and indefinite integrals
- Definite integrals and the Riemann sum
- Fundamental Theorem of Calculus (FTC)
- Basic integration techniques (substitution, parts)
- Applications: area under curves, probability, expected values

**Why Integration Matters in Data Science:**
- **Probability**: Computing probabilities from probability density functions (PDFs)
- **Statistics**: Expected values and moments of distributions
- **Optimization**: Verifying minimum loss achieved (integral of gradient)
- **Signal Processing**: Fourier transforms integrate over frequencies
- **Machine Learning**: Normalizing constants in probabilistic models

---

## 1. Antiderivatives and Indefinite Integrals

### 1.1 Definition

An **antiderivative** of $f(x)$ is a function $F(x)$ such that:

$$F'(x) = f(x)$$

The **indefinite integral** represents the family of all antiderivatives:

$$\int f(x) \, dx = F(x) + C$$

where $C$ is the **constant of integration** (any constant differentiates to zero).

**Example:** If $f(x) = 2x$, then $F(x) = x^2 + C$ because $\frac{d}{dx}[x^2 + C] = 2x$.

### 1.2 Basic Integration Rules

| Function $f(x)$ | Integral $\int f(x) \, dx$ | Rule Name |
|-----------------|---------------------------|-----------|
| $k$ (constant) | $kx + C$ | Constant |
| $x^n$ (n ≠ -1) | $\frac{x^{n+1}}{n+1} + C$ | Power Rule |
| $\frac{1}{x}$ | $\ln\|x\| + C$ | Natural Log |
| $e^x$ | $e^x + C$ | Exponential |
| $\sin x$ | $-\cos x + C$ | Sine |
| $\cos x$ | $\sin x + C$ | Cosine |
| $\sec^2 x$ | $\tan x + C$ | Secant Squared |

**Properties:**
- **Linearity:** $\int [af(x) + bg(x)] \, dx = a\int f(x) \, dx + b\int g(x) \, dx$
- **Constant Multiple:** $\int kf(x) \, dx = k \int f(x) \, dx$

**Worked Example 1:** Find $\int (3x^2 - 4x + 5) \, dx$

**Solution:**
$$\int (3x^2 - 4x + 5) \, dx = 3\int x^2 \, dx - 4\int x \, dx + 5\int 1 \, dx$$
$$= 3 \cdot \frac{x^3}{3} - 4 \cdot \frac{x^2}{2} + 5x + C$$
$$= x^3 - 2x^2 + 5x + C$$

**Verification:** Differentiate to check:
$$\frac{d}{dx}[x^3 - 2x^2 + 5x + C] = 3x^2 - 4x + 5 \quad ✓$$

---

## 2. Definite Integrals

### 2.1 Definition via Riemann Sums

The **definite integral** from $a$ to $b$ represents the **signed area** between the curve $y = f(x)$ and the x-axis:

$$\int_a^b f(x) \, dx = \lim_{n \to \infty} \sum_{i=1}^{n} f(x_i^*) \Delta x$$

where:
- $\Delta x = \frac{b-a}{n}$ (width of each rectangle)
- $x_i^* \in [x_{i-1}, x_i]$ (sample point in $i$-th subinterval)

**Geometric Interpretation:**
- Area **above** x-axis: positive contribution
- Area **below** x-axis: negative contribution
- Net signed area = definite integral

### 2.2 Properties of Definite Integrals

1. **Reversal of Limits:** $\int_a^b f(x) \, dx = -\int_b^a f(x) \, dx$

2. **Zero-Width Interval:** $\int_a^a f(x) \, dx = 0$

3. **Additivity:** $\int_a^b f(x) \, dx + \int_b^c f(x) \, dx = \int_a^c f(x) \, dx$

4. **Linearity:** $\int_a^b [kf(x) + g(x)] \, dx = k\int_a^b f(x) \, dx + \int_a^b g(x) \, dx$

5. **Comparison:** If $f(x) \leq g(x)$ on $[a,b]$, then $\int_a^b f(x) \, dx \leq \int_a^b g(x) \, dx$

**Worked Example 2:** Approximate $\int_0^2 x^2 \, dx$ using 4 rectangles (right endpoints)

**Solution:**
- Interval: $[0, 2]$, $n = 4$, so $\Delta x = \frac{2-0}{4} = 0.5$
- Subintervals: $[0, 0.5], [0.5, 1], [1, 1.5], [1.5, 2]$
- Right endpoints: $x_1 = 0.5, x_2 = 1, x_3 = 1.5, x_4 = 2$

Riemann sum:
$$\sum_{i=1}^{4} f(x_i) \Delta x = [f(0.5) + f(1) + f(1.5) + f(2)] \cdot 0.5$$
$$= [0.25 + 1 + 2.25 + 4] \cdot 0.5 = 7.5 \cdot 0.5 = 3.75$$

**True value:** $\int_0^2 x^2 \, dx = \left[\frac{x^3}{3}\right]_0^2 = \frac{8}{3} \approx 2.667$

The approximation overestimates because $f(x) = x^2$ is increasing.

---

## 3. Fundamental Theorem of Calculus (FTC)

The FTC establishes the profound connection between differentiation and integration.

### 3.1 FTC Part 1 (Derivative of Integral)

If $f$ is continuous on $[a, b]$ and $F(x) = \int_a^x f(t) \, dt$, then:

$$F'(x) = f(x)$$

**Interpretation:** The derivative of an accumulation function is the original function.

**Example:** Let $F(x) = \int_0^x t^2 \, dt$. Find $F'(x)$.

**Solution:** By FTC Part 1, $F'(x) = x^2$.

### 3.2 FTC Part 2 (Evaluation Theorem)

If $f$ is continuous on $[a, b]$ and $F$ is any antiderivative of $f$, then:

$$\int_a^b f(x) \, dx = F(b) - F(a) = \left[F(x)\right]_a^b$$

**Interpretation:** To compute a definite integral, find any antiderivative and evaluate at boundaries.

**Worked Example 3:** Evaluate $\int_1^3 (2x + 1) \, dx$

**Solution:**
1. Find antiderivative: $F(x) = x^2 + x + C$
2. Apply FTC Part 2:
$$\int_1^3 (2x + 1) \, dx = [x^2 + x]_1^3 = (9 + 3) - (1 + 1) = 12 - 2 = 10$$

Note: The constant $C$ cancels when computing $F(b) - F(a)$.

---

## 4. Integration Techniques

### 4.1 Substitution Method (u-substitution)

**Idea:** Reverse the chain rule. If $u = g(x)$, then $du = g'(x) \, dx$.

**Formula:** $\int f(g(x)) g'(x) \, dx = \int f(u) \, du$

**Strategy:**
1. Identify inner function $u = g(x)$
2. Compute $du = g'(x) \, dx$
3. Rewrite integral in terms of $u$
4. Integrate with respect to $u$
5. Substitute back $x$

**Worked Example 4:** Evaluate $\int 2x \cos(x^2) \, dx$

**Solution:**
1. Let $u = x^2$, then $du = 2x \, dx$
2. Rewrite: $\int \cos(u) \, du$
3. Integrate: $\sin(u) + C$
4. Substitute back: $\sin(x^2) + C$

**Verification:** $\frac{d}{dx}[\sin(x^2)] = \cos(x^2) \cdot 2x$ ✓

### 4.2 Definite Integrals with Substitution

**Important:** When using substitution for definite integrals, change the limits!

**Worked Example 5:** Evaluate $\int_0^1 x e^{x^2} \, dx$

**Solution:**
1. Let $u = x^2$, then $du = 2x \, dx$, so $x \, dx = \frac{1}{2} du$
2. Change limits:
   - When $x = 0$: $u = 0^2 = 0$
   - When $x = 1$: $u = 1^2 = 1$
3. Rewrite:
$$\int_0^1 x e^{x^2} \, dx = \int_0^1 e^u \cdot \frac{1}{2} \, du = \frac{1}{2} \int_0^1 e^u \, du$$
4. Evaluate:
$$= \frac{1}{2} [e^u]_0^1 = \frac{1}{2}(e^1 - e^0) = \frac{1}{2}(e - 1) \approx 0.859$$

### 4.3 Integration by Parts

**Formula:** $\int u \, dv = uv - \int v \, du$

**Strategy:** Choose $u$ and $dv$ such that $\int v \, du$ is simpler than the original integral.

**LIATE rule** for choosing $u$ (in order of priority):
- **L**ogarithmic: $\ln x$
- **I**nverse trig: $\arcsin x, \arctan x$
- **A**lgebraic: $x^n$
- **T**rigonometric: $\sin x, \cos x$
- **E**xponential: $e^x$

**Worked Example 6:** Evaluate $\int x e^x \, dx$

**Solution:**
1. Choose: $u = x$ (algebraic), $dv = e^x \, dx$
2. Then: $du = dx$, $v = e^x$
3. Apply formula:
$$\int x e^x \, dx = x e^x - \int e^x \, dx = x e^x - e^x + C = e^x(x - 1) + C$$

---

## 5. Applications of Integration

### 5.1 Area Under a Curve

The area between $y = f(x)$ and the x-axis from $x = a$ to $x = b$ is:

$$\text{Area} = \int_a^b |f(x)| \, dx$$

If $f(x) \geq 0$ on $[a, b]$, then Area $= \int_a^b f(x) \, dx$.

**Example:** Find the area under $y = x^2$ from $x = 0$ to $x = 3$.

$$\text{Area} = \int_0^3 x^2 \, dx = \left[\frac{x^3}{3}\right]_0^3 = \frac{27}{3} - 0 = 9 \text{ square units}$$

### 5.2 Area Between Two Curves

The area between $y = f(x)$ and $y = g(x)$ from $x = a$ to $x = b$ (with $f(x) \geq g(x)$):

$$\text{Area} = \int_a^b [f(x) - g(x)] \, dx$$

**Worked Example 7:** Find area between $y = x^2$ and $y = x$ from $x = 0$ to $x = 1$.

**Solution:**
1. Determine which is on top: At $x = 0.5$, $x = 0.5 > x^2 = 0.25$, so $y = x$ is above
2. Compute:
$$\text{Area} = \int_0^1 (x - x^2) \, dx = \left[\frac{x^2}{2} - \frac{x^3}{3}\right]_0^1$$
$$= \left(\frac{1}{2} - \frac{1}{3}\right) - 0 = \frac{3 - 2}{6} = \frac{1}{6} \text{ square units}$$

### 5.3 Average Value of a Function

The **average value** of $f$ on $[a, b]$ is:

$$f_{\text{avg}} = \frac{1}{b-a} \int_a^b f(x) \, dx$$

**Interpretation:** The constant height that gives the same area as the variable function.

**Example:** Find average value of $f(x) = x^2$ on $[0, 2]$.

$$f_{\text{avg}} = \frac{1}{2-0} \int_0^2 x^2 \, dx = \frac{1}{2} \cdot \frac{8}{3} = \frac{4}{3}$$

---

## 6. Data Science Applications

### 6.1 Probability Density Functions (PDFs)

A **probability density function** $f(x)$ satisfies:
1. $f(x) \geq 0$ for all $x$
2. $\int_{-\infty}^{\infty} f(x) \, dx = 1$ (normalization)

The probability that a random variable $X$ falls in $[a, b]$ is:

$$P(a \leq X \leq b) = \int_a^b f(x) \, dx$$

**Example:** Uniform distribution on $[0, 1]$: $f(x) = 1$ for $0 \leq x \leq 1$.

$$P(0.2 \leq X \leq 0.5) = \int_{0.2}^{0.5} 1 \, dx = 0.5 - 0.2 = 0.3$$

### 6.2 Expected Value (Mean)

The **expected value** of a continuous random variable $X$ with PDF $f(x)$:

$$E[X] = \int_{-\infty}^{\infty} x f(x) \, dx$$

**Example:** For uniform distribution on $[0, 1]$:

$$E[X] = \int_0^1 x \cdot 1 \, dx = \left[\frac{x^2}{2}\right]_0^1 = \frac{1}{2}$$

This confirms the mean of uniform $[0, 1]$ is 0.5.

### 6.3 Loss Functions and Optimization

In training neural networks, the **total loss** over a dataset can be viewed as a continuous sum (integral in limit):

$$\text{Total Loss} = \int f(x; \theta) \, dx$$

where $\theta$ are model parameters. Minimizing loss requires:

$$\frac{d}{d\theta} \int f(x; \theta) \, dx = 0$$

By FTC and Leibniz rule, we can differentiate under the integral.

### 6.4 Cumulative Distribution Functions (CDFs)

The **CDF** of a random variable $X$ with PDF $f(x)$ is:

$$F(x) = P(X \leq x) = \int_{-\infty}^x f(t) \, dt$$

**Key property:** $F'(x) = f(x)$ (by FTC Part 1).

**Example:** For exponential distribution $f(x) = \lambda e^{-\lambda x}$ (x ≥ 0):

$$F(x) = \int_0^x \lambda e^{-\lambda t} \, dt = [-e^{-\lambda t}]_0^x = 1 - e^{-\lambda x}$$

### 6.5 Information Theory: Entropy

**Differential entropy** for continuous random variable with PDF $f(x)$:

$$H(X) = -\int_{-\infty}^{\infty} f(x) \ln f(x) \, dx$$

This measures uncertainty in the distribution and is computed via integration.

---

## 7. Numerical Integration

When antiderivatives are hard or impossible to find analytically, use **numerical integration**.

### 7.1 Trapezoidal Rule

Approximate $\int_a^b f(x) \, dx$ using trapezoids:

$$\int_a^b f(x) \, dx \approx \frac{\Delta x}{2} [f(x_0) + 2f(x_1) + 2f(x_2) + \cdots + 2f(x_{n-1}) + f(x_n)]$$

where $\Delta x = \frac{b-a}{n}$ and $x_i = a + i\Delta x$.

### 7.2 Simpson's Rule

Uses parabolic approximations (requires even $n$):

$$\int_a^b f(x) \, dx \approx \frac{\Delta x}{3} [f(x_0) + 4f(x_1) + 2f(x_2) + 4f(x_3) + \cdots + 4f(x_{n-1}) + f(x_n)]$$

**Pattern:** Alternating coefficients 1, 4, 2, 4, 2, ..., 4, 1

### 7.3 Monte Carlo Integration

For high-dimensional integrals, use random sampling:

$$\int_a^b f(x) \, dx \approx (b-a) \cdot \frac{1}{N} \sum_{i=1}^{N} f(x_i)$$

where $x_i$ are random samples from $[a, b]$.

**Application:** Computing normalizing constants in Bayesian statistics.

---

## 8. Common Pitfalls and Mistakes

**Pitfall 1: Forgetting Constant of Integration**
- ❌ Wrong: $\int x \, dx = \frac{x^2}{2}$
- ✅ Correct: $\int x \, dx = \frac{x^2}{2} + C$

**Pitfall 2: Not Changing Limits in Substitution**
- When using $u$-substitution for definite integrals, transform limits from $x$ to $u$.

**Pitfall 3: Ignoring Absolute Value in $\ln$**
- ❌ Wrong: $\int \frac{1}{x} \, dx = \ln x + C$
- ✅ Correct: $\int \frac{1}{x} \, dx = \ln|x| + C$

**Pitfall 4: Misapplying Power Rule at $n = -1$**
- The power rule $\frac{x^{n+1}}{n+1}$ fails when $n = -1$ (division by zero).
- Use $\int x^{-1} \, dx = \ln|x| + C$ instead.

**Pitfall 5: Forgetting Negative Sign**
- $\int \sin x \, dx = -\cos x + C$ (note the minus sign!)

---

## 9. Complex Worked Examples

### Example 1: Integration by Substitution (Definite Integral)

**Problem:** Evaluate $\int_0^{\pi/2} \sin x \cos x \, dx$

**Solution:**
**Method 1:** Let $u = \sin x$, then $du = \cos x \, dx$

Limits: When $x = 0$, $u = \sin 0 = 0$; when $x = \pi/2$, $u = \sin(\pi/2) = 1$

$$\int_0^{\pi/2} \sin x \cos x \, dx = \int_0^1 u \, du = \left[\frac{u^2}{2}\right]_0^1 = \frac{1}{2} - 0 = \frac{1}{2}$$

**Method 2:** Use identity $\sin x \cos x = \frac{1}{2}\sin(2x)$

$$\int_0^{\pi/2} \sin x \cos x \, dx = \int_0^{\pi/2} \frac{1}{2}\sin(2x) \, dx$$

Let $v = 2x$, $dv = 2dx$:
$$= \frac{1}{2} \cdot \frac{1}{2} \int_0^{\pi} \sin v \, dv = \frac{1}{4}[-\cos v]_0^{\pi}$$
$$= \frac{1}{4}[-\cos \pi + \cos 0] = \frac{1}{4}[1 + 1] = \frac{1}{2}$$

Both methods agree! ✓

### Example 2: Integration by Parts

**Problem:** Evaluate $\int x^2 e^x \, dx$

**Solution:** Apply integration by parts twice.

**First application:**
- $u = x^2$, $dv = e^x dx$
- $du = 2x dx$, $v = e^x$

$$\int x^2 e^x \, dx = x^2 e^x - \int 2x e^x \, dx$$

**Second application on** $\int 2x e^x \, dx$:
- $u = 2x$, $dv = e^x dx$
- $du = 2dx$, $v = e^x$

$$\int 2x e^x \, dx = 2x e^x - \int 2e^x \, dx = 2x e^x - 2e^x$$

**Combine:**
$$\int x^2 e^x \, dx = x^2 e^x - (2x e^x - 2e^x) = x^2 e^x - 2x e^x + 2e^x + C$$
$$= e^x(x^2 - 2x + 2) + C$$

### Example 3: Area Between Curves

**Problem:** Find area enclosed by $y = x^2 - 4$ and $y = -x^2 + 2x + 2$

**Solution:**
1. Find intersection points: $x^2 - 4 = -x^2 + 2x + 2$
   $$2x^2 - 2x - 6 = 0 \implies x^2 - x - 3 = 0$$
   $$x = \frac{1 \pm \sqrt{1 + 12}}{2} = \frac{1 \pm \sqrt{13}}{2}$$

   Let $a = \frac{1 - \sqrt{13}}{2} \approx -1.303$ and $b = \frac{1 + \sqrt{13}}{2} \approx 2.303$

2. Determine which is on top: At $x = 0$, we have $-4$ vs $2$, so $-x^2 + 2x + 2$ is above.

3. Compute area:
$$\text{Area} = \int_a^b [(-x^2 + 2x + 2) - (x^2 - 4)] \, dx$$
$$= \int_a^b (-2x^2 + 2x + 6) \, dx$$
$$= \left[-\frac{2x^3}{3} + x^2 + 6x\right]_a^b$$

Evaluating at endpoints (using $a, b$ satisfy $x^2 = x + 3$):
$$= -\frac{2b^3}{3} + b^2 + 6b + \frac{2a^3}{3} - a^2 - 6a$$

(Exact evaluation requires careful algebra or numerical methods.)

**Numerical result:** Area $\approx 15.59$ square units.

---

## 10. Practice Problems

### Basic Problems (5)

1. Find $\int (4x^3 - 3x^2 + 2x - 1) \, dx$

2. Evaluate $\int_0^2 (x^2 + 1) \, dx$

3. Use substitution to find $\int 2x(x^2 + 1)^5 \, dx$

4. Find the derivative of $F(x) = \int_0^x t^3 \, dt$ (use FTC Part 1)

5. Compute the average value of $f(x) = \sin x$ on $[0, \pi]$

### Intermediate Problems (5)

6. Evaluate $\int_1^e \frac{\ln x}{x} \, dx$ (use substitution)

7. Use integration by parts to find $\int x \sin x \, dx$

8. Find the area between $y = x^2$ and $y = 2x$ from $x = 0$ to $x = 2$

9. Evaluate $\int \frac{1}{1 + x^2} \, dx$ (recall $\frac{d}{dx}[\arctan x] = \frac{1}{1+x^2}$)

10. A probability density function is $f(x) = 2x$ for $0 \leq x \leq 1$. Find $P(0.5 \leq X \leq 1)$.

### Advanced Problems (5)

11. Use integration by parts twice to evaluate $\int e^x \sin x \, dx$

12. Find area enclosed between $y = e^x$, $y = e^{-x}$, and $x = 1$

13. Evaluate $\int_0^{\pi/4} \tan^2 x \, dx$ (use $\tan^2 x = \sec^2 x - 1$)

14. For exponential distribution $f(x) = \lambda e^{-\lambda x}$ (x ≥ 0), compute $E[X] = \int_0^{\infty} x \lambda e^{-\lambda x} \, dx$ (use parts)

15. Approximate $\int_0^1 e^{-x^2} \, dx$ using Trapezoidal rule with $n = 4$ subintervals (this integral has no elementary antiderivative!)

---

## 11. Summary

**Core Concepts Mastered:**
- Integration as the reverse of differentiation (antiderivatives)
- Definite integrals as limits of Riemann sums (accumulated area)
- Fundamental Theorem of Calculus linking derivatives and integrals
- Integration techniques: substitution, parts
- Applications: area, probability, expected values

**Integration in Data Science:**
- **Probability:** Computing probabilities from PDFs
- **Statistics:** Expected values, variance, moments
- **Bayesian Inference:** Normalizing constants
- **Information Theory:** Entropy calculations
- **Signal Processing:** Fourier transforms

**Key Formulas:**
- Power rule: $\int x^n \, dx = \frac{x^{n+1}}{n+1} + C$ (n ≠ -1)
- FTC Part 2: $\int_a^b f(x) \, dx = F(b) - F(a)$
- Substitution: $\int f(g(x))g'(x) \, dx = \int f(u) \, du$ where $u = g(x)$
- Integration by parts: $\int u \, dv = uv - \int v \, du$

---

## 12. Looking Ahead: Week 12

Next week, we synthesize Weeks 4-11 with **comprehensive applications**:
- Multivariate calculus preview (partial derivatives, gradients)
- Optimization in machine learning (gradient descent, Newton's method)
- Differential equations basics (exponential growth/decay models)
- Real-world case studies combining all topics

**Connection:** Integration enables us to solve differential equations, model cumulative effects, and understand probability distributions—all fundamental to advanced data science.

---

## References

1. Stewart, J. (2015). *Calculus: Early Transcendentals* (8th ed.). Cengage Learning.
2. Thomas, G. B., et al. (2018). *Thomas' Calculus* (14th ed.). Pearson.
3. Strang, G. (2016). *Calculus* (3rd ed.). Wellesley-Cambridge Press.
4. IIT Madras BS in Data Science - BSMA1001 Course Materials

---

**End of Week 11 Notes**
