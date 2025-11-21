# Week 9: Limits and Continuity

---
**Date**: 2025-11-21
**Course**: BSMA1001 - Mathematics for Data Science I
**Level**: Foundation
**Week**: 9 of 12
**Source**: IIT Madras Mathematics I Week 9
**Topic Area**: Calculus - Limits and Continuity
**Tags**: #BSMA1001 #Limits #Continuity #Week9 #Foundation #Calculus
---

## Overview

**Limits and continuity** form the theoretical foundation of calculus. While previous weeks explored polynomial operations, functions, and sequences, this week introduces the precise mathematical framework for understanding how functions behave "near" points and at infinity.

**Why this matters for Data Science:**
- **Gradient Descent**: Optimization algorithms rely on continuous, differentiable functions
- **Loss Functions**: Understanding continuity ensures well-behaved optimization landscapes
- **Convergence Analysis**: Neural networks and iterative algorithms depend on limit theory
- **Numerical Methods**: Root finding and approximation algorithms use continuity theorems
- **Probability**: Continuous probability distributions require understanding limits

**Week 9 Learning Objectives:**
1. Understand the formal ε-δ definition of limits
2. Compute limits using algebraic techniques
3. Recognize and evaluate one-sided limits
4. Work with infinite limits and limits at infinity
5. Define and test continuity at points and on intervals
6. Classify types of discontinuities
7. Apply the Intermediate Value Theorem
8. Use limits in optimization and algorithm analysis

---

## 1. Intuitive Understanding of Limits

Before formal definitions, consider the intuitive idea: **As x approaches a value, what does f(x) approach?**

**Notation:** $\lim_{x \to a} f(x) = L$

**Interpretation:** As $x$ gets arbitrarily close to $a$ (but not equal to $a$), $f(x)$ gets arbitrarily close to $L$.

**Key Insight:** The limit depends on behavior *near* $a$, not necessarily *at* $a$.

### Example 1.1: Visual Limit Understanding

Consider $f(x) = \frac{x^2 - 4}{x - 2}$ at $x = 2$:

- At $x = 2$: $f(2) = \frac{0}{0}$ (undefined)
- Near $x = 2$: $f(x) = \frac{(x-2)(x+2)}{x-2} = x + 2$ (for $x \neq 2$)
- As $x \to 2$: $f(x) \to 4$

Therefore: $\lim_{x \to 2} \frac{x^2 - 4}{x - 2} = 4$

**Data Science Connection:** Similar situations arise when computing gradients numerically - we evaluate derivatives using limits of difference quotients.

---

## 2. Formal Definition of Limits (ε-δ Definition)

**Definition:** $\lim_{x \to a} f(x) = L$ means:

For every $\varepsilon > 0$, there exists $\delta > 0$ such that:
$$0 < |x - a| < \delta \implies |f(x) - L| < \varepsilon$$

**Translation:**
- $\varepsilon$ (epsilon): How close we want $f(x)$ to be to $L$
- $\delta$ (delta): How close $x$ needs to be to $a$
- The condition $0 < |x - a|$ excludes the point $x = a$ itself

**Visual Interpretation:**
- Choose any "tolerance" $\varepsilon$ around $L$
- Find a corresponding "neighborhood" $\delta$ around $a$
- All points in the $\delta$-neighborhood (except $a$) map to the $\varepsilon$-neighborhood of $L$

### Example 2.1: ε-δ Proof

**Prove:** $\lim_{x \to 3} (2x + 1) = 7$

**Proof:**
Given $\varepsilon > 0$, we need to find $\delta > 0$ such that:
$$0 < |x - 3| < \delta \implies |(2x + 1) - 7| < \varepsilon$$

Working backwards:
$$|(2x + 1) - 7| = |2x - 6| = 2|x - 3|$$

For this to be less than $\varepsilon$:
$$2|x - 3| < \varepsilon$$
$$|x - 3| < \frac{\varepsilon}{2}$$

**Choice:** Let $\delta = \frac{\varepsilon}{2}$

**Verification:** If $0 < |x - 3| < \delta = \frac{\varepsilon}{2}$, then:
$$|(2x + 1) - 7| = 2|x - 3| < 2 \cdot \frac{\varepsilon}{2} = \varepsilon$$ ∎

---

## 3. Limit Laws and Properties

Limits can be computed algebraically using these fundamental properties:

**Theorem (Limit Laws):** If $\lim_{x \to a} f(x) = L$ and $\lim_{x \to a} g(x) = M$, then:

1. **Sum Rule:** $\lim_{x \to a} [f(x) + g(x)] = L + M$

2. **Difference Rule:** $\lim_{x \to a} [f(x) - g(x)] = L - M$

3. **Constant Multiple:** $\lim_{x \to a} [c \cdot f(x)] = c \cdot L$

4. **Product Rule:** $\lim_{x \to a} [f(x) \cdot g(x)] = L \cdot M$

5. **Quotient Rule:** $\lim_{x \to a} \frac{f(x)}{g(x)} = \frac{L}{M}$ (if $M \neq 0$)

6. **Power Rule:** $\lim_{x \to a} [f(x)]^n = L^n$

7. **Root Rule:** $\lim_{x \to a} \sqrt[n]{f(x)} = \sqrt[n]{L}$ (if $L > 0$ for even $n$)

### Example 3.1: Using Limit Laws

Evaluate: $\lim_{x \to 2} \frac{x^3 + 3x^2 - 4}{x^2 + 1}$

**Solution:**
$$\lim_{x \to 2} \frac{x^3 + 3x^2 - 4}{x^2 + 1} = \frac{\lim_{x \to 2}(x^3 + 3x^2 - 4)}{\lim_{x \to 2}(x^2 + 1)}$$

Numerator: $2^3 + 3(2^2) - 4 = 8 + 12 - 4 = 16$

Denominator: $2^2 + 1 = 5$

$$= \frac{16}{5}$$

---

## 4. Indeterminate Forms and Techniques

When direct substitution gives $\frac{0}{0}$, $\frac{\infty}{\infty}$, $0 \cdot \infty$, $\infty - \infty$, $0^0$, $1^\infty$, or $\infty^0$, we have **indeterminate forms**.

### Common Techniques:

#### 4.1 Algebraic Simplification

**Example:** $\lim_{x \to 3} \frac{x^2 - 9}{x - 3}$

Direct: $\frac{0}{0}$ (indeterminate)

**Solution:** Factor and cancel
$$\lim_{x \to 3} \frac{x^2 - 9}{x - 3} = \lim_{x \to 3} \frac{(x-3)(x+3)}{x-3} = \lim_{x \to 3} (x+3) = 6$$

#### 4.2 Rationalization

**Example:** $\lim_{x \to 0} \frac{\sqrt{x+4} - 2}{x}$

Direct: $\frac{0}{0}$

**Solution:** Multiply by conjugate
$$\lim_{x \to 0} \frac{\sqrt{x+4} - 2}{x} \cdot \frac{\sqrt{x+4} + 2}{\sqrt{x+4} + 2}$$

$$= \lim_{x \to 0} \frac{(x+4) - 4}{x(\sqrt{x+4} + 2)} = \lim_{x \to 0} \frac{x}{x(\sqrt{x+4} + 2)}$$

$$= \lim_{x \to 0} \frac{1}{\sqrt{x+4} + 2} = \frac{1}{4}$$

#### 4.3 Trigonometric Limits

**Key Limits:**
$$\lim_{x \to 0} \frac{\sin x}{x} = 1$$

$$\lim_{x \to 0} \frac{1 - \cos x}{x} = 0$$

$$\lim_{x \to 0} \frac{1 - \cos x}{x^2} = \frac{1}{2}$$

**Example:** $\lim_{x \to 0} \frac{\sin(3x)}{x}$

**Solution:**
$$\lim_{x \to 0} \frac{\sin(3x)}{x} = \lim_{x \to 0} \frac{\sin(3x)}{3x} \cdot 3 = 1 \cdot 3 = 3$$

---

## 5. One-Sided Limits

Sometimes the limit depends on which direction we approach from.

**Left-hand limit:** $\lim_{x \to a^-} f(x) = L$ (approaching from left, $x < a$)

**Right-hand limit:** $\lim_{x \to a^+} f(x) = R$ (approaching from right, $x > a$)

**Theorem:** $\lim_{x \to a} f(x) = L$ if and only if:
$$\lim_{x \to a^-} f(x) = \lim_{x \to a^+} f(x) = L$$

### Example 5.1: Piecewise Function

Consider:
$$f(x) = \begin{cases}
x^2 & \text{if } x < 1 \\
2x & \text{if } x \geq 1
\end{cases}$$

At $x = 1$:
- Left limit: $\lim_{x \to 1^-} f(x) = \lim_{x \to 1^-} x^2 = 1$
- Right limit: $\lim_{x \to 1^+} f(x) = \lim_{x \to 1^+} 2x = 2$

Since $1 \neq 2$, the limit $\lim_{x \to 1} f(x)$ **does not exist**.

### Example 5.2: Absolute Value Function

Consider: $f(x) = \frac{|x|}{x}$

At $x = 0$:
- Left limit: $\lim_{x \to 0^-} \frac{|x|}{x} = \lim_{x \to 0^-} \frac{-x}{x} = -1$
- Right limit: $\lim_{x \to 0^+} \frac{|x|}{x} = \lim_{x \to 0^+} \frac{x}{x} = 1$

The limit does not exist, but both one-sided limits exist.

**ML Application:** ReLU activation function has different one-sided derivatives at zero.

---

## 6. Infinite Limits and Limits at Infinity

### 6.1 Infinite Limits

**Notation:** $\lim_{x \to a} f(x) = \infty$ (or $-\infty$)

**Meaning:** As $x \to a$, $f(x)$ grows without bound.

**Example:** $\lim_{x \to 0} \frac{1}{x^2} = \infty$

As $x \to 0$ from either side, $\frac{1}{x^2} \to \infty$.

**Note:** This is NOT saying the limit exists! It's describing unbounded behavior.

**Vertical Asymptotes:** If $\lim_{x \to a^-} f(x) = \pm\infty$ or $\lim_{x \to a^+} f(x) = \pm\infty$, then $x = a$ is a vertical asymptote.

### 6.2 Limits at Infinity

**Notation:** $\lim_{x \to \infty} f(x) = L$ or $\lim_{x \to -\infty} f(x) = L$

**Example 6.2.1:** $\lim_{x \to \infty} \frac{1}{x} = 0$

**Example 6.2.2:** $\lim_{x \to \infty} \frac{3x^2 + 2x - 1}{x^2 + 5}$

**Solution:** Divide numerator and denominator by highest power ($x^2$):
$$\lim_{x \to \infty} \frac{3x^2 + 2x - 1}{x^2 + 5} = \lim_{x \to \infty} \frac{3 + \frac{2}{x} - \frac{1}{x^2}}{1 + \frac{5}{x^2}}$$

As $x \to \infty$: $\frac{2}{x} \to 0$, $\frac{1}{x^2} \to 0$, $\frac{5}{x^2} \to 0$

$$= \frac{3 + 0 - 0}{1 + 0} = 3$$

**Horizontal Asymptotes:** If $\lim_{x \to \infty} f(x) = L$ or $\lim_{x \to -\infty} f(x) = L$, then $y = L$ is a horizontal asymptote.

**General Rule for Rational Functions:** For $f(x) = \frac{a_n x^n + \cdots}{b_m x^m + \cdots}$:

- If $n < m$: $\lim_{x \to \infty} f(x) = 0$
- If $n = m$: $\lim_{x \to \infty} f(x) = \frac{a_n}{b_m}$
- If $n > m$: $\lim_{x \to \infty} f(x) = \pm\infty$

---

## 7. Continuity

### 7.1 Definition of Continuity

A function $f$ is **continuous at** $x = a$ if:
1. $f(a)$ is defined
2. $\lim_{x \to a} f(x)$ exists
3. $\lim_{x \to a} f(x) = f(a)$

**Intuition:** You can draw the graph without lifting your pen.

**Continuous on an interval:** $f$ is continuous at every point in the interval.

### 7.2 Types of Continuity

**Everywhere Continuous:** Polynomials, $e^x$, $\sin x$, $\cos x$

**Continuous on Domain:** Rational functions, $\sqrt{x}$, $\ln x$, $\tan x$

### Example 7.1: Testing Continuity

Test continuity of $f(x) = \frac{x^2 - 1}{x - 1}$ at $x = 1$.

**Check conditions:**
1. $f(1) = \frac{0}{0}$ - **NOT DEFINED** ✗
2. $\lim_{x \to 1} f(x) = \lim_{x \to 1} \frac{(x-1)(x+1)}{x-1} = \lim_{x \to 1} (x+1) = 2$ ✓
3. Cannot equal $f(1)$ since $f(1)$ undefined ✗

**Conclusion:** $f$ is **discontinuous** at $x = 1$.

**Removable Discontinuity:** Can be "fixed" by redefining:
$$g(x) = \begin{cases}
\frac{x^2-1}{x-1} & \text{if } x \neq 1 \\
2 & \text{if } x = 1
\end{cases}$$

Now $g$ is continuous everywhere.

---

## 8. Types of Discontinuities

### 8.1 Removable Discontinuity

**Characteristic:** $\lim_{x \to a} f(x)$ exists, but either $f(a)$ undefined or $f(a) \neq \lim_{x \to a} f(x)$

**Example:** $f(x) = \frac{\sin x}{x}$ at $x = 0$

- Limit exists: $\lim_{x \to 0} \frac{\sin x}{x} = 1$
- Function undefined at $x = 0$
- **Fix:** Define $f(0) = 1$

### 8.2 Jump Discontinuity

**Characteristic:** Left and right limits exist but are different

**Example:** $f(x) = \lfloor x \rfloor$ (floor function) at integers

At $x = 2$:
- $\lim_{x \to 2^-} \lfloor x \rfloor = 1$
- $\lim_{x \to 2^+} \lfloor x \rfloor = 2$
- $f(2) = 2$

**Cannot be removed** - intrinsic to the function.

**ML Example:** Step activation functions have jump discontinuities.

### 8.3 Infinite Discontinuity

**Characteristic:** Function approaches $\pm\infty$ at the point

**Example:** $f(x) = \frac{1}{x}$ at $x = 0$

- $\lim_{x \to 0^+} \frac{1}{x} = \infty$
- $\lim_{x \to 0^-} \frac{1}{x} = -\infty$

**Vertical asymptote** at $x = 0$.

### 8.4 Oscillating Discontinuity

**Characteristic:** Function oscillates infinitely as $x \to a$

**Example:** $f(x) = \sin(\frac{1}{x})$ at $x = 0$

As $x \to 0$, $\frac{1}{x} \to \infty$, so $\sin(\frac{1}{x})$ oscillates between -1 and 1 infinitely fast.

---

## 9. Properties of Continuous Functions

**Theorem 9.1 (Continuity Preservation):** If $f$ and $g$ are continuous at $x = a$, then:
- $f + g$, $f - g$, $f \cdot g$ are continuous at $a$
- $\frac{f}{g}$ is continuous at $a$ (if $g(a) \neq 0$)
- $f \circ g$ is continuous at $a$ (composition)

**Theorem 9.2 (Intermediate Value Theorem - IVT):**

If $f$ is continuous on $[a, b]$ and $k$ is between $f(a)$ and $f(b)$, then there exists $c \in (a, b)$ such that $f(c) = k$.

**Intuition:** Continuous functions take all intermediate values.

### Example 9.1: IVT Application - Root Finding

**Show:** $x^3 - 2x - 5 = 0$ has a solution in $(2, 3)$.

**Proof:** Let $f(x) = x^3 - 2x - 5$

$f$ is a polynomial, so continuous everywhere.

Evaluate:
- $f(2) = 8 - 4 - 5 = -1 < 0$
- $f(3) = 27 - 6 - 5 = 16 > 0$

Since $f(2) < 0 < f(3)$ and $f$ is continuous on $[2, 3]$, by IVT there exists $c \in (2, 3)$ such that $f(c) = 0$. ∎

**Data Science Application:** Bisection method for optimization uses IVT.

---

## 10. Extreme Value Theorem

**Theorem (Extreme Value Theorem - EVT):**

If $f$ is continuous on a closed interval $[a, b]$, then $f$ attains both a maximum and minimum value on $[a, b]$.

**Significance:**
- Guarantees optimal solutions exist for continuous functions on closed domains
- Foundation for optimization algorithms
- Used in machine learning loss minimization

### Example 10.1: EVT Application

Consider $f(x) = x^2 - 4x + 5$ on $[0, 5]$.

$f$ is continuous (polynomial), so by EVT:
- Maximum exists
- Minimum exists

Finding them:
- $f(0) = 5$
- $f(5) = 25 - 20 + 5 = 10$
- Critical point: $f'(x) = 2x - 4 = 0 \implies x = 2$
- $f(2) = 4 - 8 + 5 = 1$

**Maximum:** $f(5) = 10$
**Minimum:** $f(2) = 1$

---

## 11. Data Science Applications

### 11.1 Gradient Descent Convergence

**Problem:** Minimize loss function $L(\theta)$

**Requirement:** $L$ must be continuous (ideally differentiable) for gradient descent to work.

**Why limits matter:**
$$\nabla L(\theta) = \lim_{h \to 0} \frac{L(\theta + h) - L(\theta)}{h}$$

The gradient is itself a limit! Continuity ensures gradients exist and behave predictably.

### 11.2 Loss Function Design

**Smooth Loss Functions:**
- Mean Squared Error: $L = \frac{1}{n}\sum(y_i - \hat{y}_i)^2$ (continuous, differentiable)
- Cross-Entropy: $L = -\sum y_i \log(\hat{y}_i)$ (continuous where defined)

**Non-Smooth Loss:**
- 0-1 Loss: $L = \mathbb{1}(y \neq \hat{y})$ (discontinuous)
  - Problem: No gradients!
  - Solution: Use continuous approximations

### 11.3 Activation Functions

**Sigmoid:** $\sigma(x) = \frac{1}{1 + e^{-x}}$
- Continuous everywhere
- Smooth (infinitely differentiable)
- Used in logistic regression

**ReLU:** $\text{ReLU}(x) = \max(0, x)$
- Continuous everywhere
- Not differentiable at $x = 0$ (but one-sided limits exist)
- Most popular in deep learning despite non-differentiability

**Step Function:** $f(x) = \begin{cases} 0 & x < 0 \\ 1 & x \geq 0 \end{cases}$
- Discontinuous
- Not used in modern ML (no gradients)

### 11.4 Bisection Method for Root Finding

**Algorithm:**
1. Start with $[a, b]$ where $f(a)f(b) < 0$ (opposite signs)
2. Compute midpoint $c = \frac{a+b}{2}$
3. If $f(c) = 0$, done
4. If $f(a)f(c) < 0$, set $b = c$; else set $a = c$
5. Repeat until desired accuracy

**Convergence:** Guaranteed by IVT for continuous $f$.

### 11.5 Numerical Stability

**Example:** Computing $\frac{\sin x}{x}$ near $x = 0$

Direct computation: $\frac{\sin(10^{-10})}{10^{-10}}$ risks numerical errors.

**Solution:** Use limit: $\lim_{x \to 0} \frac{\sin x}{x} = 1$

For small $|x|$, return 1 directly.

---

## 12. Common Pitfalls and Misconceptions

### Pitfall 1: Confusing Limits with Function Values

**Wrong:** If $f(2) = 5$, then $\lim_{x \to 2} f(x) = 5$

**Correct:** The limit depends on behavior *near* 2, not *at* 2.

**Example:** $f(x) = \frac{x^2-4}{x-2}$ is undefined at $x=2$, but $\lim_{x \to 2} f(x) = 4$.

### Pitfall 2: Assuming All Limits Exist

**Wrong:** Every function has a limit at every point.

**Correct:** Limits may not exist due to:
- Oscillation: $\sin(\frac{1}{x})$ at $x = 0$
- Jump: $\lfloor x \rfloor$ at integers
- Unbounded: $\frac{1}{x}$ at $x = 0$

### Pitfall 3: Ignoring Domain

**Wrong:** $\lim_{x \to 0} \sqrt{x} = 0$

**Issue:** $\sqrt{x}$ only defined for $x \geq 0$, so left-hand limit doesn't exist.

**Correct:** $\lim_{x \to 0^+} \sqrt{x} = 0$ (right-hand limit only)

### Pitfall 4: Misapplying L'Hôpital's Rule

(Covered in Week 10 - Derivatives)

**Wrong:** Using L'Hôpital on determinate forms

**Correct:** Only use for $\frac{0}{0}$ or $\frac{\infty}{\infty}$ forms

### Pitfall 5: Continuity vs Differentiability

**Wrong:** Continuous functions are always differentiable.

**Correct:** Continuity does NOT imply differentiability.

**Example:** $f(x) = |x|$ is continuous at $x = 0$ but not differentiable there.

---

## 13. Worked Examples

### Example 13.1: Complex Limit

**Evaluate:** $\lim_{x \to 0} \frac{\sqrt{1+x} - \sqrt{1-x}}{x}$

**Solution:**
Direct substitution: $\frac{0}{0}$ (indeterminate)

Rationalize numerator:
$$\frac{\sqrt{1+x} - \sqrt{1-x}}{x} \cdot \frac{\sqrt{1+x} + \sqrt{1-x}}{\sqrt{1+x} + \sqrt{1-x}}$$

$$= \frac{(1+x) - (1-x)}{x(\sqrt{1+x} + \sqrt{1-x})} = \frac{2x}{x(\sqrt{1+x} + \sqrt{1-x})}$$

$$= \frac{2}{\sqrt{1+x} + \sqrt{1-x}}$$

Now substitute $x = 0$:
$$= \frac{2}{\sqrt{1} + \sqrt{1}} = \frac{2}{2} = 1$$

**Answer:** $\lim_{x \to 0} \frac{\sqrt{1+x} - \sqrt{1-x}}{x} = 1$

### Example 13.2: Piecewise Continuity

**Given:**
$$f(x) = \begin{cases}
\frac{x^2 - 9}{x - 3} & \text{if } x \neq 3 \\
k & \text{if } x = 3
\end{cases}$$

**Find $k$ to make $f$ continuous at $x = 3$.**

**Solution:**
For continuity at $x = 3$: $\lim_{x \to 3} f(x) = f(3) = k$

Compute limit:
$$\lim_{x \to 3} \frac{x^2 - 9}{x - 3} = \lim_{x \to 3} \frac{(x-3)(x+3)}{x-3} = \lim_{x \to 3} (x+3) = 6$$

For continuity: $k = 6$

**Answer:** $k = 6$

### Example 13.3: Infinite Limit at Infinity

**Evaluate:** $\lim_{x \to \infty} \frac{2x^3 - 5x}{x^2 + 3}$

**Solution:**
Divide by highest power in denominator ($x^2$):
$$\lim_{x \to \infty} \frac{2x^3 - 5x}{x^2 + 3} = \lim_{x \to \infty} \frac{2x - \frac{5}{x}}{1 + \frac{3}{x^2}}$$

As $x \to \infty$:
- Numerator: $2x \to \infty$
- Denominator: $1 + 0 = 1$

$$\lim_{x \to \infty} \frac{2x - \frac{5}{x}}{1 + \frac{3}{x^2}} = \infty$$

**Answer:** $\lim_{x \to \infty} \frac{2x^3 - 5x}{x^2 + 3} = \infty$

### Example 13.4: Using IVT

**Show that $\cos x = x$ has a solution in $(0, \frac{\pi}{2})$.**

**Solution:**
Let $f(x) = \cos x - x$

$f$ is continuous (difference of continuous functions).

Evaluate endpoints:
- $f(0) = \cos(0) - 0 = 1 - 0 = 1 > 0$
- $f(\frac{\pi}{2}) = \cos(\frac{\pi}{2}) - \frac{\pi}{2} = 0 - \frac{\pi}{2} < 0$

Since $f(0) > 0$ and $f(\frac{\pi}{2}) < 0$, and $f$ is continuous on $[0, \frac{\pi}{2}]$, by IVT there exists $c \in (0, \frac{\pi}{2})$ such that $f(c) = 0$.

This means $\cos c = c$. ∎

---

## 14. Practice Problems

### Basic Problems (Fundamental Understanding)

**Problem 1:** Evaluate $\lim_{x \to 4} \frac{x^2 - 16}{x - 4}$

**Problem 2:** Find $\lim_{x \to 2^-} f(x)$ and $\lim_{x \to 2^+} f(x)$ where:
$$f(x) = \begin{cases}
x^2 - 1 & \text{if } x < 2 \\
3x & \text{if } x \geq 2
\end{cases}$$

**Problem 3:** Evaluate $\lim_{x \to \infty} \frac{5x^2 + 2x - 1}{2x^2 - 3x + 4}$

**Problem 4:** Test continuity of $g(x) = \frac{x^2 - 25}{x - 5}$ at $x = 5$.

**Problem 5:** Classify the discontinuity (if any) of $h(x) = \frac{|x|}{x}$ at $x = 0$.

### Intermediate Problems (Application & Analysis)

**Problem 6:** Evaluate $\lim_{x \to 0} \frac{\sqrt{4+x} - 2}{x}$

**Problem 7:** Find $\lim_{x \to 0} \frac{\sin(5x)}{3x}$

**Problem 8:** For what value of $k$ is $f$ continuous everywhere?
$$f(x) = \begin{cases}
\frac{x^2 + 3x - 10}{x - 2} & \text{if } x \neq 2 \\
k & \text{if } x = 2
\end{cases}$$

**Problem 9:** Evaluate $\lim_{x \to -\infty} \frac{4x^3 - 2x}{x^3 + x + 1}$

**Problem 10:** Use IVT to show $x^3 + x - 1 = 0$ has a solution in $(0, 1)$.

### Advanced Problems (Synthesis & Proof)

**Problem 11:** Prove that $f(x) = x^3$ is continuous at $x = 2$ using the ε-δ definition.

**Problem 12:** Find all values of $a$ and $b$ that make $f$ continuous:
$$f(x) = \begin{cases}
ax + b & \text{if } x < 1 \\
x^2 & \text{if } 1 \leq x < 2 \\
3x & \text{if } x \geq 2
\end{cases}$$

**Problem 13:** Evaluate $\lim_{x \to 0} \frac{1 - \cos x}{x^2}$ (Hint: Use $1 - \cos x = 2\sin^2(\frac{x}{2})$)

**Problem 14:** Show that if $f$ is continuous on $[a, b]$ and $f(a) < 0 < f(b)$, then the equation $f(x) = 0$ has at least one solution in $(a, b)$.

**Problem 15:** Design a continuous activation function that approximates the step function $\theta(x) = \begin{cases} 0 & x < 0 \\ 1 & x \geq 0 \end{cases}$. Graph your function and explain its ML applications.

---

## Summary and Key Takeaways

**Limits:**
- Describe function behavior near points (not necessarily at points)
- Computed using limit laws and algebraic techniques
- One-sided limits handle directional approach
- Infinite limits and limits at infinity describe unbounded behavior and asymptotes

**Continuity:**
- Three conditions: defined, limit exists, limit equals value
- Continuous functions are "unbroken" - no jumps, holes, or vertical asymptotes
- Types of discontinuities: removable, jump, infinite, oscillating

**Major Theorems:**
- **Intermediate Value Theorem**: Continuous functions take all intermediate values
- **Extreme Value Theorem**: Continuous functions on closed intervals attain max/min

**Data Science Relevance:**
- Gradient descent requires continuous, ideally differentiable loss functions
- Activation functions balance continuity with computational efficiency
- Numerical methods (root finding, optimization) rely on IVT and EVT
- Understanding limits crucial for algorithm convergence analysis

**Common Techniques:**
- Algebraic simplification (factoring, canceling)
- Rationalization (multiply by conjugate)
- Trigonometric identities and limits
- Dividing by highest power for limits at infinity

**Next Week Preview:** Derivatives - the formal limit definition of instantaneous rate of change. Building on limits, we'll explore differentiation rules, optimization, and critical point analysis.

---

## References and Further Reading

1. **IIT Madras Mathematics I Course Materials** - Week 9 lectures and notes
2. Stewart, J. *Calculus: Early Transcendentals* (Sections 2.1-2.6)
3. Spivak, M. *Calculus* (Chapters 5-7) - Rigorous treatment
4. Khan Academy: [Limits and Continuity](https://www.khanacademy.org/math/calculus-1/cs1-limits-and-continuity)
5. 3Blue1Brown: [Essence of Calculus](https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr) - Visual intuition
6. Goodfellow, I., et al. *Deep Learning* (Chapter 4) - Numerical optimization and continuous functions
7. MIT OCW 18.01: Single Variable Calculus - Lecture notes and problem sets

---

*This comprehensive guide provides both theoretical foundations and practical applications of limits and continuity. Master these concepts to build a solid foundation for calculus, optimization, and advanced machine learning topics.*
