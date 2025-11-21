# Week 10: Derivatives - Fundamentals and Applications

---
**Date**: 2025-11-22
**Course**: BSMA1001 - Mathematics for Data Science I
**Level**: Foundation
**Week**: 10 of 12
**Source**: IIT Madras Mathematics I Week 10
**Topic Area**: Calculus - Derivatives
**Tags**: #BSMA1001 #Derivatives #Differentiation #Week10 #Foundation #Calculus
---

## Overview

**Derivatives** measure instantaneous rates of change and are the cornerstone of optimization in data science. Building on limits from Week 9, derivatives formalize the concept of "slope at a point" and enable us to find where functions increase, decrease, and reach optimal values.

**Why this matters for Data Science:**
- **Gradient Descent**: The derivative *is* the gradient used to minimize loss functions
- **Backpropagation**: Neural network training relies on computing derivatives via chain rule
- **Feature Importance**: Derivatives show how sensitive outputs are to input changes
- **Optimization**: Finding maxima/minima is central to model training
- **Marginal Analysis**: Understanding how small changes affect outcomes

**Week 10 Learning Objectives:**
1. Understand derivatives as limits and rates of change
2. Master differentiation rules (power, product, quotient, chain)
3. Compute derivatives of common functions
4. Find critical points and classify them
5. Apply first and second derivative tests
6. Solve optimization problems
7. Understand relationship between derivatives and continuity
8. Connect derivatives to machine learning concepts

---

## 1. Definition of the Derivative

**Formal Definition:** The derivative of $f(x)$ at $x = a$ is:
$$f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$$

provided this limit exists.

**Alternative form:**
$$f'(a) = \lim_{x \to a} \frac{f(x) - f(a)}{x - a}$$

**Notation:** Multiple notations for derivatives:
- $f'(x)$ (Lagrange)
- $\frac{df}{dx}$ (Leibniz)
- $\frac{d}{dx}f(x)$ (operator notation)
- $Df(x)$ (differential operator)
- $\dot{x}$ (Newton, for time derivatives)

**Interpretation:**
- **Geometric:** Slope of tangent line at point $(a, f(a))$
- **Physical:** Instantaneous rate of change
- **Data Science:** Gradient component, sensitivity measure

### Example 1.1: Computing Derivative from Definition

Find $f'(x)$ for $f(x) = x^2$ using the limit definition.

**Solution:**
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

$$= \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h}$$

$$= \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h}$$

$$= \lim_{h \to 0} \frac{2xh + h^2}{h}$$

$$= \lim_{h \to 0} \frac{h(2x + h)}{h}$$

$$= \lim_{h \to 0} (2x + h) = 2x$$

**Result:** $f'(x) = 2x$

**Verification:** At $x = 3$, slope of $f(x) = x^2$ is $f'(3) = 6$.

---

## 2. Differentiability and Continuity

**Theorem:** If $f$ is differentiable at $x = a$, then $f$ is continuous at $x = a$.

**Important:** The converse is FALSE! Continuous functions need not be differentiable.

**Counterexample:** $f(x) = |x|$ at $x = 0$
- Continuous at 0: $\lim_{x \to 0} |x| = 0 = f(0)$ ✓
- Not differentiable at 0:
  - Left derivative: $\lim_{h \to 0^-} \frac{|h| - 0}{h} = \lim_{h \to 0^-} \frac{-h}{h} = -1$
  - Right derivative: $\lim_{h \to 0^+} \frac{|h| - 0}{h} = \lim_{h \to 0^+} \frac{h}{h} = 1$
  - Since $-1 \neq 1$, derivative doesn't exist ✗

**Non-differentiable points occur at:**
1. **Corners/cusps:** $f(x) = |x|$ at $x = 0$
2. **Vertical tangents:** $f(x) = \sqrt[3]{x}$ at $x = 0$
3. **Discontinuities:** Any discontinuous point

**ML Relevance:** ReLU function $\max(0, x)$ is continuous but not differentiable at $x = 0$ (uses subgradient in practice).

---

## 3. Basic Differentiation Rules

### 3.1 Power Rule

$$\frac{d}{dx}[x^n] = nx^{n-1}$$

**Examples:**
- $\frac{d}{dx}[x^3] = 3x^2$
- $\frac{d}{dx}[x^{-2}] = -2x^{-3} = -\frac{2}{x^3}$
- $\frac{d}{dx}[\sqrt{x}] = \frac{d}{dx}[x^{1/2}] = \frac{1}{2}x^{-1/2} = \frac{1}{2\sqrt{x}}$

### 3.2 Constant Rule

$$\frac{d}{dx}[c] = 0$$

Constants have zero rate of change.

### 3.3 Constant Multiple Rule

$$\frac{d}{dx}[cf(x)] = c \cdot f'(x)$$

**Example:** $\frac{d}{dx}[5x^3] = 5 \cdot 3x^2 = 15x^2$

### 3.4 Sum/Difference Rule

$$\frac{d}{dx}[f(x) \pm g(x)] = f'(x) \pm g'(x)$$

**Example:**
$$\frac{d}{dx}[x^3 + 4x^2 - 7x + 2] = 3x^2 + 8x - 7$$

### Example 3.1: Combining Rules

Find $\frac{dy}{dx}$ for $y = 3x^4 - 2x^3 + 5x - 8$.

**Solution:**
$$\frac{dy}{dx} = 3(4x^3) - 2(3x^2) + 5(1) - 0$$
$$= 12x^3 - 6x^2 + 5$$

---

## 4. Product and Quotient Rules

### 4.1 Product Rule

If $y = u(x) \cdot v(x)$, then:
$$\frac{dy}{dx} = u'v + uv'$$

**Mnemonic:** "First times derivative of second, plus second times derivative of first"

### Example 4.1: Product Rule

Find $\frac{d}{dx}[(x^2 + 1)(x^3 - 2x)]$

**Solution:**
Let $u = x^2 + 1$ and $v = x^3 - 2x$

$u' = 2x$ and $v' = 3x^2 - 2$

$$\frac{d}{dx}[uv] = u'v + uv'$$
$$= (2x)(x^3 - 2x) + (x^2 + 1)(3x^2 - 2)$$
$$= 2x^4 - 4x^2 + 3x^4 - 2x^2 + 3x^2 - 2$$
$$= 5x^4 - 3x^2 - 2$$

### 4.2 Quotient Rule

If $y = \frac{u(x)}{v(x)}$, then:
$$\frac{dy}{dx} = \frac{u'v - uv'}{v^2}$$

**Mnemonic:** "Low d-high minus high d-low, over low squared"

### Example 4.2: Quotient Rule

Find $\frac{d}{dx}\left[\frac{x^2 + 1}{x - 2}\right]$

**Solution:**
Let $u = x^2 + 1$ and $v = x - 2$

$u' = 2x$ and $v' = 1$

$$\frac{dy}{dx} = \frac{u'v - uv'}{v^2}$$
$$= \frac{(2x)(x-2) - (x^2+1)(1)}{(x-2)^2}$$
$$= \frac{2x^2 - 4x - x^2 - 1}{(x-2)^2}$$
$$= \frac{x^2 - 4x - 1}{(x-2)^2}$$

---

## 5. Chain Rule

The **chain rule** handles composite functions and is crucial for backpropagation in neural networks.

**Theorem (Chain Rule):** If $y = f(g(x))$, then:
$$\frac{dy}{dx} = f'(g(x)) \cdot g'(x)$$

**Alternative notation:** If $y = f(u)$ and $u = g(x)$, then:
$$\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}$$

**Intuition:** Derivative of outer function (evaluated at inner) times derivative of inner function.

### Example 5.1: Chain Rule

Find $\frac{d}{dx}[(x^2 + 3x)^5]$

**Solution:**
Outer function: $f(u) = u^5$, so $f'(u) = 5u^4$

Inner function: $u = x^2 + 3x$, so $u' = 2x + 3$

By chain rule:
$$\frac{dy}{dx} = 5(x^2 + 3x)^4 \cdot (2x + 3)$$

### Example 5.2: Multiple Compositions

Find $\frac{d}{dx}[\sqrt{x^2 + 1}]$

**Solution:**
Rewrite as $(x^2 + 1)^{1/2}$

Outer: $f(u) = u^{1/2}$, so $f'(u) = \frac{1}{2}u^{-1/2}$

Inner: $u = x^2 + 1$, so $u' = 2x$

$$\frac{dy}{dx} = \frac{1}{2}(x^2 + 1)^{-1/2} \cdot 2x = \frac{x}{\sqrt{x^2 + 1}}$$

### Example 5.3: Backpropagation Connection

In neural networks, if loss $L$ depends on activation $a$, which depends on weighted input $z$, which depends on weight $w$:

$$L \to a \to z \to w$$

Then:
$$\frac{dL}{dw} = \frac{dL}{da} \cdot \frac{da}{dz} \cdot \frac{dz}{dw}$$

This is the chain rule! Backpropagation computes these derivatives efficiently.

---

## 6. Derivatives of Common Functions

### 6.1 Exponential Functions

$$\frac{d}{dx}[e^x] = e^x$$

$$\frac{d}{dx}[a^x] = a^x \ln(a)$$

$$\frac{d}{dx}[e^{f(x)}] = e^{f(x)} \cdot f'(x)$$ (chain rule)

### 6.2 Logarithmic Functions

$$\frac{d}{dx}[\ln x] = \frac{1}{x}$$

$$\frac{d}{dx}[\log_a x] = \frac{1}{x \ln a}$$

$$\frac{d}{dx}[\ln|f(x)|] = \frac{f'(x)}{f(x)}$$ (chain rule)

### 6.3 Trigonometric Functions

$$\frac{d}{dx}[\sin x] = \cos x$$

$$\frac{d}{dx}[\cos x] = -\sin x$$

$$\frac{d}{dx}[\tan x] = \sec^2 x$$

$$\frac{d}{dx}[\cot x] = -\csc^2 x$$

$$\frac{d}{dx}[\sec x] = \sec x \tan x$$

$$\frac{d}{dx}[\csc x] = -\csc x \cot x$$

### 6.4 Inverse Trigonometric Functions

$$\frac{d}{dx}[\arcsin x] = \frac{1}{\sqrt{1-x^2}}$$

$$\frac{d}{dx}[\arccos x] = -\frac{1}{\sqrt{1-x^2}}$$

$$\frac{d}{dx}[\arctan x] = \frac{1}{1+x^2}$$

### Example 6.1: Exponential with Chain Rule

Find $\frac{d}{dx}[e^{x^2 + 3x}]$

**Solution:**
$$\frac{d}{dx}[e^{x^2 + 3x}] = e^{x^2 + 3x} \cdot (2x + 3)$$

### Example 6.2: Logarithmic Derivative

Find $\frac{d}{dx}[\ln(x^2 + 5)]$

**Solution:**
$$\frac{d}{dx}[\ln(x^2 + 5)] = \frac{1}{x^2 + 5} \cdot 2x = \frac{2x}{x^2 + 5}$$

---

## 7. Higher-Order Derivatives

The **second derivative** is the derivative of the derivative:
$$f''(x) = \frac{d^2f}{dx^2} = \frac{d}{dx}[f'(x)]$$

**Notation:**
- First derivative: $f'$, $\frac{df}{dx}$, $D_x f$
- Second derivative: $f''$, $\frac{d^2f}{dx^2}$, $D_x^2 f$
- Third derivative: $f'''$, $\frac{d^3f}{dx^3}$
- $n$-th derivative: $f^{(n)}$, $\frac{d^nf}{dx^n}$

**Interpretation:**
- $f'(x)$: Rate of change (velocity if $f$ is position)
- $f''(x)$: Rate of change of rate of change (acceleration)
- $f''(x) > 0$: Concave up (function curving upward)
- $f''(x) < 0$: Concave down (function curving downward)

### Example 7.1: Second Derivative

For $f(x) = x^4 - 3x^3 + 2x$:

First derivative:
$$f'(x) = 4x^3 - 9x^2 + 2$$

Second derivative:
$$f''(x) = 12x^2 - 18x$$

Third derivative:
$$f'''(x) = 24x - 18$$

---

## 8. Critical Points and Extrema

### 8.1 Critical Points

**Definition:** $x = c$ is a **critical point** of $f$ if:
1. $f'(c) = 0$ (horizontal tangent), OR
2. $f'(c)$ does not exist (cusp, corner, vertical tangent)

**Significance:** Local extrema (maxima/minima) can only occur at critical points or endpoints.

### 8.2 First Derivative Test

To classify critical point $x = c$:

1. If $f'$ changes from **positive to negative** at $c$: **local maximum**
2. If $f'$ changes from **negative to positive** at $c$: **local minimum**
3. If $f'$ does **not change sign** at $c$: **neither** (inflection point or saddle)

### Example 8.1: Finding and Classifying Critical Points

Find and classify critical points of $f(x) = x^3 - 3x^2 - 9x + 5$.

**Solution:**

Step 1: Find $f'(x)$:
$$f'(x) = 3x^2 - 6x - 9$$

Step 2: Solve $f'(x) = 0$:
$$3x^2 - 6x - 9 = 0$$
$$x^2 - 2x - 3 = 0$$
$$(x - 3)(x + 1) = 0$$

Critical points: $x = 3$ and $x = -1$

Step 3: Test intervals using sign of $f'(x)$:

| Interval | Test $x$ | $f'(x)$ | Sign | Behavior |
|----------|----------|---------|------|----------|
| $x < -1$ | $x = -2$ | $3(4) - 6(-2) - 9 = 15$ | $+$ | Increasing |
| $-1 < x < 3$ | $x = 0$ | $-9$ | $-$ | Decreasing |
| $x > 3$ | $x = 4$ | $3(16) - 6(4) - 9 = 15$ | $+$ | Increasing |

**Classification:**
- $x = -1$: $f'$ changes from $+$ to $-$ → **Local maximum**
- $x = 3$: $f'$ changes from $-$ to $+$ → **Local minimum**

**Function values:**
- $f(-1) = -1 - 3 + 9 + 5 = 10$ (local max)
- $f(3) = 27 - 27 - 27 + 5 = -22$ (local min)

### 8.3 Second Derivative Test

Alternative method to classify critical points where $f'(c) = 0$:

1. If $f''(c) > 0$: **local minimum** (concave up)
2. If $f''(c) < 0$: **local maximum** (concave down)
3. If $f''(c) = 0$: **inconclusive** (use first derivative test)

### Example 8.2: Second Derivative Test

For $f(x) = x^3 - 3x^2 - 9x + 5$ (from Example 8.1):

$$f''(x) = 6x - 6$$

At $x = -1$: $f''(-1) = -6 - 6 = -12 < 0$ → **Local maximum** ✓

At $x = 3$: $f''(3) = 18 - 6 = 12 > 0$ → **Local minimum** ✓

---

## 9. Optimization Problems

**General Strategy:**
1. Identify the quantity to optimize (objective function)
2. Express it as a function of one variable
3. Find critical points (set derivative = 0)
4. Test critical points and endpoints
5. Verify solution makes sense in context

### Example 9.1: Maximizing Area

A farmer has 100 meters of fence to enclose a rectangular field next to a river (no fence needed along river). What dimensions maximize the enclosed area?

**Solution:**

Step 1: Set up variables
- Let $x$ = width (perpendicular to river)
- Let $y$ = length (parallel to river)

Step 2: Constraints
- Fence: $2x + y = 100$
- So: $y = 100 - 2x$

Step 3: Objective function
$$A(x) = xy = x(100 - 2x) = 100x - 2x^2$$

Step 4: Find critical points
$$A'(x) = 100 - 4x$$

Set $A'(x) = 0$:
$$100 - 4x = 0$$
$$x = 25$$

Step 5: Verify it's a maximum
$$A''(x) = -4 < 0$$ → Concave down → Maximum ✓

**Solution:** $x = 25$ m, $y = 100 - 50 = 50$ m

**Maximum area:** $A = 25 \times 50 = 1250$ m²

### Example 9.2: Minimizing Cost

A company produces $x$ units at cost $C(x) = 100 + 10x + 0.01x^2$ dollars. Find the production level that minimizes average cost per unit.

**Solution:**

Average cost: $\bar{C}(x) = \frac{C(x)}{x} = \frac{100}{x} + 10 + 0.01x$

Find critical points:
$$\bar{C}'(x) = -\frac{100}{x^2} + 0.01$$

Set $\bar{C}'(x) = 0$:
$$-\frac{100}{x^2} + 0.01 = 0$$
$$0.01 = \frac{100}{x^2}$$
$$x^2 = 10000$$
$$x = 100$$ (negative solution discarded)

Verify minimum:
$$\bar{C}''(x) = \frac{200}{x^3}$$
$$\bar{C}''(100) = \frac{200}{1000000} > 0$$ → Minimum ✓

**Solution:** Produce 100 units to minimize average cost.

**Minimum average cost:** $\bar{C}(100) = 1 + 10 + 1 = \$12$ per unit

---

## 10. Related Rates

**Related rates** problems involve finding how fast one quantity changes given how fast another related quantity changes.

**Method:**
1. Identify all variables and relationships
2. Write equation relating variables
3. Differentiate both sides with respect to time $t$ (implicit differentiation)
4. Substitute known values and solve

### Example 10.1: Expanding Circle

A circular oil spill expands so that its radius increases at 2 m/s. How fast is the area increasing when the radius is 10 m?

**Solution:**

Given: $\frac{dr}{dt} = 2$ m/s

Find: $\frac{dA}{dt}$ when $r = 10$ m

Relationship: $A = \pi r^2$

Differentiate with respect to $t$:
$$\frac{dA}{dt} = 2\pi r \cdot \frac{dr}{dt}$$

Substitute $r = 10$ and $\frac{dr}{dt} = 2$:
$$\frac{dA}{dt} = 2\pi (10)(2) = 40\pi \approx 125.7 \text{ m}^2/\text{s}$$

---

## 11. Data Science Applications

### 11.1 Gradient Descent

**Problem:** Minimize loss function $L(\theta)$

**Algorithm:** Update parameters using gradient:
$$\theta_{new} = \theta_{old} - \alpha \cdot \nabla L(\theta_{old})$$

where $\alpha$ is learning rate.

For single parameter: $\theta_{new} = \theta_{old} - \alpha \cdot \frac{dL}{d\theta}$

**Example:** Linear regression with MSE loss

Loss: $L(w) = \frac{1}{n}\sum_{i=1}^n (y_i - wx_i)^2$

Gradient: $\frac{dL}{dw} = \frac{2}{n}\sum_{i=1}^n (wx_i - y_i)x_i$

Update: $w_{new} = w_{old} - \alpha \cdot \frac{dL}{dw}$

### 11.2 Neural Network Backpropagation

**Simple network:** Input $x$ → Hidden layer → Output $\hat{y}$ → Loss $L$

Forward pass computes output.

Backward pass computes gradients using chain rule:
$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w_1}$$

where $z$ is weighted input to neuron.

### 11.3 Feature Sensitivity Analysis

**Question:** How sensitive is model output to input feature changes?

**Answer:** Partial derivative! $\frac{\partial y}{\partial x_i}$ measures sensitivity to feature $i$.

Large $|\frac{\partial y}{\partial x_i}|$ → Feature $i$ is important.

### 11.4 Learning Rate Scheduling

**Idea:** Adjust learning rate based on loss landscape curvature (second derivative).

If loss function has high curvature ($|f''|$ large), use smaller learning rate.

**Newton's method:** Uses second derivative for better updates:
$$\theta_{new} = \theta_{old} - \frac{f'(\theta_{old})}{f''(\theta_{old})}$$

### 11.5 Activation Function Derivatives

**Sigmoid:** $\sigma(x) = \frac{1}{1 + e^{-x}}$

Derivative: $\sigma'(x) = \sigma(x)(1 - \sigma(x))$

**ReLU:** $\text{ReLU}(x) = \max(0, x)$

Derivative: $\text{ReLU}'(x) = \begin{cases} 1 & x > 0 \\ 0 & x < 0 \\ \text{undefined} & x = 0 \end{cases}$

(In practice, use 0 or 1 at $x = 0$)

**Tanh:** $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$

Derivative: $\tanh'(x) = 1 - \tanh^2(x)$

---

## 12. Common Pitfalls and Misconceptions

### Pitfall 1: Forgetting Chain Rule

**Wrong:** $\frac{d}{dx}[(x^2 + 1)^3] = 3(x^2 + 1)^2$

**Correct:** $\frac{d}{dx}[(x^2 + 1)^3] = 3(x^2 + 1)^2 \cdot 2x$ ✓

Always multiply by derivative of inner function!

### Pitfall 2: Misapplying Product Rule

**Wrong:** $\frac{d}{dx}[x \cdot e^x] = 1 \cdot e^x = e^x$

**Correct:** $\frac{d}{dx}[x \cdot e^x] = 1 \cdot e^x + x \cdot e^x = e^x(1 + x)$ ✓

### Pitfall 3: Confusing $f'(x)$ and $f'(a)$

$f'(x)$ is a **function**.
$f'(a)$ is a **number** (derivative evaluated at $x = a$).

### Pitfall 4: Assuming Critical Points Are Extrema

Critical points are **candidates** for extrema. Must test them!

Example: $f(x) = x^3$ has $f'(0) = 0$, but $x = 0$ is **not** a local extremum (it's an inflection point).

### Pitfall 5: Ignoring Domain

The derivative of $f(x) = \sqrt{x}$ is $f'(x) = \frac{1}{2\sqrt{x}}$, which is undefined at $x = 0$ and for $x < 0$.

Always consider where the function and its derivative are defined.

---

## 13. Worked Examples

### Example 13.1: Complex Derivative

Find $\frac{dy}{dx}$ for $y = \frac{x^2 \sin x}{e^x}$

**Solution:**

Use quotient rule: $\frac{d}{dx}\left[\frac{u}{v}\right] = \frac{u'v - uv'}{v^2}$

$u = x^2 \sin x$ (need product rule)
$v = e^x$

Find $u'$ using product rule:
$$u' = 2x \sin x + x^2 \cos x$$

Find $v'$:
$$v' = e^x$$

Apply quotient rule:
$$\frac{dy}{dx} = \frac{(2x \sin x + x^2 \cos x)e^x - x^2 \sin x \cdot e^x}{(e^x)^2}$$

Factor out $e^x$:
$$= \frac{e^x(2x \sin x + x^2 \cos x - x^2 \sin x)}{e^{2x}}$$

Simplify:
$$= \frac{2x \sin x + x^2 \cos x - x^2 \sin x}{e^x}$$

$$= \frac{x(2 - x)\sin x + x^2 \cos x}{e^x}$$

### Example 13.2: Implicit Differentiation

Find $\frac{dy}{dx}$ for $x^2 + y^2 = 25$ (circle).

**Solution:**

Differentiate both sides with respect to $x$:
$$\frac{d}{dx}[x^2 + y^2] = \frac{d}{dx}[25]$$

$$2x + 2y\frac{dy}{dx} = 0$$

Solve for $\frac{dy}{dx}$:
$$2y\frac{dy}{dx} = -2x$$

$$\frac{dy}{dx} = -\frac{x}{y}$$

**Interpretation:** At any point $(x, y)$ on the circle, the tangent line has slope $-\frac{x}{y}$.

At $(3, 4)$: $\frac{dy}{dx} = -\frac{3}{4}$

### Example 13.3: Optimization with Constraint

Find the rectangle of maximum area that can be inscribed in a semicircle of radius $r$.

**Solution:**

Place semicircle with center at origin, diameter along $x$-axis.

Rectangle has:
- Width: $2x$ (symmetric about $y$-axis)
- Height: $y$

Constraint (on semicircle): $x^2 + y^2 = r^2$, so $y = \sqrt{r^2 - x^2}$

Area: $A(x) = 2xy = 2x\sqrt{r^2 - x^2}$

Find critical points:
$$A'(x) = 2\sqrt{r^2 - x^2} + 2x \cdot \frac{-x}{\sqrt{r^2 - x^2}}$$

$$= 2\sqrt{r^2 - x^2} - \frac{2x^2}{\sqrt{r^2 - x^2}}$$

$$= \frac{2(r^2 - x^2) - 2x^2}{\sqrt{r^2 - x^2}}$$

$$= \frac{2r^2 - 4x^2}{\sqrt{r^2 - x^2}}$$

Set $A'(x) = 0$:
$$2r^2 - 4x^2 = 0$$
$$x^2 = \frac{r^2}{2}$$
$$x = \frac{r}{\sqrt{2}}$$

Then: $y = \sqrt{r^2 - \frac{r^2}{2}} = \frac{r}{\sqrt{2}}$

**Dimensions:** Width = $\frac{2r}{\sqrt{2}} = r\sqrt{2}$, Height = $\frac{r}{\sqrt{2}}$

**Maximum area:** $A = r\sqrt{2} \cdot \frac{r}{\sqrt{2}} = r^2$

---

## 14. Practice Problems

### Basic Problems

**Problem 1:** Find $f'(x)$ for $f(x) = 3x^4 - 2x^3 + 5x - 7$

**Problem 2:** Compute $\frac{d}{dx}[(x^2 + 1)(x^3 - 2)]$ using product rule

**Problem 3:** Find $\frac{d}{dx}\left[\frac{x^2 - 4}{x + 1}\right]$ using quotient rule

**Problem 4:** Use chain rule to find $\frac{d}{dx}[(3x^2 - 5)^4]$

**Problem 5:** Find critical points of $f(x) = x^3 - 6x^2 + 9x$

### Intermediate Problems

**Problem 6:** Find $\frac{dy}{dx}$ for $y = e^{x^2} \sin(3x)$

**Problem 7:** Compute $\frac{d}{dx}[\ln(x^2 + 4x + 5)]$

**Problem 8:** Find and classify all critical points of $f(x) = x^4 - 4x^3$

**Problem 9:** A box with square base and open top has volume 32 cm³. Find dimensions that minimize surface area.

**Problem 10:** Use implicit differentiation to find $\frac{dy}{dx}$ for $x^3 + y^3 = 6xy$

### Advanced Problems

**Problem 11:** Find the point on the curve $y = \sqrt{x}$ closest to the point $(4, 0)$.

**Problem 12:** A ladder 10 m long leans against a wall. If the bottom slides away at 0.5 m/s, how fast is the top sliding down when the bottom is 6 m from the wall?

**Problem 13:** Prove that $\frac{d}{dx}[\sin x] = \cos x$ using the limit definition.

**Problem 14:** For sigmoid function $\sigma(x) = \frac{1}{1 + e^{-x}}$, show that $\sigma'(x) = \sigma(x)(1 - \sigma(x))$.

**Problem 15:** Design a gradient descent algorithm to find the minimum of $f(x) = x^4 - 4x^2 + 4$. Analyze convergence from different starting points.

---

## Summary and Key Takeaways

**Derivatives:**
- Measure instantaneous rate of change (slope of tangent line)
- Defined as a limit: $f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$
- Differentiability implies continuity (but not vice versa)

**Differentiation Rules:**
- **Power Rule:** $\frac{d}{dx}[x^n] = nx^{n-1}$
- **Product Rule:** $(uv)' = u'v + uv'$
- **Quotient Rule:** $\left(\frac{u}{v}\right)' = \frac{u'v - uv'}{v^2}$
- **Chain Rule:** $(f \circ g)' = f'(g(x)) \cdot g'(x)$ (CRITICAL for ML!)

**Critical Points and Optimization:**
- Critical points where $f'(x) = 0$ or undefined
- First derivative test: check sign changes
- Second derivative test: $f'' > 0$ → min, $f'' < 0$ → max
- Optimization: model problem, differentiate, find critical points, verify

**Data Science Applications:**
- Gradient descent uses derivatives to minimize loss
- Backpropagation is chain rule applied to neural networks
- Feature sensitivity measured by partial derivatives
- Activation functions must be differentiable (or nearly so)

**Common Functions:**
- $\frac{d}{dx}[e^x] = e^x$, $\frac{d}{dx}[\ln x] = \frac{1}{x}$
- $\frac{d}{dx}[\sin x] = \cos x$, $\frac{d}{dx}[\cos x] = -\sin x$

**Next Week Preview:** Integration - the reverse of differentiation. We'll explore antiderivatives, definite integrals, area under curves, and the Fundamental Theorem of Calculus.

---

## References and Further Reading

1. **IIT Madras Mathematics I Course Materials** - Week 10 lectures and notes
2. Stewart, J. *Calculus: Early Transcendentals* (Chapters 3-4)
3. Apostol, T. *Calculus, Volume 1* (Chapters 4-5) - Rigorous treatment
4. Khan Academy: [Derivatives](https://www.khanacademy.org/math/calculus-1/cs1-derivatives-definition-and-basic-rules)
5. 3Blue1Brown: [Essence of Calculus - Derivatives](https://www.youtube.com/watch?v=9vKqVkMQHKk)
6. Goodfellow, I., et al. *Deep Learning* (Chapter 4.3) - Gradient-based optimization
7. Bishop, C. *Pattern Recognition and Machine Learning* (Appendix C) - Calculus review for ML
8. MIT OCW 18.01: Single Variable Calculus - Differentiation lectures

---

*Master derivatives to unlock optimization, understand gradient descent, and build intuition for how neural networks learn. These concepts are fundamental to all of machine learning!*
