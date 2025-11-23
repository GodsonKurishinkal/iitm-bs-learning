# Week 7: Single Variable Calculus - Derivatives and Optimization

---
Date: 2025-11-22
Course: BSMA1003 - Mathematics for Data Science II
Level: Foundation
Week: 7 of 11
Source: IIT Madras Mathematics II Week 7
Topic Area: Calculus - Single Variable Derivatives
Tags: #BSMA1003 #Calculus #Week7 #Derivatives #Optimization #Foundation
---

## Topics Covered

1. **Limits and Continuity Review**
2. **Definition of Derivative**
3. **Differentiation Rules**
4. **Critical Points and Local Extrema**
5. **First Derivative Test**
6. **Second Derivative Test**
7. **Concavity and Inflection Points**
8. **Optimization Problems**
9. **Applications to Data Science**

---

## Key Concepts

### 1. Derivative Definition

The **derivative** of function $f(x)$ at point $a$ is:
$$f'(a) = \lim_{h \to 0} \frac{f(a+h) - f(a)}{h}$$

**Geometric interpretation**: Slope of tangent line at $(a, f(a))$
**Physical interpretation**: Instantaneous rate of change

**Alternative notations**:
- $f'(x)$, $\frac{df}{dx}$, $\frac{d}{dx}f(x)$, $Df(x)$

#### Example 1: Derivative from First Principles

Find derivative of $f(x) = x^2$ using definition.

**Solution**:
$$f'(x) = \lim_{h \to 0} \frac{(x+h)^2 - x^2}{h}$$
$$= \lim_{h \to 0} \frac{x^2 + 2xh + h^2 - x^2}{h}$$
$$= \lim_{h \to 0} \frac{2xh + h^2}{h}$$
$$= \lim_{h \to 0} (2x + h) = 2x$$

**Result**: $f'(x) = 2x$ ✓

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import derivative

# Function
def f(x):
    return x**2

# Numerical derivative
x_point = 2
deriv_numerical = derivative(f, x_point, dx=1e-6)
deriv_analytical = 2 * x_point

print(f"At x = {x_point}:")
print(f"Numerical derivative: {deriv_numerical:.6f}")
print(f"Analytical derivative: {deriv_analytical}")
```

### 2. Differentiation Rules

**Power Rule**: $\frac{d}{dx}x^n = nx^{n-1}$

**Sum Rule**: $(f + g)' = f' + g'$

**Product Rule**: $(fg)' = f'g + fg'$

**Quotient Rule**: $\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$

**Chain Rule**: $(f(g(x)))' = f'(g(x)) \cdot g'(x)$

#### Example 2: Apply All Rules

Differentiate: $h(x) = \frac{x^3 \sin(x)}{e^x}$

**Solution**:

Using quotient rule:
$$h'(x) = \frac{(x^3\sin x)' \cdot e^x - x^3\sin x \cdot (e^x)'}{(e^x)^2}$$

Product rule for numerator:
$$(x^3\sin x)' = 3x^2\sin x + x^3\cos x$$

Therefore:
$$h'(x) = \frac{(3x^2\sin x + x^3\cos x)e^x - x^3\sin x \cdot e^x}{e^{2x}}$$
$$= \frac{e^x(3x^2\sin x + x^3\cos x - x^3\sin x)}{e^{2x}}$$
$$= \frac{3x^2\sin x + x^3\cos x - x^3\sin x}{e^x}$$

```python
from sympy import *

x = symbols('x')
h = x**3 * sin(x) / exp(x)

# Symbolic differentiation
h_prime = diff(h, x)
print(f"h'(x) = {h_prime}")
print(f"Simplified: {simplify(h_prime)}")
```

### 3. Critical Points

A **critical point** occurs where $f'(x) = 0$ or $f'(x)$ is undefined.

**Theorem** (Fermat): If $f$ has a local extremum at $c$, then $c$ is a critical point.

**Types of critical points**:
- Local maximum
- Local minimum  
- Saddle point (neither max nor min)

#### Example 3: Finding Critical Points

Find critical points of $f(x) = x^3 - 6x^2 + 9x + 1$

**Solution**:

**Step 1**: Find derivative
$$f'(x) = 3x^2 - 12x + 9$$

**Step 2**: Set equal to zero
$$3x^2 - 12x + 9 = 0$$
$$x^2 - 4x + 3 = 0$$
$$(x - 1)(x - 3) = 0$$

**Critical points**: $x = 1$ and $x = 3$

```python
from scipy.optimize import fminbound, minimize_scalar

def f(x):
    return x**3 - 6*x**2 + 9*x + 1

def f_prime(x):
    return 3*x**2 - 12*x + 9

# Find critical points
from numpy.polynomial import polynomial as P
coeffs = [9, -12, 3]  # f'(x) coefficients
roots = np.roots(coeffs[::-1])
print(f"Critical points: {roots}")

# Evaluate function at critical points
for x_c in roots:
    print(f"f({x_c:.1f}) = {f(x_c):.2f}")
```

### 4. First Derivative Test

**Theorem**: At critical point $c$:
- If $f'$ changes from **positive to negative** at $c$: **local maximum**
- If $f'$ changes from **negative to positive** at $c$: **local minimum**
- If $f'$ does **not change sign**: neither (saddle point)

#### Example 4: First Derivative Test

Classify critical points of $f(x) = x^3 - 6x^2 + 9x + 1$ (from Example 3)

**Solution**:

**Test intervals** around $x = 1, 3$:

| Interval | Test Point | $f'$ | Sign | Behavior |
|----------|-----------|------|------|----------|
| $(-\infty, 1)$ | $x = 0$ | $f'(0) = 9$ | $+$ | Increasing |
| $(1, 3)$ | $x = 2$ | $f'(2) = -3$ | $-$ | Decreasing |
| $(3, \infty)$ | $x = 4$ | $f'(4) = 9$ | $+$ | Increasing |

**Conclusions**:
- At $x = 1$: $f'$ changes from $+$ to $-$ → **LOCAL MAXIMUM**
- At $x = 3$: $f'$ changes from $-$ to $+$ → **LOCAL MINIMUM**

```python
# Visualize
x = np.linspace(-1, 5, 1000)
y = f(x)
y_prime = f_prime(x)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Function
ax1.plot(x, y, 'b-', linewidth=2)
ax1.plot([1, 3], [f(1), f(3)], 'ro', markersize=10, label='Critical points')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.grid(True, alpha=0.3)
ax1.set_title('Function f(x)', fontsize=14)
ax1.legend()

# Derivative
ax2.plot(x, y_prime, 'r-', linewidth=2)
ax2.plot([1, 3], [0, 0], 'ro', markersize=10)
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=1, color='g', linestyle='--', alpha=0.5, label='x=1 (max)')
ax2.axvline(x=3, color='orange', linestyle='--', alpha=0.5, label='x=3 (min)')
ax2.grid(True, alpha=0.3)
ax2.set_title("Derivative f'(x)", fontsize=14)
ax2.legend()

plt.tight_layout()
# plt.savefig('week07_first_derivative_test.png', dpi=150, bbox_inches='tight')
```

### 5. Second Derivative Test

**Theorem**: At critical point $c$ where $f'(c) = 0$:
- If $f''(c) > 0$: **local minimum** (concave up)
- If $f''(c) < 0$: **local maximum** (concave down)
- If $f''(c) = 0$: test is **inconclusive**

#### Example 5: Second Derivative Test

Classify critical points of $f(x) = x^4 - 4x^3 + 10$

**Solution**:

**Step 1**: Find critical points
$$f'(x) = 4x^3 - 12x^2 = 4x^2(x - 3)$$
Critical points: $x = 0, 3$

**Step 2**: Compute second derivative
$$f''(x) = 12x^2 - 24x$$

**Step 3**: Evaluate at critical points
- $f''(0) = 0$ (inconclusive!)
- $f''(3) = 12(9) - 24(3) = 108 - 72 = 36 > 0$ → **LOCAL MINIMUM**

For $x = 0$, use first derivative test:
$f'$ does not change sign at $x=0$ (stays non-negative) → **saddle point**

```python
def f(x):
    return x**4 - 4*x**3 + 10

def f_double_prime(x):
    return 12*x**2 - 24*x

critical_points = [0, 3]

for x_c in critical_points:
    second_deriv = f_double_prime(x_c)
    if second_deriv > 0:
        classification = "LOCAL MINIMUM"
    elif second_deriv < 0:
        classification = "LOCAL MAXIMUM"
    else:
        classification = "INCONCLUSIVE (use first derivative test)"
    
    print(f"x = {x_c}: f''({x_c}) = {second_deriv:.1f} → {classification}")
```

### 6. Concavity and Inflection Points

**Concave up**: $f''(x) > 0$ (holds water)
**Concave down**: $f''(x) < 0$ (sheds water)

**Inflection point**: Where concavity changes ($f''(x) = 0$ and changes sign)

#### Example 6: Finding Inflection Points

Find inflection points of $f(x) = x^4 - 6x^2 + 3x - 2$

**Solution**:

**Step 1**: Find second derivative
$$f'(x) = 4x^3 - 12x + 3$$
$$f''(x) = 12x^2 - 12 = 12(x^2 - 1)$$

**Step 2**: Solve $f''(x) = 0$
$$12(x^2 - 1) = 0 \Rightarrow x = \pm 1$$

**Step 3**: Test for sign change

| Interval | Test Point | $f''$ | Concavity |
|----------|-----------|-------|-----------|
| $(-\infty, -1)$ | $x = -2$ | $f''(-2) = 36$ | Up |
| $(-1, 1)$ | $x = 0$ | $f''(0) = -12$ | Down |
| $(1, \infty)$ | $x = 2$ | $f''(2) = 36$ | Up |

**Inflection points**: $x = -1$ and $x = 1$ (concavity changes at both)

```python
def f_double_prime(x):
    return 12*x**2 - 12

# Find where f'' = 0
inflection_candidates = np.array([-1, 1])

for x_i in inflection_candidates:
    # Check sign change
    left = f_double_prime(x_i - 0.1)
    right = f_double_prime(x_i + 0.1)
    
    if left * right < 0:  # Different signs
        print(f"Inflection point at x = {x_i}")
```

### 7. Optimization Problems

**Steps for optimization**:
1. Define objective function $f(x)$
2. Find domain of $x$
3. Find critical points: solve $f'(x) = 0$
4. Check boundaries of domain
5. Evaluate $f$ at critical points and boundaries
6. Choose maximum or minimum as needed

#### Example 7: Real-World Optimization

A farmer has 1000 meters of fencing to enclose a rectangular field next to a river. No fence is needed along the river. What dimensions maximize the area?

**Solution**:

**Setup**:
- Let $x$ = width perpendicular to river
- Let $y$ = length parallel to river
- Constraint: $2x + y = 1000$ (amount of fencing)
- Objective: Maximize $A = xy$

**Step 1**: Express $A$ in terms of one variable
$$y = 1000 - 2x$$
$$A(x) = x(1000 - 2x) = 1000x - 2x^2$$

**Step 2**: Find domain
$$x > 0 \text{ and } y > 0 \Rightarrow 0 < x < 500$$

**Step 3**: Find critical points
$$A'(x) = 1000 - 4x = 0 \Rightarrow x = 250$$

**Step 4**: Verify maximum
$$A''(x) = -4 < 0$$ (concave down → maximum)

**Step 5**: Find dimensions
$$x = 250 \text{ m}, \quad y = 1000 - 2(250) = 500 \text{ m}$$

**Maximum area**: $A = 250 \times 500 = 125,000 \text{ m}^2$

```python
def area(x):
    return x * (1000 - 2*x)

# Find maximum
from scipy.optimize import minimize_scalar

result = minimize_scalar(lambda x: -area(x), bounds=(0, 500), method='bounded')
x_opt = result.x
y_opt = 1000 - 2*x_opt
max_area = area(x_opt)

print(f"Optimal width: {x_opt:.1f} m")
print(f"Optimal length: {y_opt:.1f} m")
print(f"Maximum area: {max_area:.0f} m²")

# Visualize
x_vals = np.linspace(0, 500, 1000)
areas = area(x_vals)

plt.figure(figsize=(10, 6))
plt.plot(x_vals, areas, 'b-', linewidth=2)
plt.plot(x_opt, max_area, 'ro', markersize=12, label=f'Maximum: ({x_opt:.0f}, {max_area:.0f})')
plt.xlabel('Width (m)', fontsize=12)
plt.ylabel('Area (m²)', fontsize=12)
plt.title('Area vs Width', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)
# plt.savefig('week07_optimization_example.png', dpi=150, bbox_inches='tight')
```

---

## Important Formulas

### Derivative Rules
$$\frac{d}{dx}x^n = nx^{n-1}$$
$$(f \pm g)' = f' \pm g'$$
$$(fg)' = f'g + fg'$$
$$\left(\frac{f}{g}\right)' = \frac{f'g - fg'}{g^2}$$
$$(f(g(x)))' = f'(g(x))g'(x)$$

### Common Derivatives
$$\frac{d}{dx}e^x = e^x$$
$$\frac{d}{dx}\ln x = \frac{1}{x}$$
$$\frac{d}{dx}\sin x = \cos x$$
$$\frac{d}{dx}\cos x = -\sin x$$

### Optimization
- **Critical point**: $f'(x) = 0$ or undefined
- **Local max**: $f''(x) < 0$
- **Local min**: $f''(x) > 0$
- **Inflection point**: $f''(x) = 0$ and changes sign

---

## Theorems & Proofs

### Theorem 1: Mean Value Theorem

**Statement**: If $f$ is continuous on $[a,b]$ and differentiable on $(a,b)$, then there exists $c \in (a,b)$ such that:
$$f'(c) = \frac{f(b) - f(a)}{b - a}$$

**Geometric interpretation**: There exists a point where the tangent is parallel to the secant line.

### Theorem 2: L'Hôpital's Rule

**Statement**: If $\lim_{x \to a} f(x) = \lim_{x \to a} g(x) = 0$ or $\pm\infty$, then:
$$\lim_{x \to a} \frac{f(x)}{g(x)} = \lim_{x \to a} \frac{f'(x)}{g'(x)}$$

(if the limit on the right exists)

### Theorem 3: Extreme Value Theorem

**Statement**: If $f$ is continuous on $[a,b]$, then $f$ attains both a maximum and minimum on $[a,b]$.

---

## Data Science Applications

### 1. Gradient Descent Foundations

The derivative is the foundation of gradient descent:

```python
def gradient_descent_1d(f, f_prime, x_init, learning_rate=0.1, iterations=100):
    """Minimize function using gradient descent"""
    x = x_init
    history = [x]
    
    for i in range(iterations):
        # Update: x_new = x_old - learning_rate * derivative
        x = x - learning_rate * f_prime(x)
        history.append(x)
    
    return x, history

# Example: minimize f(x) = x^2 - 4x + 5
def f(x):
    return x**2 - 4*x + 5

def f_prime(x):
    return 2*x - 4

x_min, history = gradient_descent_1d(f, f_prime, x_init=0, learning_rate=0.1)

print(f"Minimum found at x = {x_min:.4f}")
print(f"True minimum at x = 2 (from calculus)")

# Visualize convergence
plt.figure(figsize=(10, 6))
plt.plot(history, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=2, color='r', linestyle='--', label='True minimum')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('x value', fontsize=12)
plt.title('Gradient Descent Convergence', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
```

### 2. Newton's Method for Optimization

```python
def newtons_method(f, f_prime, f_double_prime, x_init, tol=1e-6, max_iter=100):
    """Find minimum using Newton's method"""
    x = x_init
    
    for i in range(max_iter):
        fx_prime = f_prime(x)
        fx_double_prime = f_double_prime(x)
        
        if abs(fx_prime) < tol:
            break
        
        # Newton update: x_new = x_old - f'(x)/f''(x)
        x = x - fx_prime / fx_double_prime
    
    return x

# Example
x_min = newtons_method(f, f_prime, lambda x: 2, x_init=0)
print(f"Newton's method found minimum at x = {x_min:.6f}")
```

### 3. Learning Rate Scheduling

```python
def adaptive_learning_rate(iteration, initial_lr=0.1, decay=0.95):
    """Exponential decay learning rate"""
    return initial_lr * (decay ** iteration)

# Visualize different schedules
iterations = np.arange(100)
constant = [0.1] * len(iterations)
exp_decay = [adaptive_learning_rate(i) for i in iterations]
step_decay = [0.1 if i < 30 else 0.01 if i < 60 else 0.001 for i in iterations]

plt.figure(figsize=(10, 6))
plt.plot(iterations, constant, label='Constant')
plt.plot(iterations, exp_decay, label='Exponential Decay')
plt.plot(iterations, step_decay, label='Step Decay')
plt.xlabel('Iteration', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Learning Rate Schedules', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.yscale('log')
```

### 4. Loss Function Behavior

```python
# Analyzing loss function landscape
def mse_loss(predictions, targets):
    return np.mean((predictions - targets)**2)

def mse_gradient(predictions, targets):
    return 2 * np.mean(predictions - targets)

# Generate data
np.random.seed(42)
true_weight = 3.5
X = np.linspace(0, 10, 50)
y_true = true_weight * X + np.random.randn(50) * 2

# Loss landscape
weights = np.linspace(0, 7, 100)
losses = [mse_loss(w * X, y_true) for w in weights]

plt.figure(figsize=(10, 6))
plt.plot(weights, losses, 'b-', linewidth=2)
plt.axvline(x=true_weight, color='r', linestyle='--', label=f'True weight = {true_weight}')
plt.xlabel('Weight', fontsize=12)
plt.ylabel('MSE Loss', fontsize=12)
plt.title('Loss Function Landscape', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
```

### 5. Finding Optimal Threshold

```python
from sklearn.metrics import accuracy_score

def find_optimal_threshold(y_true, y_proba):
    """Find threshold that maximizes accuracy"""
    thresholds = np.linspace(0, 1, 100)
    accuracies = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        acc = accuracy_score(y_true, y_pred)
        accuracies.append(acc)
    
    optimal_idx = np.argmax(accuracies)
    optimal_threshold = thresholds[optimal_idx]
    
    return optimal_threshold, thresholds, accuracies

# Simulated example
np.random.seed(42)
y_true = np.random.binomial(1, 0.6, 200)
y_proba = y_true * 0.7 + (1 - y_true) * 0.3 + np.random.randn(200) * 0.15
y_proba = np.clip(y_proba, 0, 1)

opt_thresh, thresholds, accuracies = find_optimal_threshold(y_true, y_proba)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, accuracies, 'b-', linewidth=2)
plt.axvline(x=opt_thresh, color='r', linestyle='--', label=f'Optimal = {opt_thresh:.3f}')
plt.xlabel('Threshold', fontsize=12)
plt.ylabel('Accuracy', fontsize=12)
plt.title('Accuracy vs Classification Threshold', fontsize=14)
plt.grid(True, alpha=0.3)
plt.legend()
```

---

## Common Pitfalls

### Pitfall 1: Forgetting Chain Rule

**Wrong**: $\frac{d}{dx}(x^2 + 1)^3 = 3(x^2 + 1)^2$
**Correct**: $\frac{d}{dx}(x^2 + 1)^3 = 3(x^2 + 1)^2 \cdot 2x$ ✓

### Pitfall 2: Second Derivative Test Inconclusive

When $f''(x) = 0$, use first derivative test instead.

### Pitfall 3: Not Checking Domain Boundaries

Always evaluate function at boundary points of domain!

### Pitfall 4: Confusing Critical Points with Extrema

Not all critical points are extrema (could be saddle points).

### Pitfall 5: Wrong Sign in Quotient Rule

Numerator is $f'g - fg'$ (not $fg' - f'g$)

---

## Practice Problems

### Basic Level

1. Find derivative of $f(x) = 3x^5 - 2x^3 + 7x - 1$
2. Compute $\frac{d}{dx}(e^{2x}\sin x)$
3. Find critical points of $f(x) = x^3 - 3x + 2$
4. Use second derivative test on $f(x) = x^4 - 4x^2$
5. Find inflection points of $f(x) = x^3 - 3x^2 + 4$

### Intermediate Level

6. A box with square base and open top must have volume 32 m³. Find dimensions that minimize surface area.
7. Find global maximum and minimum of $f(x) = x^3 - 3x^2 + 1$ on $[-1, 3]$
8. Prove using calculus: $e^x \geq 1 + x$ for all $x$
9. Find equation of tangent line to $y = \ln(x^2 + 1)$ at $x = 1$
10. Use L'Hôpital's rule: $\lim_{x \to 0} \frac{\sin x - x}{x^3}$

### Advanced Level

11. A ladder 10m long rests against a vertical wall. If bottom slides away at 1 m/s, how fast is top sliding down when bottom is 6m from wall?
12. Find point on parabola $y = x^2$ closest to point $(3, 0)$
13. Prove Mean Value Theorem geometrically
14. Show that $f(x) = x^5 + 2x^3 + x - 1$ has exactly one real root
15. Implement bisection method to find roots numerically

---

## Self-Assessment Checklist

- [ ] Can you compute derivatives using all differentiation rules?
- [ ] Can you find and classify critical points?
- [ ] Do you understand first and second derivative tests?
- [ ] Can you identify concavity and inflection points?
- [ ] Can you solve optimization problems?
- [ ] Do you understand the connection to gradient descent?
- [ ] Can you apply derivatives to real-world problems?

---

## Key Takeaways

1. **Derivative measures rate of change**: Foundation of calculus and optimization
2. **Critical points where $f'(x) = 0$**: Potential extrema locations
3. **Second derivative reveals concavity**: Helps classify extrema
4. **Optimization requires calculus**: Find best solutions to real problems
5. **Gradient descent uses derivatives**: Core algorithm in machine learning
6. **Understanding behavior matters**: Not just finding answer, but understanding why

---

## References

- **Textbook**: Stewart - *Calculus: Early Transcendentals*
- **Videos**: Week 7 lectures - Single Variable Calculus
- **Online**: Khan Academy Calculus

---

## Connection to Next Week

Week 8 extends to **multivariable calculus**:
- **Partial derivatives**: Derivatives with respect to multiple variables
- **Gradients**: Vector of partial derivatives
- **Chain rule in multiple dimensions**

Single-variable derivatives are the foundation!

---

**Last Updated**: 2025-11-22
**Next Week**: Multivariable Calculus - Partial Derivatives and Gradients
