# Week 8: Multivariable Calculus - Partial Derivatives

---
Date: 2025-11-22
Course: BSMA1003 - Mathematics for Data Science II
Level: Foundation
Week: 8 of 11
Source: IIT Madras Mathematics II Week 8
Topic Area: Multivariable Calculus
Tags: #BSMA1003 #Calculus #Week8 #PartialDerivatives #Gradient #MultivariableCalculus #Foundation
---

## Overview (BLUF)

**One-sentence definition**: Partial derivatives measure how a multivariable function changes when we vary one variable at a time while holding all others constant, forming the foundation for optimization in machine learning.

**Why it matters**: Almost every machine learning algorithm involves optimizing a loss function with multiple parameters (weights, biases). Understanding partial derivatives and gradients is essential for gradient descent, backpropagation in neural networks, and multivariate regression. Without this knowledge, you cannot understand how models learn from data.

**When to use**: Whenever you need to find optimal values for multiple parameters simultaneously (linear regression with many features, neural network training, support vector machines), analyze sensitivity of outputs to input changes (feature importance), or understand how systems change in high-dimensional spaces (data science, economics, physics).

**Prerequisites**: Single-variable derivatives ([week-07-single-variable-calculus.md](week-07-single-variable-calculus.md)), basic linear algebra (vectors, dot products from [week-01-vectors-matrices-intro.md](week-01-vectors-matrices-intro.md)), limits and continuity concepts. Must understand what a derivative means geometrically (slope of tangent line) and algebraically (rate of change).

---

## Core Theory

### 1. Functions of Multiple Variables

**Intuitive explanation**: Imagine standing on a mountainside. Your elevation depends on two coordinates: how far north you are (x) and how far east you are (y). This is a function of two variables: $z = f(x, y)$. Unlike single-variable functions that are curves on a flat plane, multivariable functions create surfaces in 3D space.

**Formal definition**: A function of $n$ variables is a rule that assigns each ordered $n$-tuple $(x_1, x_2, \ldots, x_n)$ in its domain $D \subseteq \mathbb{R}^n$ to a unique real number:

$$f: \mathbb{R}^n \rightarrow \mathbb{R}$$

**Common examples**:
- **Two variables**: Temperature $T(x, y)$ at position $(x, y)$ on a metal plate
- **Three variables**: Air pressure $P(x, y, z)$ at point $(x, y, z)$ in atmosphere
- **Many variables**: Machine learning loss function $L(w_1, w_2, \ldots, w_n)$ depending on n weights

**Everyday analogy**: Think of your monthly expenses as $E(g, f, r)$ where $g$ = groceries, $f$ = fuel, $r$ = rent. Change one category while keeping others fixed to see its impact.

**Visualizing multivariable functions**:

1. **Level curves (contour plots)**: For $z = f(x, y)$, draw curves where $f(x, y) = c$ for constants $c$
   - Like topographic map lines connecting points of equal elevation
   - Closer lines = steeper slope
   - Example: $f(x, y) = x^2 + y^2$ has circular level curves

2. **3D surface plots**: Graph of $(x, y, f(x, y))$ forms a surface
   - Paraboloid: $z = x^2 + y^2$ (bowl shape, opens upward)
   - Saddle: $z = x^2 - y^2$ (horse saddle shape)
   - Plane: $z = 2x + 3y + 1$ (flat tilted surface)

3. **Cross sections**: Fix one variable, see curve in remaining variables
   - Fix $y = 0$ in $z = x^2 + y^2$ → parabola $z = x^2$
   - Fix $x = 2$ → parabola $z = 4 + y^2$

**Data science connection**: In machine learning, a model with 100 features has a 100-dimensional parameter space. We can't visualize it directly, but the same principles apply - we're finding optimal points on high-dimensional surfaces.

### 2. Partial Derivatives - The Foundation

**Intuitive explanation**: Standing on that mountainside, you want to know: "If I walk directly east (changing only x), how steep is the climb?" That's the partial derivative with respect to x. Then ask: "If I walk directly north (changing only y), how steep?" That's the partial derivative with respect to y.

**Formal definition**: The **partial derivative** of $f(x, y)$ with respect to $x$ at point $(a, b)$ is:

$$\frac{\partial f}{\partial x}(a, b) = \lim_{h \to 0} \frac{f(a+h, b) - f(a, b)}{h}$$

Similarly for y:

$$\frac{\partial f}{\partial y}(a, b) = \lim_{h \to 0} \frac{f(a, b+h) - f(a, b)}{h}$$

**Key insight**: To find $\frac{\partial f}{\partial x}$, treat all variables except $x$ as constants and use regular single-variable derivative rules.

**Notation** (all equivalent):
- $\frac{\partial f}{\partial x}$, $f_x$, $\partial_x f$, $D_x f$, $D_1 f$

**Geometric interpretation**:
- $f_x(a, b)$ = slope of curve formed by intersecting surface $z = f(x, y)$ with vertical plane $y = b$
- $f_y(a, b)$ = slope of curve formed by intersecting surface $z = f(x, y)$ with vertical plane $x = a$

**Everyday analogy**: Your car's fuel efficiency $E(s, w)$ depends on speed $s$ and weight $w$. The partial derivative $\frac{\partial E}{\partial s}$ tells you how efficiency changes with speed (holding weight fixed). The partial derivative $\frac{\partial E}{\partial w}$ tells you how efficiency changes with weight (holding speed fixed).

---

## Practical Implementation

### Computing Partial Derivatives - Step by Step

**Rule**: To compute $\frac{\partial f}{\partial x_i}$, treat all variables except $x_i$ as constants and differentiate using standard rules.
#### Example 1: Basic Polynomial Function

**Function**: $f(x, y) = x^3 + 2x^2y + y^2$

**Find**: $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$

**Solution for** $\frac{\partial f}{\partial x}$:

Treat $y$ as a constant. Looking at each term:
- $\frac{\partial}{\partial x}(x^3) = 3x^2$ (power rule)
- $\frac{\partial}{\partial x}(2x^2y) = 4xy$ (y is constant, so $2y$ is just a coefficient)
- $\frac{\partial}{\partial x}(y^2) = 0$ (y is constant, derivative of constant is zero)

**Answer**:
$$\frac{\partial f}{\partial x} = 3x^2 + 4xy$$

**Solution for** $\frac{\partial f}{\partial y}$:

Treat $x$ as a constant. Looking at each term:
- $\frac{\partial}{\partial y}(x^3) = 0$ (x is constant)
- $\frac{\partial}{\partial y}(2x^2y) = 2x^2$ (x² is constant, derivative of $cy$ is $c$)
- $\frac{\partial}{\partial y}(y^2) = 2y$ (power rule)

**Answer**:
$$\frac{\partial f}{\partial y} = 2x^2 + 2y$$

**Verify at point (1, 2)**:
$$f_x(1, 2) = 3(1)^2 + 4(1)(2) = 3 + 8 = 11$$
$$f_y(1, 2) = 2(1)^2 + 2(2) = 2 + 4 = 6$$

**Interpretation**: At point (1, 2), if we move in the positive x-direction, the function increases at rate 11. If we move in the positive y-direction, it increases at rate 6.

```python
import numpy as np
from scipy.misc import derivative

# Define the function
def f(x, y):
    return x**3 + 2*x**2*y + y**2

# Partial derivative with respect to x (hold y constant)
def f_x_numerical(x, y, h=1e-5):
    return (f(x+h, y) - f(x, y)) / h

# Partial derivative with respect to y (hold x constant)
def f_y_numerical(x, y, h=1e-5):
    return (f(x, y+h) - f(x, y)) / h

# Test at point (1, 2)
x, y = 1, 2
print(f"Numerical f_x(1, 2) = {f_x_numerical(x, y):.6f}")  # Should be ~11
print(f"Numerical f_y(1, 2) = {f_y_numerical(x, y):.6f}")  # Should be ~6

# Analytical (exact) formulas
def f_x_analytical(x, y):
    return 3*x**2 + 4*x*y

def f_y_analytical(x, y):
    return 2*x**2 + 2*y

print(f"Analytical f_x(1, 2) = {f_x_analytical(x, y)}")  # Exactly 11
print(f"Analytical f_y(1, 2) = {f_y_analytical(x, y)}")  # Exactly 6
```

#### Example 2: Exponential and Trigonometric Functions

**Function**: $f(x, y) = e^{xy}\sin(x)$

**Find**: $\frac{\partial f}{\partial x}$ and $\frac{\partial f}{\partial y}$

**Solution for** $\frac{\partial f}{\partial x}$:

This requires product rule: $\frac{\partial}{\partial x}(uv) = \frac{\partial u}{\partial x}v + u\frac{\partial v}{\partial x}$

Let $u = e^{xy}$ and $v = \sin(x)$

For $\frac{\partial u}{\partial x}$: Use chain rule. Let $g = xy$, then $u = e^g$
- $\frac{\partial g}{\partial x} = y$ (x differentiates to 1, y is constant)
- $\frac{du}{dg} = e^g$
- $\frac{\partial u}{\partial x} = e^g \cdot y = ye^{xy}$

For $\frac{\partial v}{\partial x}$: Standard derivative
- $\frac{\partial v}{\partial x} = \cos(x)$

**Combining**:
$$\frac{\partial f}{\partial x} = ye^{xy}\sin(x) + e^{xy}\cos(x) = e^{xy}(y\sin(x) + \cos(x))$$

**Solution for** $\frac{\partial f}{\partial y}$:

For $e^{xy}$: Chain rule gives $xe^{xy}$
For $\sin(x)$: Treating x as constant gives 0

$$\frac{\partial f}{\partial y} = xe^{xy}\sin(x)$$

**Why this matters**: Exponential and trig functions appear in activation functions (sigmoid, tanh), Fourier analysis, and signal processing in ML.

#### Example 3: Rational Functions

**Function**: $f(x, y) = \frac{x^2 - y^2}{x^2 + y^2}$

**Find**: $\frac{\partial f}{\partial x}$

**Solution**: Use quotient rule: $\frac{\partial}{\partial x}\left(\frac{u}{v}\right) = \frac{\frac{\partial u}{\partial x}v - u\frac{\partial v}{\partial x}}{v^2}$

Let $u = x^2 - y^2$ and $v = x^2 + y^2$

$\frac{\partial u}{\partial x} = 2x$ (treat y as constant)

$\frac{\partial v}{\partial x} = 2x$ (treat y as constant)

$$\frac{\partial f}{\partial x} = \frac{(2x)(x^2 + y^2) - (x^2 - y^2)(2x)}{(x^2 + y^2)^2}$$

Expand numerator:
$$= \frac{2x^3 + 2xy^2 - 2x^3 + 2xy^2}{(x^2 + y^2)^2} = \frac{4xy^2}{(x^2 + y^2)^2}$$

Similarly, $\frac{\partial f}{\partial y} = \frac{-4x^2y}{(x^2 + y^2)^2}$

**Key observation**: Both partial derivatives are zero at the origin (when both are defined). This could be a critical point!

---

### 3. Higher-Order Partial Derivatives

**Concept**: Since partial derivatives are themselves functions, we can differentiate them again.

**Second-order partial derivatives**:

$$f_{xx} = \frac{\partial^2 f}{\partial x^2} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial x}\right)$$

$$f_{yy} = \frac{\partial^2 f}{\partial y^2} = \frac{\partial}{\partial y}\left(\frac{\partial f}{\partial y}\right)$$

**Mixed partial derivatives** (differentiate with respect to different variables):

$$f_{xy} = \frac{\partial^2 f}{\partial y \partial x} = \frac{\partial}{\partial y}\left(\frac{\partial f}{\partial x}\right)$$

$$f_{yx} = \frac{\partial^2 f}{\partial x \partial y} = \frac{\partial}{\partial x}\left(\frac{\partial f}{\partial y}\right)$$

**Clairaut's Theorem (Schwarz's Theorem)**: If $f_{xy}$ and $f_{yx}$ are both continuous, then:
$$f_{xy} = f_{yx}$$

**Why it matters**: Order of differentiation doesn't matter for "nice" (continuous) functions. This is crucial for verifying calculations and understanding optimization (the Hessian matrix is symmetric).

#### Example 4: Computing All Second-Order Derivatives

**Function**: $f(x, y) = x^3y^2 + x^2y^3$

**Step 1**: Find first-order derivatives

$$f_x = 3x^2y^2 + 2xy^3$$
$$f_y = 2x^3y + 3x^2y^2$$

**Step 2**: Find second-order derivatives

$$f_{xx} = \frac{\partial}{\partial x}(3x^2y^2 + 2xy^3) = 6xy^2 + 2y^3$$

$$f_{yy} = \frac{\partial}{\partial y}(2x^3y + 3x^2y^2) = 2x^3 + 6x^2y$$

$$f_{xy} = \frac{\partial}{\partial y}(3x^2y^2 + 2xy^3) = 6x^2y + 6xy^2$$

$$f_{yx} = \frac{\partial}{\partial x}(2x^3y + 3x^2y^2) = 6x^2y + 6xy^2$$

**Verification**: $f_{xy} = f_{yx}$ ✓ (Clairaut's theorem holds)

**Physical interpretation**:
- $f_{xx}$: How the rate of change in x-direction changes as x varies (concavity in x)
- $f_{yy}$: How the rate of change in y-direction changes as y varies (concavity in y)
- $f_{xy}$: How the rate of change in x-direction changes as y varies (interaction effect)

#### Example 5: Functions of Three Variables

**Function**: $f(x, y, z) = xyz + x^2 + y^2 + z^2$

**First-order partials**:
$$f_x = yz + 2x$$
$$f_y = xz + 2y$$
$$f_z = xy + 2z$$

**Second-order partials**:
$$f_{xx} = 2, \quad f_{yy} = 2, \quad f_{zz} = 2$$
$$f_{xy} = z, \quad f_{xz} = y, \quad f_{yz} = x$$

**Observations**:
- Pure second derivatives are constant (linear functions have constant curvature)
- Mixed partials show how variables interact
- All mixed partials obey Clairaut's theorem

---

## Gradient Vectors

### 4. The Gradient - Combining All Partial Derivatives

**Intuitive explanation**: Instead of asking "How steep is it in the x-direction?" and "How steep is it in the y-direction?" separately, the gradient combines both into a single vector that points in the direction of steepest increase.

**Definition**: The **gradient** of $f$ is a vector of all first-order partial derivatives:

$$\nabla f = \left\langle \frac{\partial f}{\partial x_1}, \frac{\partial f}{\partial x_2}, \ldots, \frac{\partial f}{\partial x_n} \right\rangle$$

For two variables:
$$\nabla f(x, y) = \left\langle \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right\rangle = f_x\mathbf{i} + f_y\mathbf{j}$$

**Notation**: $\nabla$ is "nabla" or "del" operator

**Key Properties**:

1. **Direction of steepest ascent**: $\nabla f$ points where $f$ increases fastest
2. **Magnitude**: $|\nabla f|$ is the maximum rate of increase
3. **Perpendicular to level curves**: At any point, $\nabla f \perp$ level curve through that point

**Everyday analogy**: Imagine a ball on a hillside. The gradient tells you which way the ball will roll (steepest descent) if you let it go. The magnitude tells you how fast it will initially accelerate.

#### Example 6: Computing and Interpreting Gradients

**Function**: $f(x, y) = x^2 + 3xy + y^2$

**Step 1**: Find partial derivatives
$$f_x = 2x + 3y$$
$$f_y = 3x + 2y$$

**Step 2**: Write gradient vector
$$\nabla f(x, y) = \langle 2x + 3y, 3x + 2y \rangle$$

**Step 3**: Evaluate at specific point, say (1, 2)
$$\nabla f(1, 2) = \langle 2(1) + 3(2), 3(1) + 2(2) \rangle = \langle 8, 7 \rangle$$

**Interpretation**:
- At point (1, 2), if you move in direction $\langle 8, 7 \rangle$, the function increases fastest
- The magnitude $|\nabla f(1, 2)| = \sqrt{64 + 49} = \sqrt{113} \approx 10.63$ is the maximum rate of increase
- Moving opposite to gradient ($\langle -8, -7 \rangle$) gives steepest descent

```python
import numpy as np
import matplotlib.pyplot as plt

# Define function and gradient
def f(x, y):
    return x**2 + 3*x*y + y**2

def grad_f(x, y):
    fx = 2*x + 3*y
    fy = 3*x + 2*y
    return np.array([fx, fy])

# Create mesh for visualization
x = np.linspace(-2, 4, 100)
y = np.linspace(-2, 4, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Plot contours and gradient vector at (1, 2)
plt.figure(figsize=(10, 8))
contours = plt.contour(X, Y, Z, levels=15, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)

# Point of interest
point = np.array([1, 2])
grad = grad_f(1, 2)

# Plot gradient vector
plt.quiver(point[0], point[1], grad[0], grad[1],
           color='red', scale=20, width=0.01,
           label='Gradient (steepest ascent)')
plt.plot(point[0], point[1], 'ro', markersize=10, label='Point (1,2)')

plt.xlabel('x')
plt.ylabel('y')
plt.title('Gradient Vector Shows Direction of Steepest Increase')
plt.legend()
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()

print(f"Gradient at (1, 2): {grad}")
print(f"Magnitude: {np.linalg.norm(grad):.4f}")
```

**Data Science Application - Gradient Descent**:

In machine learning, we want to **minimize** a loss function $L(w_1, w_2, \ldots, w_n)$. The gradient $\nabla L$ points uphill, so we move in the **opposite direction**:

$$\mathbf{w}^{(new)} = \mathbf{w}^{(old)} - \alpha \nabla L(\mathbf{w}^{(old)})$$

where $\alpha$ is the learning rate.

#### Example 7: Gradient in Three Dimensions

**Function**: $f(x, y, z) = e^x\cos(y) + z^2$

**Gradient**:
$$\nabla f = \left\langle \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}, \frac{\partial f}{\partial z} \right\rangle$$

$$\nabla f = \langle e^x\cos(y), -e^x\sin(y), 2z \rangle$$

**At point (0, 0, 1)**:
$$\nabla f(0, 0, 1) = \langle e^0\cos(0), -e^0\sin(0), 2(1) \rangle = \langle 1, 0, 2 \rangle$$

**Interpretation**: From point (0, 0, 1), the function increases fastest in direction $\langle 1, 0, 2 \rangle$.

---

### 5. Directional Derivatives

**Motivation**: Partial derivatives tell us rate of change along coordinate axes (x, y, z directions). But what if we want rate of change in an arbitrary direction, like northeast at 45°?

**Definition**: The **directional derivative** of $f$ at point $(x_0, y_0)$ in direction of unit vector $\mathbf{u} = \langle a, b \rangle$ is:

$$D_{\mathbf{u}}f(x_0, y_0) = \lim_{h \to 0} \frac{f(x_0 + ha, y_0 + hb) - f(x_0, y_0)}{h}$$

**CRITICAL**: $\mathbf{u}$ must be a **unit vector** ($|\mathbf{u}| = 1$)

**Theorem**: If $f$ is differentiable:
$$D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u}$$

**This is huge!** The directional derivative is just the dot product of the gradient with the direction vector.

**Key Results**:

1. **Maximum rate of increase**: When $\mathbf{u}$ is parallel to $\nabla f$
   $$\max(D_{\mathbf{u}}f) = |\nabla f|$$

2. **Maximum rate of decrease**: When $\mathbf{u}$ is opposite to $\nabla f$
   $$\min(D_{\mathbf{u}}f) = -|\nabla f|$$

3. **No change**: When $\mathbf{u} \perp \nabla f$ (perpendicular)
   $$D_{\mathbf{u}}f = 0$$

#### Example 8: Computing Directional Derivatives

**Function**: $f(x, y) = x^2 - xy + y^2$

**Task**: Find rate of change at point (1, 1) in direction of vector $\mathbf{v} = \langle 3, 4 \rangle$

**Step 1**: Find gradient
$$\nabla f = \langle 2x - y, -x + 2y \rangle$$

At (1, 1):
$$\nabla f(1, 1) = \langle 2(1) - 1, -1 + 2(1) \rangle = \langle 1, 1 \rangle$$

**Step 2**: Normalize direction vector
$$\mathbf{u} = \frac{\mathbf{v}}{|\mathbf{v}|} = \frac{\langle 3, 4 \rangle}{\sqrt{9 + 16}} = \frac{\langle 3, 4 \rangle}{5} = \left\langle \frac{3}{5}, \frac{4}{5} \right\rangle$$

**Step 3**: Compute directional derivative
$$D_{\mathbf{u}}f(1, 1) = \nabla f(1, 1) \cdot \mathbf{u}$$
$$= \langle 1, 1 \rangle \cdot \left\langle \frac{3}{5}, \frac{4}{5} \right\rangle$$
$$= 1 \cdot \frac{3}{5} + 1 \cdot \frac{4}{5} = \frac{7}{5} = 1.4$$

**Interpretation**: Moving from (1, 1) in direction $\langle 3, 4 \rangle$, the function increases at rate 1.4 per unit distance.

#### Example 9: Maximum Rate of Change

**Function**: $T(x, y) = 100 - x^2 - y^2$ (temperature distribution)

**Task**: At point (3, 4), find:
- Direction of maximum temperature increase
- Maximum rate of increase
- Direction of maximum cooling (decrease)

**Solution**:

**Step 1**: Find gradient
$$\nabla T = \langle -2x, -2y \rangle$$

At (3, 4):
$$\nabla T(3, 4) = \langle -6, -8 \rangle$$

**Step 2**: Maximum rate of increase
Direction: $\nabla T(3, 4) = \langle -6, -8 \rangle$ (or normalized: $\langle -0.6, -0.8 \rangle$)
Rate: $|\nabla T(3, 4)| = \sqrt{36 + 64} = 10$

**Step 3**: Maximum rate of decrease (cooling)
Direction: Opposite to gradient: $\langle 6, 8 \rangle$ (or normalized: $\langle 0.6, 0.8 \rangle$)
Rate: $-|\nabla T(3, 4)| = -10$

**Practical meaning**: At point (3, 4), moving in direction $\langle 6, 8 \rangle$ (roughly northeast) cools fastest at 10 degrees per unit distance.

```python
# Verify maximum vs other directions
point = np.array([3, 4])
grad_T = np.array([-6, -8])

# Direction of steepest descent (cooling)
max_cooling_dir = -grad_T / np.linalg.norm(grad_T)  # Normalized

# Test various directions
angles = np.linspace(0, 2*np.pi, 100)
rates = []

for angle in angles:
    direction = np.array([np.cos(angle), np.sin(angle)])  # Unit vector
    rate = np.dot(grad_T, direction)  # Directional derivative
    rates.append(rate)

plt.figure(figsize=(10, 6))
plt.plot(np.degrees(angles), rates, linewidth=2)
plt.axhline(y=np.linalg.norm(grad_T), color='r', linestyle='--',
            label=f'Max increase: {np.linalg.norm(grad_T):.1f}')
plt.axhline(y=-np.linalg.norm(grad_T), color='b', linestyle='--',
            label=f'Max decrease: {-np.linalg.norm(grad_T):.1f}')
plt.xlabel('Direction (degrees)')
plt.ylabel('Rate of Change')
plt.title('Temperature Change Rate vs Direction at (3,4)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()
```

---

### 6. Chain Rule for Multivariable Functions

**The problem**: If $z = f(x, y)$ but $x$ and $y$ themselves depend on other variables, how do we find derivatives?

**Everyday analogy**: Your happiness $H$ depends on sleep $s$ and coffee $c$. But both sleep and coffee depend on time of day $t$. How does your happiness change over time?

#### Case 1: One Independent Variable

If $z = f(x, y)$ where $x = x(t)$ and $y = y(t)$:

$$\frac{dz}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$$

**In gradient notation**:
$$\frac{dz}{dt} = \nabla f \cdot \left\langle \frac{dx}{dt}, \frac{dy}{dt} \right\rangle$$

**Why this makes sense**: As $t$ changes, both $x$ and $y$ change. The total change in $z$ is the sum of:
- Change in $z$ due to change in $x$: $\frac{\partial f}{\partial x}\frac{dx}{dt}$
- Change in $z$ due to change in $y$: $\frac{\partial f}{\partial y}\frac{dy}{dt}$

#### Example 10: Chain Rule with One Parameter

**Given**: $z = x^2 + y^2$ where $x = t^2$ and $y = t^3$

**Find**: $\frac{dz}{dt}$

**Method 1 - Direct substitution**:
$$z = (t^2)^2 + (t^3)^2 = t^4 + t^6$$
$$\frac{dz}{dt} = 4t^3 + 6t^5$$

**Method 2 - Chain rule**:
$$\frac{\partial z}{\partial x} = 2x, \quad \frac{\partial z}{\partial y} = 2y$$
$$\frac{dx}{dt} = 2t, \quad \frac{dy}{dt} = 3t^2$$

$$\frac{dz}{dt} = 2x(2t) + 2y(3t^2) = 4xt + 6yt^2$$

Substitute $x = t^2$ and $y = t^3$:
$$= 4(t^2)(t) + 6(t^3)(t^2) = 4t^3 + 6t^5$$ ✓

**Both methods agree!** Chain rule is powerful when direct substitution is messy.

#### Case 2: Multiple Independent Variables

If $z = f(x, y)$ where $x = x(s, t)$ and $y = y(s, t)$:

$$\frac{\partial z}{\partial s} = \frac{\partial f}{\partial x}\frac{\partial x}{\partial s} + \frac{\partial f}{\partial y}\frac{\partial y}{\partial s}$$

$$\frac{\partial z}{\partial t} = \frac{\partial f}{\partial x}\frac{\partial x}{\partial t} + \frac{\partial f}{\partial y}\frac{\partial y}{\partial t}$$

#### Example 11: Chain Rule with Two Parameters

**Given**: $z = e^{xy}$ where $x = s^2 + t^2$ and $y = s - t$

**Find**: $\frac{\partial z}{\partial s}$ and $\frac{\partial z}{\partial t}$

**Step 1**: Partial derivatives of $z$
$$\frac{\partial z}{\partial x} = ye^{xy}, \quad \frac{\partial z}{\partial y} = xe^{xy}$$

**Step 2**: Partial derivatives of $x$ and $y$
$$\frac{\partial x}{\partial s} = 2s, \quad \frac{\partial x}{\partial t} = 2t$$
$$\frac{\partial y}{\partial s} = 1, \quad \frac{\partial y}{\partial t} = -1$$

**Step 3**: Apply chain rule for $\frac{\partial z}{\partial s}$
$$\frac{\partial z}{\partial s} = ye^{xy}(2s) + xe^{xy}(1)$$
$$= e^{xy}(2sy + x)$$

**Step 4**: Apply chain rule for $\frac{\partial z}{\partial t}$
$$\frac{\partial z}{\partial t} = ye^{xy}(2t) + xe^{xy}(-1)$$
$$= e^{xy}(2ty - x)$$

**Neural Network Connection**: Backpropagation is repeated application of the chain rule! When training a deep network, gradients flow backward through layers using this principle.

---

### 7. Tangent Planes and Linear Approximations

**Single-variable review**: Tangent line at $(a, f(a))$ is:
$$y = f(a) + f'(a)(x - a)$$

This is the best linear approximation to the curve near $x = a$.

**Multivariable extension**: For surface $z = f(x, y)$, the **tangent plane** at $(x_0, y_0, f(x_0, y_0))$ is:

$$z = f(x_0, y_0) + f_x(x_0, y_0)(x - x_0) + f_y(x_0, y_0)(y - y_0)$$

**Geometric meaning**: This plane "kisses" the surface at the point—it's the best flat approximation to the curved surface nearby.

**Linear approximation**: For points near $(x_0, y_0)$:
$$f(x, y) \approx f(x_0, y_0) + f_x(x_0, y_0)(x - x_0) + f_y(x_0, y_0)(y - y_0)$$

This is called the **linearization** of $f$ at $(x_0, y_0)$.

#### Example 12: Finding a Tangent Plane

**Function**: $f(x, y) = x^2 + 2y^2$

**Task**: Find tangent plane at point $(1, 1, 3)$

**Step 1**: Verify point is on surface
$$f(1, 1) = 1^2 + 2(1)^2 = 3$$ ✓

**Step 2**: Compute partial derivatives
$$f_x = 2x, \quad f_y = 4y$$

At $(1, 1)$:
$$f_x(1, 1) = 2, \quad f_y(1, 1) = 4$$

**Step 3**: Write tangent plane equation
$$z - 3 = 2(x - 1) + 4(y - 1)$$
$$z = 2x + 4y - 3$$

**Verification**: Plane passes through $(1, 1, 3)$?
$$z = 2(1) + 4(1) - 3 = 3$$ ✓

#### Example 13: Using Linear Approximation

**Function**: $f(x, y) = \sqrt{x^2 + y^2}$

**Task**: Approximate $f(3.02, 3.97)$ using linearization at $(3, 4)$

**Step 1**: Base point value
$$f(3, 4) = \sqrt{9 + 16} = 5$$

**Step 2**: Partial derivatives
$$f_x = \frac{x}{\sqrt{x^2 + y^2}}, \quad f_y = \frac{y}{\sqrt{x^2 + y^2}}$$

At $(3, 4)$:
$$f_x(3, 4) = \frac{3}{5}, \quad f_y(3, 4) = \frac{4}{5}$$

**Step 3**: Linear approximation
$$L(x, y) = 5 + \frac{3}{5}(x - 3) + \frac{4}{5}(y - 4)$$

$$f(3.02, 3.97) \approx L(3.02, 3.97)$$
$$= 5 + \frac{3}{5}(0.02) + \frac{4}{5}(-0.03)$$
$$= 5 + 0.012 - 0.024 = 4.988$$

**Exact value**: $\sqrt{(3.02)^2 + (3.97)^2} \approx 4.9841$

**Error**: $|4.988 - 4.9841| = 0.0039$ (very small—approximation is excellent!)

```python
# Compare approximation quality
def f(x, y):
    return np.sqrt(x**2 + y**2)

# Base point and linearization
x0, y0 = 3, 4
f0 = f(x0, y0)
fx0 = x0 / f0  # 3/5
fy0 = y0 / f0  # 4/5

def linearization(x, y):
    return f0 + fx0*(x - x0) + fy0*(y - y0)

# Test points near (3, 4)
test_points = [(3.02, 3.97), (3.1, 4.1), (2.9, 3.8)]

for x, y in test_points:
    exact = f(x, y)
    approx = linearization(x, y)
    error = abs(exact - approx)
    print(f"f({x}, {y}) = {exact:.6f}, approx = {approx:.6f}, error = {error:.6f}")
```

**Why this matters in ML**: Many optimization algorithms use linear approximations around the current point to decide where to step next (Newton's method, quasi-Newton methods).

---

## Data Science Applications

### 1. Gradient Descent - The Heart of Machine Learning

**Problem**: Minimize loss function $L(w_1, w_2, \ldots, w_n)$ with $n$ parameters.

**Algorithm**:
1. Start with initial weights $\mathbf{w}^{(0)}$
2. Compute gradient $\nabla L(\mathbf{w}^{(k)})$ at current position
3. Update: $\mathbf{w}^{(k+1)} = \mathbf{w}^{(k)} - \alpha \nabla L(\mathbf{w}^{(k)})$
4. Repeat until convergence

**Why it works**: Gradient points uphill, so $-\nabla L$ points downhill toward minimum.

**Learning rate $\alpha$**: Controls step size
- Too large: Overshoot, oscillate, diverge
- Too small: Slow convergence
- Adaptive methods (Adam, RMSprop) adjust $\alpha$ automatically

**Example**: Linear regression with 2 features

$$L(w_0, w_1) = \frac{1}{m}\sum_{i=1}^{m}(y_i - (w_0 + w_1x_i))^2$$

Partial derivatives:
$$\frac{\partial L}{\partial w_0} = -\frac{2}{m}\sum_{i=1}^{m}(y_i - (w_0 + w_1x_i))$$
$$\frac{\partial L}{\partial w_1} = -\frac{2}{m}\sum_{i=1}^{m}(y_i - (w_0 + w_1x_i))x_i$$

```python
import numpy as np

# Simple gradient descent for linear regression
def gradient_descent_linear_reg(X, y, learning_rate=0.01, iterations=1000):
    m = len(y)
    w0, w1 = 0, 0  # Initial weights
    history = []

    for i in range(iterations):
        # Predictions
        y_pred = w0 + w1 * X

        # Compute gradients
        dw0 = -(2/m) * np.sum(y - y_pred)
        dw1 = -(2/m) * np.sum((y - y_pred) * X)

        # Update weights
        w0 -= learning_rate * dw0
        w1 -= learning_rate * dw1

        # Compute loss
        loss = (1/m) * np.sum((y - y_pred)**2)
        history.append(loss)

    return w0, w1, history

# Example data
X = np.array([1, 2, 3, 4, 5])
y = np.array([2, 4, 5, 4, 5])

w0, w1, history = gradient_descent_linear_reg(X, y)
print(f"Final weights: w0={w0:.4f}, w1={w1:.4f}")

# Plot convergence
plt.plot(history)
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Gradient Descent Convergence')
plt.show()
```

### 2. Backpropagation in Neural Networks

**The challenge**: Neural network has millions of parameters. How to compute $\frac{\partial L}{\partial w_{ij}^{(l)}}$ for every weight in every layer?

**Answer**: Chain rule! Gradients flow backward through network.

For weight $w_{ij}^{(l)}$ connecting neuron $j$ in layer $l-1$ to neuron $i$ in layer $l$:

$$\frac{\partial L}{\partial w_{ij}^{(l)}} = \frac{\partial L}{\partial z_i^{(l)}} \cdot \frac{\partial z_i^{(l)}}{\partial w_{ij}^{(l)}}$$

where $z_i^{(l)} = \sum_j w_{ij}^{(l)} a_j^{(l-1)}$ is the weighted input to neuron $i$.

**Key insight**: The hard part ($\frac{\partial L}{\partial z_i^{(l)}}$) is computed recursively using chain rule from later layers. This is why deep learning works—we can train deep networks efficiently.

### 3. Feature Sensitivity Analysis

**Question**: How does model output change with input features?

**Answer**: Partial derivatives show sensitivity!

$$\text{Sensitivity to feature } x_j = \left|\frac{\partial y}{\partial x_j}\right|$$

- Large $|\frac{\partial y}{\partial x_j}|$ → Feature $x_j$ strongly influences prediction
- Small $|\frac{\partial y}{\partial x_j}|$ → Feature $x_j$ has little impact

**Example**: House price model $P(size, age, location)$

If $\frac{\partial P}{\partial size} = 200$ and $\frac{\partial P}{\partial age} = -5000$:
- Each additional sq ft adds $200 to price
- Each additional year subtracts $5000 from price
- Age is more important than size (in absolute terms)

### 4. Optimization in High Dimensions

**Reality**: ML models have thousands to billions of parameters
- GPT-3: 175 billion parameters
- ResNet-50: 25 million parameters
- Simple neural net: 10,000+ parameters

**Challenge**: Can't visualize 10,000-dimensional space!

**Solution**: Gradient-based optimization works in any dimension. The math is identical:

$$\mathbf{w}^{(new)} = \mathbf{w}^{(old)} - \alpha \nabla L(\mathbf{w}^{(old)})$$

Even though we can't visualize it, the gradient still points toward better solutions.

---

## Common Pitfalls and Misconceptions

### Pitfall 1: Confusing Partial and Total Derivatives

❌ **Wrong**: "Partial derivative is just like regular derivative"

✅ **Right**: Partial derivative holds other variables constant. Total derivative accounts for all dependencies.

**Example**: If $z = f(x, y)$ and $y = g(x)$:
- Partial: $\frac{\partial f}{\partial x}$ treats $y$ as constant
- Total: $\frac{df}{dx} = \frac{\partial f}{\partial x} + \frac{\partial f}{\partial y}\frac{dy}{dx}$

### Pitfall 2: Wrong Order in Mixed Partials

❌ **Wrong**: Computing $f_{xy}$ by differentiating w.r.t. $y$ first

✅ **Right**: For $f_{xy}$, differentiate w.r.t. $x$ first (get $f_x$), then w.r.t. $y$ (get $f_{xy}$)

**Good news**: For continuous functions, order doesn't matter (Clairaut's theorem), so if you get different answers, you made an error!

### Pitfall 3: Forgetting to Normalize Direction Vectors

❌ **Wrong**: $D_{\mathbf{v}}f = \nabla f \cdot \mathbf{v}$ for any vector $\mathbf{v}$

✅ **Right**: $D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u}$ where $\mathbf{u}$ is a **unit vector** ($|\mathbf{u}| = 1$)

**Fix**: Always normalize! $\mathbf{u} = \frac{\mathbf{v}}{|\mathbf{v}|}$

### Pitfall 4: Assuming Partials Exist → Differentiable

❌ **Wrong**: "Both $f_x$ and $f_y$ exist, so $f$ is differentiable"

✅ **Right**: Existence of partials does NOT guarantee differentiability (or even continuity!)

**Requirement**: Partials must exist AND be continuous near the point.

**Counterexample**:
$$f(x, y) = \begin{cases} \frac{xy^2}{x^2 + y^4} & (x,y) \neq (0,0) \\ 0 & (x,y) = (0,0) \end{cases}$$

Both $f_x(0, 0)$ and $f_y(0, 0)$ exist and equal 0, but $f$ is NOT continuous at origin!

### Pitfall 5: Sign Errors in Gradient Descent

❌ **Wrong**: $\mathbf{w}^{(new)} = \mathbf{w}^{(old)} + \alpha \nabla L$ (gradient **ascent**)

✅ **Right**: $\mathbf{w}^{(new)} = \mathbf{w}^{(old)} - \alpha \nabla L$ (gradient **descent**)

**Mnemonic**: To minimize, go DOWNhill. Gradient points UPhill. So subtract!

---

## Self-Assessment and Active Recall

### Concept Check Questions

1. **Explain in your own words**: What does $\frac{\partial f}{\partial x}$ mean? How is it different from $\frac{df}{dx}$?

2. **Geometric intuition**: If $\nabla f(a, b) = \langle 3, -4 \rangle$, which direction should you walk to increase $f$ fastest? To decrease fastest? To stay at the same value?

3. **True or False**: If $f_{xy} \neq f_{yx}$, you made a calculation error. Explain why.

4. **Practical scenario**: Your model has loss $L(w_1, w_2)$ and you compute $\nabla L = \langle 100, 0.01 \rangle$. Which parameter should you focus on tuning?

5. **Chain rule application**: If $z = x^2y$ where $x = \cos(t)$ and $y = e^t$, find $\frac{dz}{dt}$ without substituting first.

### Coding Challenges

**Easy**: Implement functions to numerically compute $f_x$, $f_y$, and $\nabla f$ for any function $f(x, y)$.

**Medium**: Implement gradient descent to fit $y = w_0 + w_1x + w_2x^2$ (polynomial regression).

**Hard**: Code backpropagation for a 2-layer neural network from scratch, computing all partial derivatives manually.

### Practice Problems

#### Basic Level

1. Find $f_x$ and $f_y$ for:
   - (a) $f(x, y) = x^3 + 2xy + y^3$
   - (b) $f(x, y) = e^{x^2 + y^2}$
   - (c) $f(x, y) = \ln(x^2 + y^2)$

2. Compute $\nabla f$ at $(1, 2)$ for $f(x, y) = x^2y + xy^2$

3. Find directional derivative of $f(x, y) = xy$ at $(1, 1)$ in direction $\mathbf{v} = \langle 1, 1 \rangle$

#### Intermediate Level

4. Find all second-order partial derivatives for $f(x, y) = x^3y^2 - x^2y^3$. Verify Clairaut's theorem.

5. Find equation of tangent plane to $z = x^2 + y^2$ at $(2, 1, 5)$.

6. Use linear approximation to estimate $f(1.98, 3.01)$ where $f(x, y) = \sqrt{x^2 + y^2}$.

7. For $z = \sin(xy)$ where $x = t^2$ and $y = e^t$, find $\frac{dz}{dt}$.

#### Advanced Level

8. Find maximum rate of increase of $f(x, y) = xe^{-y}$ at $(2, 0)$ and direction.

9. Given $w = xy + yz + zx$ where $x = \cos(t)$, $y = \sin(t)$, $z = t$, find $\frac{dw}{dt}$ at $t = 0$.

10. Prove that for $f(x, y) = x^2 + y^2$, the gradient $\nabla f$ is perpendicular to level curves.

11. For loss $L(w_1, w_2) = (w_1 - 3)^2 + (w_2 - 2)^2$, perform 3 iterations of gradient descent from $(0, 0)$ with $\alpha = 0.1$.

12. Show that
$$f(x, y) = \begin{cases} \frac{xy}{x^2 + y^2} & (x,y) \neq (0,0) \\ 0 & (x,y) = (0,0) \end{cases}$$
has partial derivatives at origin but is not continuous there.

---

## Quick Reference Summary

### Key Formulas

**Partial Derivatives**:
$$\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}$$

**Gradient**:
$$\nabla f = \left\langle \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right\rangle$$

**Directional Derivative**:
$$D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u} \quad (|\mathbf{u}| = 1)$$

**Chain Rule** (one parameter):
$$\frac{dz}{dt} = \frac{\partial f}{\partial x}\frac{dx}{dt} + \frac{\partial f}{\partial y}\frac{dy}{dt}$$

**Tangent Plane**:
$$z = f(x_0, y_0) + f_x(x_0, y_0)(x - x_0) + f_y(x_0, y_0)(y - y_0)$$

**Clairaut's Theorem**:
$$f_{xy} = f_{yx}$$ (if continuous)

### When to Use What

| Task | Tool |
|------|------|
| Rate of change in x-direction | $f_x$ |
| Rate of change in y-direction | $f_y$ |
| Direction of steepest increase | $\nabla f$ |
| Maximum rate of increase | $|\nabla f|$ |
| Rate in arbitrary direction | $D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u}$ |
| Optimize function | Gradient descent: $w \leftarrow w - \alpha\nabla f$ |
| Approximate function near point | Tangent plane / linearization |
| Derivatives with dependencies | Chain rule |

### Code Skeleton

```python
import numpy as np

def numerical_gradient(f, point, h=1e-5):
    """Compute gradient numerically."""
    x, y = point
    fx = (f(x+h, y) - f(x, y)) / h
    fy = (f(x, y+h) - f(x, y)) / h
    return np.array([fx, fy])

def directional_derivative(grad, direction):
    """Compute D_u f = grad · u."""
    u = direction / np.linalg.norm(direction)  # Normalize!
    return np.dot(grad, u)

def gradient_descent(f, grad_f, initial, alpha, iterations):
    """Basic gradient descent."""
    w = np.array(initial)
    for i in range(iterations):
        w = w - alpha * grad_f(w)
    return w
```

### Top 3 Things to Remember

1. **Partial derivatives** treat other variables as constants
2. **Gradient** $\nabla f$ points in direction of steepest ascent, magnitude is max rate
3. **Gradient descent** moves opposite to gradient to minimize: $w \leftarrow w - \alpha\nabla L$

---

## Further Resources

### Documentation
- NumPy: `np.gradient()` for numerical gradients
- SciPy: `scipy.optimize.minimize()` for optimization
- PyTorch/TensorFlow: Automatic differentiation (autograd)

### Papers & Books
- Stewart, "Calculus: Early Transcendentals" - Chapter 14
- Thomas' Calculus - Multivariable sections
- Goodfellow et al., "Deep Learning" - Chapter 4 (Numerical Computation)

### Practice Resources
- Khan Academy: Multivariable Calculus
- 3Blue1Brown: "Essence of Calculus" series
- Calculus.org: Interactive exercises

### Review Schedule
- **After 1 day**: Review concept check questions
- **After 3 days**: Redo basic practice problems from memory
- **After 1 week**: Implement gradient descent for new problem
- **After 2 weeks**: Teach these concepts to someone else

---

**Related Notes**:
- Previous: [week-07-single-variable-calculus.md](week-07-single-variable-calculus.md)
- Next: [week-09-optimization-basics.md](week-09-optimization-basics.md)
- Foundations: [week-01-vectors-matrices-intro.md](week-01-vectors-matrices-intro.md)

**Last Updated**: 2025-11-22
**Status**: Complete and comprehensive ✅
