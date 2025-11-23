# Week 9: Optimization - Finding Extrema in Multiple Dimensions

---
Date: 2025-11-22
Course: BSMA1003 - Mathematics for Data Science II
Level: Foundation
Week: 9 of 11
Source: IIT Madras Mathematics II Week 9
Topic Area: Optimization Theory
Tags: #BSMA1003 #Optimization #HessianMatrix #LagrangeMultipliers #ConstrainedOptimization #Week9 #Foundation
---

## Overview (BLUF)

**One-sentence definition**: Optimization finds maximum or minimum values of multivariable functions, which is the core problem in machine learning where we minimize loss functions to train models.

**Why it matters**: Every machine learning algorithm—linear regression, neural networks, support vector machines—solves an optimization problem. Understanding how to find extrema, classify critical points with the Hessian matrix, and handle constraints with Lagrange multipliers is essential for designing and debugging ML algorithms. Without optimization theory, modern AI wouldn't exist.

**When to use**: Finding optimal model parameters (training ML models), resource allocation problems (maximize profit subject to budget), engineering design (minimize cost while meeting specifications), portfolio optimization (maximize return for given risk), hyperparameter tuning (finding best learning rate, regularization strength).

**Prerequisites**: Partial derivatives and gradients ([week-08-multivariable-calculus-partial-derivatives.md](week-08-rank-nullity.md)), single-variable optimization from [week-07-single-variable-calculus.md](week-07-single-variable-calculus.md), matrix operations ([week-02-matrix-operations.md](week-02-matrix-operations.md)). Must understand what a critical point is (where derivative equals zero) and second derivative test for single-variable functions.

---

## Core Theory

### 1. Critical Points in Multiple Dimensions

**Intuitive explanation**: Imagine a hilly landscape. The **critical points** are places where the ground is perfectly level—the tops of hills (maxima), bottoms of valleys (minima), and saddle points (neither maximum nor minimum, like a mountain pass).

**Definition**: A point $(a, b)$ is a **critical point** of $f(x, y)$ if:
$$\nabla f(a, b) = \mathbf{0}$$

This means:
$$\frac{\partial f}{\partial x}(a, b) = 0 \quad \text{and} \quad \frac{\partial f}{\partial y}(a, b) = 0$$

**Why this makes sense**: At a maximum or minimum, you can't go uphill or downhill in ANY direction—the gradient must be zero.

**Three types of critical points**:

1. **Local Maximum**: Function value is highest nearby (hilltop)
   - $f(a, b) \geq f(x, y)$ for all $(x, y)$ near $(a, b)$

2. **Local Minimum**: Function value is lowest nearby (valley bottom)
   - $f(a, b) \leq f(x, y)$ for all $(x, y)$ near $(a, b)$

3. **Saddle Point**: Maximum in some directions, minimum in others (mountain pass)
   - Neither maximum nor minimum

**Everyday analogy**: You're at a critical point if:
- You're at the top of a hill (max)
- You're at the bottom of a valley (min)
- You're at a mountain pass—uphill both left/right, downhill forward/backward (saddle)

#### Example 1: Finding Critical Points

**Function**: $f(x, y) = x^2 + y^2 - 2x - 4y + 5$

**Step 1**: Compute partial derivatives
$$f_x = 2x - 2$$
$$f_y = 2y - 4$$

**Step 2**: Set both equal to zero
$$2x - 2 = 0 \Rightarrow x = 1$$
$$2y - 4 = 0 \Rightarrow y = 2$$

**Critical point**: $(1, 2)$

**Step 3**: What is the function value there?
$$f(1, 2) = 1 + 4 - 2 - 8 + 5 = 0$$

**Question**: Is this a max, min, or saddle? We need the Hessian matrix to classify it!

```python
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Define function
def f(x, y):
    return x**2 + y**2 - 2*x - 4*y + 5

# Create mesh
x = np.linspace(-2, 4, 100)
y = np.linspace(-2, 6, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# Plot surface
fig = plt.figure(figsize=(14, 6))

# 3D surface
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax1.scatter([1], [2], [0], color='red', s=100, label='Critical point (1,2)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.set_title('3D Surface Plot')

# Contour plot
ax2 = fig.add_subplot(122)
contours = ax2.contour(X, Y, Z, levels=20, cmap='viridis')
ax2.clabel(contours, inline=True, fontsize=8)
ax2.plot(1, 2, 'ro', markersize=10, label='Critical point (1,2)')
ax2.set_xlabel('x')
ax2.set_ylabel('y')
ax2.set_title('Contour Plot')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### 2. The Hessian Matrix - Second Derivative Test

**Single-variable review**: For $f(x)$, second derivative test:
- If $f'(a) = 0$ and $f''(a) > 0$ → local minimum (concave up)
- If $f'(a) = 0$ and $f''(a) < 0$ → local maximum (concave down)
- If $f'(a) = 0$ and $f''(a) = 0$ → test inconclusive

**Multivariable extension**: We need a **matrix** of second partial derivatives—the **Hessian matrix**.

**Definition**: The **Hessian matrix** $H$ of $f(x, y)$ at point $(a, b)$ is:

$$H(a, b) = \begin{bmatrix}
\frac{\partial^2 f}{\partial x^2} & \frac{\partial^2 f}{\partial x \partial y} \\[8pt]
\frac{\partial^2 f}{\partial y \partial x} & \frac{\partial^2 f}{\partial y^2}
\end{bmatrix}_{(a,b)} = \begin{bmatrix}
f_{xx}(a,b) & f_{xy}(a,b) \\
f_{yx}(a,b) & f_{yy}(a,b)
\end{bmatrix}$$

**For $n$ variables**: $H$ is an $n \times n$ symmetric matrix (by Clairaut's theorem: $f_{ij} = f_{ji}$)

**Second Derivative Test (2 variables)**:

At critical point $(a, b)$ where $\nabla f(a, b) = \mathbf{0}$:

1. Compute **determinant** of Hessian: $D = f_{xx}f_{yy} - (f_{xy})^2$
2. Classify:
   - **Local minimum** if $D > 0$ and $f_{xx} > 0$
   - **Local maximum** if $D > 0$ and $f_{xx} < 0$
   - **Saddle point** if $D < 0$
   - **Inconclusive** if $D = 0$

**Intuitive meaning**:
- $f_{xx}$: Concavity in x-direction
- $f_{yy}$: Concavity in y-direction
- $f_{xy}$: How x and y interact
- If both $f_{xx}$ and $f_{yy}$ are positive and large enough (relative to $f_{xy}^2$), we have a bowl shape → minimum

#### Example 2: Classifying Critical Points with Hessian

**Function**: $f(x, y) = x^2 + y^2 - 2x - 4y + 5$ (from Example 1)

**Critical point**: $(1, 2)$

**Step 1**: Compute second partial derivatives
$$f_{xx} = 2, \quad f_{yy} = 2, \quad f_{xy} = 0$$

**Step 2**: Form Hessian
$$H(1, 2) = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$$

**Step 3**: Compute determinant
$$D = f_{xx} \cdot f_{yy} - (f_{xy})^2 = 2 \cdot 2 - 0^2 = 4 > 0$$

**Step 4**: Check $f_{xx}$
$$f_{xx} = 2 > 0$$

**Conclusion**: Since $D > 0$ and $f_{xx} > 0$, point $(1, 2)$ is a **local minimum** ✓

**Function value at minimum**: $f(1, 2) = 0$

**Geometric interpretation**: The function is a paraboloid opening upward with vertex at $(1, 2, 0)$.

#### Example 3: Finding and Classifying Multiple Critical Points

**Function**: $f(x, y) = x^3 - 3x + y^2$

**Step 1**: Find critical points
$$f_x = 3x^2 - 3 = 0 \Rightarrow x^2 = 1 \Rightarrow x = \pm 1$$
$$f_y = 2y = 0 \Rightarrow y = 0$$

**Critical points**: $(1, 0)$ and $(-1, 0)$

**Step 2**: Compute second derivatives
$$f_{xx} = 6x, \quad f_{yy} = 2, \quad f_{xy} = 0$$

**Step 3**: Classify point $(1, 0)$
$$f_{xx}(1, 0) = 6, \quad f_{yy}(1, 0) = 2, \quad f_{xy}(1, 0) = 0$$
$$D = 6 \cdot 2 - 0^2 = 12 > 0$$
$$f_{xx} = 6 > 0$$

**Conclusion**: $(1, 0)$ is a **local minimum** with $f(1, 0) = 1 - 3 + 0 = -2$

**Step 4**: Classify point $(-1, 0)$
$$f_{xx}(-1, 0) = -6, \quad f_{yy}(-1, 0) = 2, \quad f_{xy}(-1, 0) = 0$$
$$D = (-6) \cdot 2 - 0^2 = -12 < 0$$

**Conclusion**: $(-1, 0)$ is a **saddle point** with $f(-1, 0) = -1 + 3 + 0 = 2$

```python
# Verify classification visually
def f(x, y):
    return x**3 - 3*x + y**2

x = np.linspace(-2, 2, 100)
y = np.linspace(-3, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

fig = plt.figure(figsize=(14, 6))

# 3D plot
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='coolwarm', alpha=0.7)
ax1.scatter([1], [0], [f(1, 0)], color='green', s=100, label='Min (1,0)')
ax1.scatter([-1], [0], [f(-1, 0)], color='red', s=100, label='Saddle (-1,0)')
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('f(x,y)')
ax1.legend()

# Contour plot
ax2 = fig.add_subplot(122)
contours = ax2.contour(X, Y, Z, levels=30, cmap='coolwarm')
ax2.clabel(contours, inline=True, fontsize=7)
ax2.plot(1, 0, 'go', markersize=10, label='Min (1,0)')
ax2.plot(-1, 0, 'ro', markersize=10, label='Saddle (-1,0)')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### Example 4: Saddle Point Identification

**Function**: $f(x, y) = x^2 - y^2$ (hyperbolic paraboloid)

**Step 1**: Find critical points
$$f_x = 2x = 0 \Rightarrow x = 0$$
$$f_y = -2y = 0 \Rightarrow y = 0$$

**Critical point**: $(0, 0)$

**Step 2**: Hessian
$$f_{xx} = 2, \quad f_{yy} = -2, \quad f_{xy} = 0$$
$$H(0, 0) = \begin{bmatrix} 2 & 0 \\ 0 & -2 \end{bmatrix}$$

**Step 3**: Determinant
$$D = 2 \cdot (-2) - 0^2 = -4 < 0$$

**Conclusion**: $(0, 0)$ is a **saddle point**

**Why it's a saddle**:
- Along x-axis ($y = 0$): $f(x, 0) = x^2$ → minimum at origin
- Along y-axis ($x = 0$): $f(0, y) = -y^2$ → maximum at origin

This is the classic "potato chip" or "Pringles" shape—a saddle!

---

### 3. Global vs Local Extrema

**Local extremum**: Best value in a small neighborhood
**Global extremum**: Best value in entire domain

**Key insight**: A function can have multiple local minima but only one global minimum.

**Finding global extrema**:
1. Find all critical points in interior of domain
2. Check boundary of domain separately
3. Compare all candidates to find absolute max/min

**Closed Bounded Domain Theorem** (Extreme Value Theorem):
If $f$ is continuous on a closed and bounded domain $D$, then $f$ has both an absolute maximum and absolute minimum in $D$.

**Strategy for closed bounded domains**:
1. Find critical points inside $D$
2. Check function values on boundary of $D$
3. Largest value = global max, smallest = global min

#### Example 5: Finding Global Extrema

**Function**: $f(x, y) = x^2 + y^2 - 2x$ on the disk $x^2 + y^2 \leq 4$

**Step 1**: Find critical points inside disk
$$f_x = 2x - 2 = 0 \Rightarrow x = 1$$
$$f_y = 2y = 0 \Rightarrow y = 0$$

Critical point: $(1, 0)$. Check if inside disk: $1^2 + 0^2 = 1 < 4$ ✓

Value: $f(1, 0) = 1 + 0 - 2 = -1$

**Step 2**: Check boundary (circle $x^2 + y^2 = 4$)

Parameterize: $x = 2\cos(t)$, $y = 2\sin(t)$

Substitute:
$$g(t) = (2\cos t)^2 + (2\sin t)^2 - 2(2\cos t)$$
$$= 4(\cos^2 t + \sin^2 t) - 4\cos t$$
$$= 4 - 4\cos t$$

Find extrema: $g'(t) = 4\sin t = 0 \Rightarrow t = 0, \pi$

- At $t = 0$: $(x, y) = (2, 0)$, $f(2, 0) = 4 + 0 - 4 = 0$
- At $t = \pi$: $(x, y) = (-2, 0)$, $f(-2, 0) = 4 + 0 + 4 = 8$

**Step 3**: Compare all candidates
- Interior critical point: $f(1, 0) = -1$
- Boundary: $f(2, 0) = 0$ and $f(-2, 0) = 8$

**Conclusion**:
- **Global minimum**: $f(1, 0) = -1$ at $(1, 0)$
- **Global maximum**: $f(-2, 0) = 8$ at $(-2, 0)$

---

### 4. Constrained Optimization - Lagrange Multipliers

**The problem**: Often we want to optimize $f(x, y)$ but subject to a constraint $g(x, y) = 0$.

**Example**: Maximize profit $P(x, y)$ subject to budget constraint $x + y = 100$.

**Geometric intuition**: Imagine hiking on a hillside (the function $f$) but constrained to stay on a specific trail (the constraint $g = 0$). The highest point you can reach on that trail might not be the highest point on the whole mountain!

**Key observation**: At the constrained optimum, the gradient of $f$ must be parallel to the gradient of $g$:
$$\nabla f = \lambda \nabla g$$

for some scalar $\lambda$ (the **Lagrange multiplier**).

**Why this works**: If $\nabla f$ weren't parallel to $\nabla g$, we could move along the constraint curve in a direction that increases $f$, meaning we haven't reached an optimum yet.

**Method of Lagrange Multipliers**:

To optimize $f(x, y)$ subject to $g(x, y) = 0$:

1. Set up system:
   $$\nabla f = \lambda \nabla g$$
   $$g(x, y) = 0$$

2. Solve for $x$, $y$, and $\lambda$

3. Evaluate $f$ at each solution to find max/min

#### Example 6: Lagrange Multipliers - Basic Application

**Problem**: Find maximum and minimum of $f(x, y) = xy$ subject to $x^2 + y^2 = 1$ (unit circle).

**Step 1**: Set up Lagrange equations

Gradients:
$$\nabla f = \langle y, x \rangle$$
$$\nabla g = \langle 2x, 2y \rangle$$

Condition $\nabla f = \lambda \nabla g$:
$$y = 2\lambda x \quad \text{...(1)}$$
$$x = 2\lambda y \quad \text{...(2)}$$

Constraint:
$$x^2 + y^2 = 1 \quad \text{...(3)}$$

**Step 2**: Solve system

From (1): $y = 2\lambda x$

Substitute into (2): $x = 2\lambda(2\lambda x) = 4\lambda^2 x$

If $x \neq 0$: $1 = 4\lambda^2 \Rightarrow \lambda = \pm \frac{1}{2}$

**Case 1**: $\lambda = \frac{1}{2}$
- From (1): $y = x$
- From (3): $x^2 + x^2 = 1 \Rightarrow x = \pm \frac{1}{\sqrt{2}}$
- Solutions: $\left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)$ and $\left(-\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}}\right)$

**Case 2**: $\lambda = -\frac{1}{2}$
- From (1): $y = -x$
- From (3): $x^2 + x^2 = 1 \Rightarrow x = \pm \frac{1}{\sqrt{2}}$
- Solutions: $\left(\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}}\right)$ and $\left(-\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)$

**Step 3**: Evaluate $f$ at all solutions

$$f\left(\pm\frac{1}{\sqrt{2}}, \pm\frac{1}{\sqrt{2}}\right) = \frac{1}{2}$$ (same sign)

$$f\left(\pm\frac{1}{\sqrt{2}}, \mp\frac{1}{\sqrt{2}}\right) = -\frac{1}{2}$$ (opposite signs)

**Conclusion**:
- **Maximum**: $f = \frac{1}{2}$ at $\left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)$ and $\left(-\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}}\right)$
- **Minimum**: $f = -\frac{1}{2}$ at $\left(\frac{1}{\sqrt{2}}, -\frac{1}{\sqrt{2}}\right)$ and $\left(-\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right)$

```python
# Visualize constrained optimization
theta = np.linspace(0, 2*np.pi, 100)
x_circle = np.cos(theta)
y_circle = np.sin(theta)
f_values = x_circle * y_circle

plt.figure(figsize=(12, 5))

# Contour plot with constraint
plt.subplot(121)
x = np.linspace(-1.5, 1.5, 200)
y = np.linspace(-1.5, 1.5, 200)
X, Y = np.meshgrid(x, y)
Z = X * Y
contours = plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.clabel(contours, inline=True, fontsize=8)
plt.plot(x_circle, y_circle, 'r-', linewidth=3, label='Constraint: $x^2+y^2=1$')
plt.plot(1/np.sqrt(2), 1/np.sqrt(2), 'go', markersize=12, label='Max')
plt.plot(-1/np.sqrt(2), -1/np.sqrt(2), 'go', markersize=12)
plt.plot(1/np.sqrt(2), -1/np.sqrt(2), 'ro', markersize=12, label='Min')
plt.plot(-1/np.sqrt(2), 1/np.sqrt(2), 'ro', markersize=12)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axis('equal')
plt.grid(True, alpha=0.3)

# Function values on constraint
plt.subplot(122)
plt.plot(np.degrees(theta), f_values, linewidth=2)
plt.axhline(y=0.5, color='g', linestyle='--', label='Max = 0.5')
plt.axhline(y=-0.5, color='r', linestyle='--', label='Min = -0.5')
plt.xlabel('Angle (degrees)')
plt.ylabel('f(x,y) = xy')
plt.title('Function Values on Constraint')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

#### Example 7: Lagrange Multipliers - Multiple Constraints

**Problem**: Optimize $f(x, y, z) = xyz$ subject to $x + y + z = 6$ and $xy = 4$.

**Note**: With two constraints $g_1 = 0$ and $g_2 = 0$, use two multipliers:
$$\nabla f = \lambda_1 \nabla g_1 + \lambda_2 \nabla g_2$$

This creates a system that can be solved, though it's more complex. In practice, substitution or elimination is often easier.

**Alternative approach using substitution**:

From $xy = 4$: $y = \frac{4}{x}$

Substitute into $x + y + z = 6$:
$$x + \frac{4}{x} + z = 6 \Rightarrow z = 6 - x - \frac{4}{x}$$

Now optimize single-constraint problem:
$$f(x) = x \cdot \frac{4}{x} \cdot \left(6 - x - \frac{4}{x}\right) = 4\left(6 - x - \frac{4}{x}\right)$$

This reduces to single-variable optimization (take derivative, set to zero).

---

### 5. Convex Functions and Global Optimality

**Definition**: Function $f$ is **convex** if for any two points and any $t \in [0, 1]$:
$$f(tx_1 + (1-t)x_2) \leq tf(x_1) + (1-t)f(x_2)$$

**Geometric meaning**: Line segment connecting any two points on graph lies above the graph (bowl shape).

**Critical property**: For convex functions, **every local minimum is a global minimum**!

**Why this matters in ML**: Many loss functions are convex (or nearly convex):
- Linear regression loss (MSE): Convex
- Logistic regression loss: Convex
- SVM loss: Convex

When your loss is convex, gradient descent is guaranteed to find the global minimum (if it converges).

**Test for convexity**: Function $f$ is convex if Hessian matrix $H$ is **positive semidefinite** everywhere.

For 2D: $f_{xx} \geq 0$, $f_{yy} \geq 0$, and $f_{xx}f_{yy} \geq (f_{xy})^2$

#### Example 8: Convex Function

**Function**: $f(x, y) = x^2 + y^2$ (paraboloid)

**Hessian**:
$$H = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$$

This is positive definite (eigenvalues are both 2 > 0), so $f$ is strictly convex.

**Implication**: The critical point $(0, 0)$ with $f(0, 0) = 0$ is the **global minimum**.

---

## Data Science Applications

### 1. Training Machine Learning Models

**Every ML model training = optimization problem**:

**Linear Regression**: Minimize MSE
$$L(\mathbf{w}) = \frac{1}{m}\sum_{i=1}^{m}(y_i - \mathbf{w}^T\mathbf{x}_i)^2$$

**Logistic Regression**: Minimize cross-entropy
$$L(\mathbf{w}) = -\frac{1}{m}\sum_{i=1}^{m}[y_i\log(\sigma(\mathbf{w}^T\mathbf{x}_i)) + (1-y_i)\log(1-\sigma(\mathbf{w}^T\mathbf{x}_i))]$$

**Neural Networks**: Minimize loss via backpropagation + gradient descent
$$\mathbf{w} \leftarrow \mathbf{w} - \alpha \nabla L(\mathbf{w})$$

### 2. Hyperparameter Tuning as Optimization

**Problem**: Find best learning rate $\alpha$ and regularization $\lambda$ that minimize validation error.

This is a constrained optimization problem:
- Objective: Minimize validation loss
- Constraints: $\alpha > 0$, $\lambda \geq 0$, computational budget

### 3. Support Vector Machines (SVM)

**Primal problem**: Minimize $\frac{1}{2}\|\mathbf{w}\|^2$ subject to $y_i(\mathbf{w}^T\mathbf{x}_i + b) \geq 1$ for all $i$.

This is exactly a constrained optimization problem solved with Lagrange multipliers, leading to the dual formulation.

### 4. Principal Component Analysis (PCA)

**Problem**: Find directions of maximum variance in data.

**Formulation**: Maximize $\mathbf{v}^T\Sigma\mathbf{v}$ subject to $\|\mathbf{v}\| = 1$ where $\Sigma$ is covariance matrix.

Solution uses Lagrange multipliers, leading to eigenvalue problem.

---

## Common Pitfalls and Misconceptions

### Pitfall 1: Critical Point ≠ Extremum

❌ **Wrong**: "I found a critical point, so it's a minimum."

✅ **Right**: Critical points can be maxima, minima, OR saddle points. Always use Hessian to classify!

### Pitfall 2: Local ≠ Global

❌ **Wrong**: "I found the minimum by gradient descent."

✅ **Right**: You found **a** local minimum. Without convexity, no guarantee it's global.

**Example**: $f(x) = x^3 - 3x$ has local min at $x=1$ and local max at $x=-1$, but no global extrema.

### Pitfall 3: Forgetting Boundary

❌ **Wrong**: "I checked all critical points and found the max."

✅ **Right**: On a bounded domain, must check boundary separately! Global extrema often occur on boundaries.

### Pitfall 4: Lagrange ≠ All Solutions

❌ **Wrong**: "I solved Lagrange equations, so I have the answer."

✅ **Right**: Lagrange finds **candidates**. Must evaluate $f$ at each to determine which is max/min.

### Pitfall 5: Hessian Test Inconclusiv when $D = 0$

❌ **Wrong**: "Determinant is zero, so it's a saddle point."

✅ **Right**: $D = 0$ means test is **inconclusive**. Need higher-order derivatives or other methods.

---

## Self-Assessment and Active Recall

### Concept Check Questions

1. **Explain**: Why must $\nabla f = \mathbf{0}$ at a local extremum (assuming it exists)?

2. **Classify**: For critical point with Hessian $H = \begin{bmatrix} -3 & 1 \\ 1 & -2 \end{bmatrix}$, is it max, min, or saddle?

3. **Intuition**: Why does $\nabla f = \lambda \nabla g$ at constrained optimum?

4. **True/False**: A function can have multiple local minima but only one global minimum. Explain.

5. **Application**: In ML, why do we care if the loss function is convex?

### Practice Problems

#### Basic Level

1. Find and classify all critical points:
   - (a) $f(x, y) = x^2 + 2y^2 - 4x + 8y$
   - (b) $f(x, y) = x^2 - xy + y^2$

2. Use Hessian to classify $(0, 0)$ for $f(x, y) = 3x^2 + 2xy + y^2$

3. Find extrema of $f(x, y) = x + y$ on $x^2 + y^2 = 1$ using Lagrange multipliers.

#### Intermediate Level

4. Find global max/min of $f(x, y) = x^2 + y^2 - x - y$ on the square $0 \leq x, y \leq 2$.

5. Minimize $f(x, y, z) = x^2 + y^2 + z^2$ subject to $x + y + z = 6$.

6. Classify all critical points of $f(x, y) = x^4 + y^4 - 4xy + 1$.

#### Advanced Level

7. Find dimensions of a rectangular box with maximum volume if surface area is fixed at $S = 96$.

8. Use Lagrange multipliers to find closest point on $x^2 + y^2 = 1$ to point $(3, 1)$.

9. Show that for positive $x, y, z$ with $xyz = 1$, the minimum of $x + y + z$ is 3.

10. Implement Newton's method (using Hessian) to minimize $f(x, y) = (x-3)^2 + (y+1)^2$ starting from $(0, 0)$.

---

## Quick Reference Summary

### Key Formulas

**Critical Point**: $\nabla f = \mathbf{0}$

**Hessian Matrix** (2D):
$$H = \begin{bmatrix} f_{xx} & f_{xy} \\ f_{yx} & f_{yy} \end{bmatrix}$$

**Second Derivative Test**:
- Min if $D > 0$ and $f_{xx} > 0$
- Max if $D > 0$ and $f_{xx} < 0$
- Saddle if $D < 0$
- Inconclusive if $D = 0$

where $D = f_{xx}f_{yy} - (f_{xy})^2$

**Lagrange Multipliers**:
$$\nabla f = \lambda \nabla g$$

### Decision Tree: Classifying Critical Points

```
Is ∇f = 0? ──No──> Not a critical point
    │
   Yes
    │
Compute D = f_xx·f_yy - (f_xy)²
    │
    ├──D > 0 ──> Check f_xx
    │             ├── f_xx > 0 ──> Local MIN
    │             └── f_xx < 0 ──> Local MAX
    │
    ├──D < 0 ──> SADDLE POINT
    │
    └──D = 0 ──> Test INCONCLUSIVE
                  (need higher-order derivatives)
```

### Code Skeleton

```python
import numpy as np
from scipy.optimize import minimize

def find_critical_points(f, f_grad, initial_guesses):
    """Find critical points by solving ∇f = 0."""
    critical_points = []
    for guess in initial_guesses:
        result = minimize(lambda x: np.linalg.norm(f_grad(x)),
                         x0=guess, method='BFGS')
        if result.success:
            critical_points.append(result.x)
    return critical_points

def hessian_test(H):
    """Classify critical point using Hessian."""
    D = H[0,0]*H[1,1] - H[0,1]**2
    if D > 0:
        return "Minimum" if H[0,0] > 0 else "Maximum"
    elif D < 0:
        return "Saddle point"
    else:
        return "Inconclusive"

def lagrange_optimization(f, g, initial):
    """Constrained optimization using Lagrange multipliers."""
    constraint = {'type': 'eq', 'fun': g}
    result = minimize(f, x0=initial, constraints=[constraint])
    return result.x, result.fun
```

### Top 3 Things to Remember

1. **Hessian classifies**: Compute $D = f_{xx}f_{yy} - (f_{xy})^2$ at critical points
2. **Lagrange for constraints**: At optimum, $\nabla f = \lambda \nabla g$
3. **Convex = global**: If loss function is convex, local minimum is global minimum

---

## Further Resources

### Documentation
- SciPy: `scipy.optimize.minimize()` for optimization
- NumPy: Matrix operations for Hessian
- SymPy: Symbolic differentiation for exact derivatives

### Papers & Books
- Boyd & Vandenberghe, "Convex Optimization"
- Nocedal & Wright, "Numerical Optimization"
- Goodfellow et al., "Deep Learning" - Chapter 4

### Practice
- Convex Optimization course (Stanford CS364)
- Optimization problems on Leetcode/Codeforces
- Kaggle competitions (hyperparameter optimization)

### Review Schedule
- **After 1 day**: Redo Hessian classification examples
- **After 3 days**: Solve constrained optimization problems
- **After 1 week**: Implement gradient descent with Hessian (Newton's method)
- **After 2 weeks**: Apply to real ML problem (tune hyperparameters)

---

**Related Notes**:
- Previous: [week-08-multivariable-calculus-partial-derivatives.md](week-08-rank-nullity.md)
- Next: [week-10-ml-applications.md](week-10-ml-applications.md)
- Applications: Machine learning model training, resource allocation

**Last Updated**: 2025-11-22
**Status**: Complete and comprehensive ✅
