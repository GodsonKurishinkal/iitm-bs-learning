# Week 11: Hessian and Optimization

**Course**: BSMA1003 - Mathematics II
**Level**: Foundation

---

## Learning Objectives
- Understand higher-order partial derivatives
- Master the Hessian matrix and second derivative test
- Learn constrained optimization with Lagrange multipliers

---

## 1. Higher-Order Partial Derivatives

### 1.1 Theory

**Second-order derivatives** describe the curvature of a function. Mixed partials are equal for smooth functions (Clairaut's theorem).

### 1.2 Mathematical Definition

$$f_{xx} = \frac{\partial^2 f}{\partial x^2}, \quad f_{yy} = \frac{\partial^2 f}{\partial y^2}$$

$$f_{xy} = \frac{\partial^2 f}{\partial y \partial x}, \quad f_{yx} = \frac{\partial^2 f}{\partial x \partial y}$$

### 1.3 Clairaut's Theorem

For functions with continuous second partial derivatives:

$$f_{xy} = f_{yx}$$

### 1.4 Supply Chain Application

**Retail Context**: Second derivatives indicate **convexity** - whether a cost function curves upward (minimum exists) or downward. Essential for ensuring optimal solutions are truly optimal.

---

## 2. The Hessian Matrix

### 2.1 Theory

The **Hessian** collects all second-order partial derivatives into a matrix. Its definiteness determines whether critical points are maxima, minima, or saddle points.

### 2.2 Mathematical Definition

For $f(x, y)$:

$$H = \begin{pmatrix} f_{xx} & f_{xy} \\ f_{yx} & f_{yy} \end{pmatrix}$$

For $f(x_1, ..., x_n)$:

$$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$$

### 2.3 The Second Derivative Test (2D)

Let $D = \det(H) = f_{xx}f_{yy} - f_{xy}^2$

| Condition | Classification |
|-----------|---------------|
| $D > 0$ and $f_{xx} > 0$ | **Local Minimum** |
| $D > 0$ and $f_{xx} < 0$ | **Local Maximum** |
| $D < 0$ | **Saddle Point** |
| $D = 0$ | **Inconclusive** |

### 2.4 General Test via Eigenvalues

| Eigenvalue Condition | Hessian Type | Critical Point |
|---------------------|--------------|----------------|
| All $\lambda_i > 0$ | Positive definite | Local minimum |
| All $\lambda_i < 0$ | Negative definite | Local maximum |
| Mixed signs | Indefinite | Saddle point |
| Some $\lambda_i = 0$ | Semi-definite | Inconclusive |

### 2.5 Supply Chain Application

**Retail Context**:
- The Hessian confirms whether the **EOQ formula** gives a true minimum
- In **portfolio optimization**, the Hessian of variance determines allocation efficiency
- Verifying cost functions are convex ensures global optimum

---

## 3. Optimization with Constraints (Lagrange Multipliers)

### 3.1 Theory

Many real problems have **constraints**. Lagrange multipliers extend optimization to constrained settings.

### 3.2 Problem Formulation

**Maximize/Minimize** $f(x, y)$ **subject to** $g(x, y) = c$

### 3.3 The Lagrangian

$$\mathcal{L}(x, y, \lambda) = f(x, y) - \lambda(g(x, y) - c)$$

### 3.4 Solution Method

Solve the system:
$$\nabla \mathcal{L} = 0$$

This gives:
1. $\frac{\partial f}{\partial x} = \lambda \frac{\partial g}{\partial x}$
2. $\frac{\partial f}{\partial y} = \lambda \frac{\partial g}{\partial y}$
3. $g(x, y) = c$

### 3.5 Interpretation of $\lambda$

The **Lagrange multiplier** $\lambda$ represents the **rate of change** of the optimal value with respect to the constraint:

$$\lambda = \frac{d f^*}{d c}$$

### 3.6 Supply Chain Application

**Retail Context**:
- **Minimize cost** subject to service level constraints
- **Optimize production** subject to capacity limits
- **Maximize profit** subject to budget constraints
- $\lambda$ indicates the value of relaxing the constraint by one unit

---

## Summary Table

| Concept | Definition | Key Property | Supply Chain Application |
|---------|------------|--------------|--------------------------|
| **Second Partials** | $f_{xx}, f_{xy}, f_{yy}$ | Describe curvature | Convexity verification |
| **Hessian** | Matrix of second partials | Symmetric | Optimality classification |
| **Determinant Test** | $D = f_{xx}f_{yy} - f_{xy}^2$ | Classification rule | Min/max confirmation |
| **Lagrange Multipliers** | $\mathcal{L} = f - \lambda(g-c)$ | Handle constraints | Constrained optimization |

---

## Key Takeaways

1. **Second-order derivatives** describe curvature - essential for classifying critical points
2. **Hessian matrix** collects all second partials - its eigenvalues determine point type
3. **Positive definite Hessian** → local minimum; **negative definite** → local maximum
4. **Lagrange multipliers** extend optimization to constraints - $\lambda$ is the shadow price

---

## Course Conclusion

This completes **Mathematics II**! You now have the linear algebra and multivariable calculus foundation for:
- **Machine learning optimization** (gradient descent, convexity)
- **Regression and prediction models** (least squares, projections)
- **Supply chain optimization** (EOQ, constrained optimization)

---

*Congratulations on completing Mathematics II!*
*IIT Madras BS Degree in Data Science*
