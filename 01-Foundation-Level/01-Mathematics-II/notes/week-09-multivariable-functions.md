# Week 09: Multivariable Functions

**Course**: BSMA1003 - Mathematics II
**Level**: Foundation

---

## Learning Objectives
- Understand functions of multiple variables
- Master partial derivatives and their computation
- Learn directional derivatives and their applications

---

## 1. Multivariable Functions

### 1.1 Theory

**Functions of multiple variables** map $\mathbb{R}^n$ to $\mathbb{R}$. They can be visualized as surfaces or through level curves (contours).

### 1.2 Mathematical Definition

$$f: \mathbb{R}^n \rightarrow \mathbb{R}$$

**Example**: $f(x, y) = x^2 + y^2$

### 1.3 Visualization Methods

| Method | Description |
|--------|-------------|
| **3D Surface** | Plot $z = f(x, y)$ as a surface in 3D |
| **Level Curves (Contours)** | Sets where $f(x, y) = c$ for constant $c$ |
| **Heat Maps** | Color-coded values of $f$ on the $(x, y)$ plane |

### 1.4 Level Curves

**Level curves**: Sets where $f(x, y) = c$

- Closely spaced curves = steep change
- Widely spaced curves = gradual change
- Circles for $f(x,y) = x^2 + y^2$ (paraboloid)

### 1.5 Supply Chain Application

**Retail Context**:
- **Cost functions** depending on production quantity and pricing
- **Demand** as function of price and advertising
- **Inventory costs** depending on order quantity and reorder point

---

## 2. Partial Derivatives

### 2.1 Theory

**Partial derivatives** measure the rate of change with respect to one variable while holding others constant.

### 2.2 Mathematical Definition

$$\frac{\partial f}{\partial x} = \lim_{h \to 0} \frac{f(x+h, y) - f(x, y)}{h}$$

$$\frac{\partial f}{\partial y} = \lim_{h \to 0} \frac{f(x, y+h) - f(x, y)}{h}$$

### 2.3 Notation

| Notation | Meaning |
|----------|---------|
| $f_x$ | Partial derivative with respect to $x$ |
| $\frac{\partial f}{\partial x}$ | Leibniz notation |
| $\partial_x f$ | Operator notation |
| $D_1 f$ | Derivative with respect to first variable |

### 2.4 Higher-Order Partial Derivatives

| Notation | Description |
|----------|-------------|
| $f_{xx}$ | Second partial with respect to $x$ twice |
| $f_{xy}$ | Mixed partial: first $x$, then $y$ |
| $f_{yx}$ | Mixed partial: first $y$, then $x$ |

**Clairaut's Theorem**: If $f_{xy}$ and $f_{yx}$ are continuous, then $f_{xy} = f_{yx}$

### 2.5 Supply Chain Application

**Retail Context**:
- **Marginal analysis** - how does profit change with one more unit produced?
- **Sensitivity analysis** - how does total cost respond to individual parameter changes?

---

## 3. Directional Derivatives

### 3.1 Theory

**Directional derivatives** measure the rate of change in any direction, not just along coordinate axes.

### 3.2 Mathematical Definition

$$D_{\mathbf{u}}f = \nabla f \cdot \mathbf{u} = \frac{\partial f}{\partial x}u_1 + \frac{\partial f}{\partial y}u_2$$

where $\mathbf{u} = (u_1, u_2)$ is a **unit vector** in the direction of interest.

### 3.3 Key Properties

| Property | Description |
|----------|-------------|
| **Maximum rate** | In direction of gradient $\nabla f$ |
| **Minimum rate** | In direction of $-\nabla f$ |
| **Zero rate** | Perpendicular to gradient (along level curves) |
| **Value** | $D_{\mathbf{u}}f = \|\nabla f\| \cos\theta$ where $\theta$ = angle between $\nabla f$ and $\mathbf{u}$ |

### 3.4 Special Cases

| Direction | Directional Derivative |
|-----------|----------------------|
| Along $x$-axis: $\mathbf{u} = (1, 0)$ | $f_x$ |
| Along $y$-axis: $\mathbf{u} = (0, 1)$ | $f_y$ |
| Along gradient | $\|\nabla f\|$ (maximum) |

### 3.5 Supply Chain Application

**Retail Context**: When simultaneously changing price and advertising budget, the directional derivative gives the **combined effect** on revenue.

---

## Summary Table

| Concept | Definition | Key Property | Supply Chain Application |
|---------|------------|--------------|--------------------------|
| **Multivariable Function** | $f: \mathbb{R}^n \to \mathbb{R}$ | Visualized as surfaces | Cost/demand models |
| **Level Curves** | $f(x,y) = c$ | Show constant values | Iso-cost, iso-profit curves |
| **Partial Derivative** | Rate of change in one variable | Hold others constant | Marginal analysis |
| **Directional Derivative** | $\nabla f \cdot \mathbf{u}$ | Rate in any direction | Combined strategy effects |

---

## Key Takeaways

1. **Multivariable functions** model real-world scenarios with multiple inputs
2. **Partial derivatives** give marginal effects - essential for sensitivity analysis
3. **Directional derivatives** measure change in any direction - maximum along gradient
4. These tools are **foundational for optimization** in supply chain management

---

## Next Week Preview

Week 10 covers **Gradient and Critical Points** - finding optima of multivariable functions.

---

*IIT Madras BS Degree in Data Science*
