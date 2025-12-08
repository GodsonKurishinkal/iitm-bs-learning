# Week 08: Orthogonality and Gram-Schmidt

**Course**: BSMA1003 - Mathematics II
**Level**: Foundation

---

## Learning Objectives
- Understand orthogonality and orthonormality
- Master the Gram-Schmidt orthogonalization process
- Learn orthogonal projections and their connection to least squares

---

## 1. Orthogonality

### 1.1 Theory

**Orthogonal vectors** are perpendicular - their inner product is zero. **Orthonormal vectors** are orthogonal with unit length.

### 1.2 Mathematical Definitions

| Concept | Definition |
|---------|------------|
| **Orthogonal** | $\langle \mathbf{u}, \mathbf{v} \rangle = 0$ |
| **Orthonormal** | Orthogonal and $\|\mathbf{u}\| = \|\mathbf{v}\| = 1$ |
| **Orthogonal Set** | All pairs are orthogonal |
| **Orthonormal Basis** | Orthogonal set of unit vectors spanning the space |

### 1.3 Orthogonal Matrices

An **orthogonal matrix** $Q$ satisfies:
$$Q^TQ = QQ^T = I$$

**Properties**:
| Property | Description |
|----------|-------------|
| **Inverse** | $Q^{-1} = Q^T$ |
| **Determinant** | $\det(Q) = \pm 1$ |
| **Preserves lengths** | $\|Q\mathbf{x}\| = \|\mathbf{x}\|$ |
| **Preserves angles** | $\langle Q\mathbf{u}, Q\mathbf{v} \rangle = \langle \mathbf{u}, \mathbf{v} \rangle$ |

### 1.4 Supply Chain Application

**Retail Context**:
- **Orthogonal features** are uncorrelated and provide independent information
- **PCA** produces orthogonal principal components for maximum variance capture
- Orthogonality helps avoid multicollinearity in regression models

---

## 2. Gram-Schmidt Process

### 2.1 Theory

**Gram-Schmidt** transforms any basis into an orthonormal basis. This is essential for QR decomposition and stable numerical computations.

### 2.2 Algorithm

Given linearly independent vectors $\{\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_n\}$, produce orthonormal $\{\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_n\}$:

**Step 1**: Orthogonalize
$$\mathbf{u}_1 = \mathbf{v}_1$$
$$\mathbf{u}_k = \mathbf{v}_k - \sum_{j=1}^{k-1} \text{proj}_{\mathbf{u}_j}(\mathbf{v}_k)$$

where $\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\langle \mathbf{v}, \mathbf{u} \rangle}{\langle \mathbf{u}, \mathbf{u} \rangle} \mathbf{u}$

**Step 2**: Normalize
$$\mathbf{e}_k = \frac{\mathbf{u}_k}{\|\mathbf{u}_k\|}$$

### 2.3 QR Decomposition

Gram-Schmidt produces the **QR decomposition**:
$$A = QR$$

where:
- $Q$ is orthogonal (columns are orthonormal)
- $R$ is upper triangular

### 2.4 Supply Chain Application

**Retail Context**: Gram-Schmidt underlies QR decomposition used in regression. It creates stable numerical solutions for demand forecasting models.

---

## 3. Orthogonal Projections

### 3.1 Theory

**Projecting** onto a subspace finds the closest point in that subspace. This is the geometric foundation of least squares.

### 3.2 Projection onto a Vector

$$\text{proj}_{\mathbf{u}}(\mathbf{v}) = \frac{\langle \mathbf{v}, \mathbf{u} \rangle}{\langle \mathbf{u}, \mathbf{u} \rangle} \mathbf{u} = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|^2} \mathbf{u}$$

### 3.3 Projection onto a Subspace

Projection of $\mathbf{b}$ onto column space of $A$:

$$\text{proj}_{\text{col}(A)}(\mathbf{b}) = A(A^TA)^{-1}A^T\mathbf{b}$$

The matrix $P = A(A^TA)^{-1}A^T$ is the **projection matrix**.

### 3.4 Properties of Projection Matrices

| Property | Description |
|----------|-------------|
| **Symmetric** | $P^T = P$ |
| **Idempotent** | $P^2 = P$ |
| **Residual** | $\mathbf{b} - P\mathbf{b}$ is orthogonal to col($A$) |

---

## 4. Least Squares

### 4.1 Connection to Projections

When $A\mathbf{x} = \mathbf{b}$ has no solution, find $\hat{\mathbf{x}}$ minimizing $\|A\mathbf{x} - \mathbf{b}\|$.

### 4.2 Normal Equations

$$\hat{\mathbf{x}} = (A^TA)^{-1}A^T\mathbf{b}$$

The fitted values: $\hat{\mathbf{b}} = A\hat{\mathbf{x}} = \text{proj}_{\text{col}(A)}(\mathbf{b})$

### 4.3 Supply Chain Application

**Retail Context**: Least squares regression for demand forecasting is an orthogonal projection. The fitted values are projections of observed demand onto the model space.

---

## Summary Table

| Concept | Definition | Key Property | Supply Chain Application |
|---------|------------|--------------|--------------------------|
| **Orthogonal** | $\langle \mathbf{u}, \mathbf{v} \rangle = 0$ | Perpendicular | Uncorrelated features |
| **Gram-Schmidt** | Orthogonalize then normalize | Produces QR | Stable regression |
| **Projection** | Closest point in subspace | $P^2 = P$ | Fitted values |
| **Least Squares** | $(A^TA)^{-1}A^T\mathbf{b}$ | Minimizes error | Demand forecasting |

---

## Key Takeaways

1. **Orthogonal vectors** provide independent information - no redundancy
2. **Gram-Schmidt** creates orthonormal bases from any basis - enables stable computation
3. **Orthogonal projection** finds the closest point - geometric core of regression
4. **Least squares** is projection onto model space - the foundation of linear regression

---

## Next Week Preview

Week 9 covers **Multivariable Functions and Partial Derivatives**.

---

*IIT Madras BS Degree in Data Science*
