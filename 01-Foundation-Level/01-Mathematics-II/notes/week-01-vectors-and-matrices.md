# Week 01: Vectors and Matrices

**Course**: BSMA1003 - Mathematics II
**Level**: Foundation

---

## Learning Objectives
- Understand vectors and their operations
- Master matrix representations and operations
- Learn to set up systems of linear equations
- Understand determinants and their properties

---

## 1. Vectors

### 1.1 Theory

**Vectors** represent quantities with both magnitude and direction. In data science, vectors represent data points, features, or any ordered collection of numbers.

### 1.2 Mathematical Definition

A vector $\mathbf{v} \in \mathbb{R}^n$:
$$\mathbf{v} = (v_1, v_2, ..., v_n)$$

### 1.3 Vector Operations

| Operation | Definition | Formula |
|-----------|------------|---------|
| **Addition** | Element-wise sum | $\mathbf{u} + \mathbf{v} = (u_1 + v_1, ..., u_n + v_n)$ |
| **Scalar Multiplication** | Scale each element | $c\mathbf{v} = (cv_1, ..., cv_n)$ |
| **Dot Product** | Sum of element products | $\mathbf{u} \cdot \mathbf{v} = \sum_{i=1}^n u_i v_i$ |

### 1.4 Supply Chain Application

**Retail Context**:
- **Demand vectors** represent quantities demanded across products
- **Inventory vectors** track stock levels
- **Feature vectors** encode product attributes for ML models

---

## 2. Matrices

### 2.1 Theory

**Matrices** are 2D arrays of numbers. They represent linear transformations, data tables, and relationships between variables.

### 2.2 Mathematical Definition

Matrix $A \in \mathbb{R}^{m \times n}$:
$$A = \begin{pmatrix} a_{11} & \cdots & a_{1n} \\ \vdots & \ddots & \vdots \\ a_{m1} & \cdots & a_{mn} \end{pmatrix}$$

### 2.3 Matrix Operations

| Operation | Description |
|-----------|-------------|
| **Addition** | Element-wise (same dimensions required) |
| **Scalar Multiplication** | Multiply each element by scalar |
| **Matrix Multiplication** | $(AB)_{ij} = \sum_k a_{ik} b_{kj}$ |

### 2.4 Supply Chain Application

**Retail Context**:
- **Store-product matrices** show inventory levels
- **Cost matrices** represent shipping costs between locations
- **Transition matrices** model state changes (e.g., supply chain stages)

---

## 3. Systems of Linear Equations

### 3.1 Theory

Many real-world problems reduce to solving systems of linear equations. Matrix notation provides a compact representation.

### 3.2 Mathematical Definition

System $A\mathbf{x} = \mathbf{b}$:
$$\begin{cases} a_{11}x_1 + a_{12}x_2 + \cdots = b_1 \\ a_{21}x_1 + a_{22}x_2 + \cdots = b_2 \\ \vdots \end{cases}$$

Where:
- $A$ is the coefficient matrix
- $\mathbf{x}$ is the unknown vector
- $\mathbf{b}$ is the constant vector

### 3.3 Supply Chain Application

**Retail Context**:
- **Resource allocation**: How much to produce at each factory
- **Blending problems**: Mixing ingredients to meet specifications
- **Transportation problems**: Optimizing shipments between locations

---

## 4. Determinants

### 4.1 Theory

The **determinant** is a scalar value that encodes important properties of a matrix:
- Invertibility
- Volume scaling
- Orientation preservation

### 4.2 Mathematical Definition

**For 2×2 matrix**:
$$\det(A) = ad - bc \quad \text{where} \quad A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$$

**For n×n matrix**: Computed via cofactor expansion or row reduction.

### 4.3 Key Properties

| Property | Formula |
|----------|---------|
| **Product** | $\det(AB) = \det(A) \cdot \det(B)$ |
| **Inverse** | $\det(A^{-1}) = \frac{1}{\det(A)}$ |
| **Transpose** | $\det(A^T) = \det(A)$ |
| **Invertibility** | $A$ is invertible $\iff \det(A) \neq 0$ |

### 4.4 Supply Chain Application

**Retail Context**: Determinants indicate whether a system has a unique solution (non-zero determinant). In optimization, they help identify when constraints are independent.

---

## Summary Table

| Concept | Definition | Key Property | Supply Chain Application |
|---------|------------|--------------|--------------------------|
| **Vector** | Ordered collection in $\mathbb{R}^n$ | Dot product measures similarity | Demand/inventory tracking |
| **Matrix** | 2D array $\mathbb{R}^{m \times n}$ | Represents transformations | Store-product data |
| **Linear System** | $A\mathbf{x} = \mathbf{b}$ | Unique solution if $\det(A) \neq 0$ | Resource allocation |
| **Determinant** | Scalar from square matrix | Zero means singular | Constraint independence |

---

## Key Takeaways

1. **Vectors** are the building blocks - they represent data points and enable operations like similarity computation
2. **Matrices** organize and transform data - essential for representing relationships
3. **Linear systems** model constraints and resource allocation problems
4. **Determinants** reveal matrix properties - non-zero means unique solutions exist

---

## Next Week Preview

Week 2 covers **Solving Linear Equations**:
- Cramer's Rule
- Gaussian elimination
- Echelon forms

---

*IIT Madras BS Degree in Data Science*
