# Week 4: Basis, Dimension, and Rank

---
Date: 2025-11-22
Course: BSMA1003 - Mathematics for Data Science II
Level: Foundation
Week: 4 of 11
Source: IIT Madras Mathematics II Week 4
Topic Area: Linear Algebra - Basis and Dimension
Tags: #BSMA1003 #LinearAlgebra #Week4 #Basis #Dimension #Rank #Foundation
---

## Topics Covered

1. **Basis of a Vector Space**
2. **Properties of Bases**
3. **Finding Bases for Vector Spaces**
4. **Span of Vectors**
5. **Rank of a Matrix**
6. **Dimension of a Vector Space**
7. **Rank-Nullity Theorem**
8. **Applications to Data Science**

---

## Key Concepts

### 1. Basis of a Vector Space

A **basis** for a vector space $V$ is a set of vectors $\{v_1, v_2, \ldots, v_n\}$ that satisfies two crucial properties:

1. **Linear Independence**: No vector in the set can be written as a linear combination of the others
2. **Spans the Space**: Every vector in $V$ can be expressed as a linear combination of the basis vectors

**Mathematical Definition**:
$$B = \{v_1, v_2, \ldots, v_n\} \text{ is a basis for } V \iff$$
- The vectors are linearly independent
- $\text{span}(B) = V$

**Why Basis Matters in Data Science**:
- **Feature Selection**: Basis vectors represent independent features in a dataset
- **Dimensionality Reduction**: Finding a smaller basis that captures essential information
- **Coordinate Systems**: Basis defines how we represent data points

#### Example 1: Standard Basis in $\mathbb{R}^3$

The **standard basis** for $\mathbb{R}^3$ is:
$$e_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad e_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad e_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

**Verification**:
1. **Linear Independence**: The only solution to $c_1e_1 + c_2e_2 + c_3e_3 = 0$ is $c_1 = c_2 = c_3 = 0$
2. **Spans $\mathbb{R}^3$**: Any vector $v = \begin{bmatrix} a \\ b \\ c \end{bmatrix} = ae_1 + be_2 + ce_3$

```python
import numpy as np

# Standard basis for R^3
e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

# Any vector can be expressed in this basis
v = np.array([3, -2, 5])
print(f"v = {v[0]}*e1 + {v[1]}*e2 + {v[2]}*e3")
# Output: v = 3*e1 + -2*e2 + 5*e3
```

### 2. Span of Vectors

The **span** of a set of vectors is the set of all possible linear combinations of those vectors.

**Definition**:
$$\text{span}\{v_1, v_2, \ldots, v_k\} = \{c_1v_1 + c_2v_2 + \cdots + c_kv_k \mid c_i \in \mathbb{R}\}$$

**Geometric Interpretation**:
- Span of one non-zero vector: A line through the origin
- Span of two non-collinear vectors: A plane through the origin
- Span of three non-coplanar vectors in $\mathbb{R}^3$: All of $\mathbb{R}^3$

#### Example 2: Computing Span

Find the span of vectors $v_1 = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}$ and $v_2 = \begin{bmatrix} 2 \\ 1 \\ 3 \end{bmatrix}$

**Solution**:
The span is all vectors of the form:
$$c_1\begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix} + c_2\begin{bmatrix} 2 \\ 1 \\ 3 \end{bmatrix} = \begin{bmatrix} c_1 + 2c_2 \\ 2c_1 + c_2 \\ c_1 + 3c_2 \end{bmatrix}$$

This represents a **plane** through the origin in $\mathbb{R}^3$.

```python
# Visualizing span of two vectors
v1 = np.array([1, 2, 1])
v2 = np.array([2, 1, 3])

# Generate points in the span
c1_vals = np.linspace(-2, 2, 10)
c2_vals = np.linspace(-2, 2, 10)
C1, C2 = np.meshgrid(c1_vals, c2_vals)

# Points in span: c1*v1 + c2*v2
X = C1 * v1[0] + C2 * v2[0]
Y = C1 * v1[1] + C2 * v2[1]
Z = C1 * v1[2] + C2 * v2[2]

# This creates a plane in 3D space
```

### 3. Finding Bases

**Procedure to Find a Basis**:

1. **Start with a spanning set**: Collect vectors that span the space
2. **Form a matrix**: Place vectors as columns
3. **Row reduce**: Convert to reduced row echelon form (RREF)
4. **Identify pivot columns**: These correspond to basis vectors
5. **Extract basis**: Take the original vectors from pivot column positions

#### Example 3: Finding a Basis for Column Space

Find a basis for the column space of matrix:
$$A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 7 \\ 1 & 2 & 4 \end{bmatrix}$$

**Solution**:

**Step 1**: Row reduce to identify pivot columns:
$$\text{RREF}(A) = \begin{bmatrix} 1 & 2 & 0 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{bmatrix}$$

**Step 2**: Pivot columns are 1 and 3

**Step 3**: Basis for $\text{Col}(A)$ is:
$$\left\{\begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}, \begin{bmatrix} 3 \\ 7 \\ 4 \end{bmatrix}\right\}$$

```python
from scipy.linalg import lu

A = np.array([[1, 2, 3],
              [2, 4, 7],
              [1, 2, 4]], dtype=float)

# Method 1: Using QR decomposition
Q, R = np.linalg.qr(A)
rank = np.linalg.matrix_rank(A)
basis = A[:, :rank]

print(f"Basis vectors:\n{basis}")
print(f"Number of basis vectors (dimension): {rank}")
```

### 4. Dimension of a Vector Space

The **dimension** of a vector space $V$ is the number of vectors in any basis for $V$.

**Key Theorem**: All bases for a vector space have the same number of vectors.

**Notation**: $\dim(V)$

**Examples**:
- $\dim(\mathbb{R}^n) = n$
- $\dim(\text{span}\{v_1, v_2\}) \leq 2$ (equals 2 if $v_1, v_2$ are independent)
- $\dim(\{0\}) = 0$ (the zero space)

#### Example 4: Dimension of Subspaces

Consider the subspace $W$ of $\mathbb{R}^4$ defined by:
$$W = \left\{\begin{bmatrix} x_1 \\ x_2 \\ x_3 \\ x_4 \end{bmatrix} \mid x_1 + x_2 = 0, \, x_3 - x_4 = 0\right\}$$

**Find $\dim(W)$**:

**Solution**:
From the constraints:
- $x_1 = -x_2$
- $x_3 = x_4$

General vector in $W$:
$$\begin{bmatrix} -x_2 \\ x_2 \\ x_4 \\ x_4 \end{bmatrix} = x_2\begin{bmatrix} -1 \\ 1 \\ 0 \\ 0 \end{bmatrix} + x_4\begin{bmatrix} 0 \\ 0 \\ 1 \\ 1 \end{bmatrix}$$

**Basis for $W$**:
$$\left\{\begin{bmatrix} -1 \\ 1 \\ 0 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 0 \\ 1 \\ 1 \end{bmatrix}\right\}$$

**Therefore**: $\dim(W) = 2$

### 5. Rank of a Matrix

The **rank** of a matrix $A$ is the dimension of its column space (equivalently, row space).

**Definition**:
$$\text{rank}(A) = \dim(\text{Col}(A)) = \dim(\text{Row}(A))$$

**Properties**:
1. $\text{rank}(A) \leq \min(m, n)$ for $m \times n$ matrix
2. $\text{rank}(A) = \text{rank}(A^T)$
3. $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$

**Computing Rank**: Count the number of pivot positions in RREF

#### Example 5: Computing Rank

Find the rank of:
$$B = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 2 & 4 & 6 & 8 \\ 1 & 3 & 4 & 5 \end{bmatrix}$$

**Solution**:

Row reduce:
$$\text{RREF}(B) = \begin{bmatrix} 1 & 0 & 1 & 2 \\ 0 & 1 & 1 & 1 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

**Number of pivot rows = 2**

**Therefore**: $\text{rank}(B) = 2$

```python
B = np.array([[1, 2, 3, 4],
              [2, 4, 6, 8],
              [1, 3, 4, 5]])

rank_B = np.linalg.matrix_rank(B)
print(f"Rank of B: {rank_B}")  # Output: 2

# Using SVD to understand rank
U, s, Vt = np.linalg.svd(B)
print(f"Singular values: {s}")
# Non-zero singular values count = rank
print(f"Non-zero singular values: {np.sum(s > 1e-10)}")
```

### 6. Rank-Nullity Theorem

**The Fundamental Theorem of Linear Algebra**:

For an $m \times n$ matrix $A$:
$$\text{rank}(A) + \text{nullity}(A) = n$$

Where:
- **Rank**: Dimension of column space
- **Nullity**: Dimension of null space (kernel)
- **$n$**: Number of columns

**Null Space (Kernel)**: $\text{Null}(A) = \{x \in \mathbb{R}^n \mid Ax = 0\}$

#### Example 6: Rank-Nullity Theorem Application

For matrix $A = \begin{bmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{bmatrix}$:

**Solution**:

**Step 1**: Find rank
$$\text{RREF}(A) = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 0 & 0 \end{bmatrix}$$
$\text{rank}(A) = 1$

**Step 2**: Apply rank-nullity theorem
$$\text{nullity}(A) = n - \text{rank}(A) = 3 - 1 = 2$$

**Step 3**: Find null space basis
Solve $Ax = 0$:
$$x_1 + 2x_2 + 3x_3 = 0$$
$$x_1 = -2x_2 - 3x_3$$

General solution:
$$\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix} = x_2\begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix} + x_3\begin{bmatrix} -3 \\ 0 \\ 1 \end{bmatrix}$$

**Basis for null space**:
$$\left\{\begin{bmatrix} -2 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} -3 \\ 0 \\ 1 \end{bmatrix}\right\}$$

**Verification**: $\dim(\text{Null}(A)) = 2$ ✓

```python
A = np.array([[1, 2, 3],
              [2, 4, 6]], dtype=float)

# Compute rank
rank = np.linalg.matrix_rank(A)
n_cols = A.shape[1]
nullity = n_cols - rank

print(f"Rank: {rank}")
print(f"Nullity: {nullity}")
print(f"Rank + Nullity = {rank + nullity} = n = {n_cols} ✓")

# Find null space basis using SVD
U, s, Vt = np.linalg.svd(A)
null_space = Vt[rank:, :].T
print(f"Null space basis:\n{null_space}")
```

---

## Definitions

- **Basis**: A linearly independent set of vectors that spans a vector space
- **Span**: The set of all linear combinations of a set of vectors
- **Linear Independence**: Vectors are linearly independent if no vector can be written as a linear combination of the others
- **Dimension**: The number of vectors in a basis for a vector space
- **Rank**: The dimension of the column space (or row space) of a matrix
- **Nullity**: The dimension of the null space of a matrix
- **Null Space (Kernel)**: The set of all vectors $x$ such that $Ax = 0$
- **Column Space**: The span of the columns of a matrix
- **Row Space**: The span of the rows of a matrix
- **Pivot Column**: A column containing a leading 1 in RREF

---

## Important Formulas

### Basis and Dimension
$$\text{A basis } B \text{ satisfies:}$$
1. $B$ is linearly independent
2. $\text{span}(B) = V$
3. $|B| = \dim(V)$

### Span
$$\text{span}\{v_1, \ldots, v_k\} = \left\{\sum_{i=1}^k c_iv_i \mid c_i \in \mathbb{R}\right\}$$

### Rank Properties
$$\text{rank}(A) = \text{rank}(A^T)$$
$$\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$$
$$\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$$

### Rank-Nullity Theorem
$$\boxed{\text{rank}(A) + \text{nullity}(A) = n}$$
For $A \in \mathbb{R}^{m \times n}$

### Dimension Properties
$$\dim(U + W) = \dim(U) + \dim(W) - \dim(U \cap W)$$

---

## Theorems & Proofs

### Theorem 1: Uniqueness of Dimension

**Statement**: If $V$ is a vector space with basis $B_1$ containing $m$ vectors and basis $B_2$ containing $n$ vectors, then $m = n$.

**Intuition**: The dimension is an intrinsic property of the space, not dependent on which basis we choose.

**Proof Sketch**:
1. Assume $m < n$
2. Since $B_1$ spans $V$, each vector in $B_2$ can be written as a linear combination of $B_1$
3. This creates a linear dependence among the $n$ vectors of $B_2$ (more vectors than dimensions)
4. Contradiction! $B_2$ must be independent
5. Therefore $m = n$

### Theorem 2: Basis Extension Theorem

**Statement**: Any linearly independent set in a finite-dimensional vector space can be extended to a basis.

**Application**: If you have some independent features, you can always find additional features to complete a basis.

### Theorem 3: Rank-Nullity Theorem (Detailed Proof)

**Statement**: For $A \in \mathbb{R}^{m \times n}$: $\text{rank}(A) + \text{nullity}(A) = n$

**Proof**:
1. Let $r = \text{rank}(A)$ and write $A$ in RREF
2. There are $r$ pivot columns (basis for column space)
3. There are $n - r$ free variables in $Ax = 0$
4. Each free variable gives one basis vector for null space
5. Therefore: $\text{nullity}(A) = n - r$
6. Thus: $r + (n - r) = n$ ✓

---

## Data Science Applications

### 1. Feature Selection and Dimensionality

**Problem**: Dataset with 100 features, but many are redundant (linearly dependent).

**Solution**: Find basis for feature space to identify independent features.

```python
# Real-world example: Feature correlation analysis
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create dataset with redundant features
np.random.seed(42)
X1 = np.random.randn(1000)
X2 = np.random.randn(1000)
X3 = 2*X1 + 3*X2  # Linearly dependent!
X4 = X1 + 0.1*np.random.randn(1000)  # Nearly dependent

data = np.column_stack([X1, X2, X3, X4])

# Compute rank to find effective dimension
rank = np.linalg.matrix_rank(data)
print(f"Number of features: {data.shape[1]}")
print(f"Effective dimension (rank): {rank}")
# Output: Rank ≈ 2 (X3 is dependent on X1, X2)

# Use PCA to find basis
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(data)
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
```

### 2. Image Compression with Low-Rank Approximation

**Concept**: Images can be approximated using low-rank matrices (fewer basis vectors).

```python
# Image compression using rank-k approximation
def compress_image(image, k):
    """Compress image using rank-k approximation"""
    U, s, Vt = np.linalg.svd(image, full_matrices=False)
    
    # Keep only top k singular values
    compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    
    # Compression ratio
    original_size = image.shape[0] * image.shape[1]
    compressed_size = k * (image.shape[0] + image.shape[1] + 1)
    ratio = compressed_size / original_size
    
    return compressed, ratio

# Example usage
# image = load_grayscale_image("photo.jpg")
# compressed_img, ratio = compress_image(image, k=50)
# print(f"Compression ratio: {ratio:.2%}")
```

### 3. Recommender Systems - Matrix Factorization

**Problem**: User-item rating matrix is high-dimensional and sparse.

**Solution**: Find low-rank approximation representing latent factors.

```python
# Collaborative filtering with low-rank matrix factorization
# User-item matrix: rows = users, columns = items

# Original matrix
ratings = np.array([
    [5, 3, 0, 1],
    [4, 0, 0, 1],
    [1, 1, 0, 5],
    [1, 0, 0, 4],
    [0, 1, 5, 4],
])

# Find low-rank approximation
U, s, Vt = np.linalg.svd(ratings, full_matrices=False)

# Use rank-2 approximation (2 latent factors)
k = 2
ratings_approx = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

print(f"Original rank: {np.linalg.matrix_rank(ratings)}")
print(f"Approximation rank: {k}")
print(f"Predicted ratings:\n{ratings_approx}")
```

### 4. Linear Regression and Multicollinearity

**Problem**: Multicollinearity occurs when features are linearly dependent, making coefficients unstable.

**Diagnosis**: Check rank of design matrix $X$.

```python
# Detecting multicollinearity
from sklearn.datasets import make_regression
from numpy.linalg import cond

# Create data with multicollinearity
X, y = make_regression(n_samples=100, n_features=5, noise=10, random_state=42)

# Add redundant feature
X_redundant = np.column_stack([X, 2*X[:, 0] + 3*X[:, 1]])

print(f"Original rank: {np.linalg.matrix_rank(X)}")
print(f"With redundant feature: {np.linalg.matrix_rank(X_redundant)}")

# Condition number indicates multicollinearity
print(f"Condition number (original): {cond(X):.2f}")
print(f"Condition number (redundant): {cond(X_redundant):.2e}")
# High condition number → multicollinearity problem!
```

### 5. Principal Component Analysis (PCA)

**Core Idea**: Find orthonormal basis (principal components) that captures maximum variance.

```python
# PCA from scratch using basis concepts
def pca_manual(X, n_components):
    """PCA implementation using eigen-decomposition"""
    # Center the data
    X_centered = X - np.mean(X, axis=0)
    
    # Compute covariance matrix
    cov = np.cov(X_centered.T)
    
    # Find eigenvectors (these form the basis!)
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    
    # Sort by eigenvalue (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvectors = eigenvectors[:, idx]
    
    # Select top n_components (basis vectors)
    basis = eigenvectors[:, :n_components]
    
    # Project data onto new basis
    X_transformed = X_centered @ basis
    
    return X_transformed, basis

# Usage
data = np.random.randn(100, 5)
X_pca, components = pca_manual(data, n_components=2)
print(f"Original dimension: {data.shape[1]}")
print(f"Reduced dimension: {X_pca.shape[1]}")
print(f"Basis vectors (principal components):\n{components}")
```

---

## Common Pitfalls

### Pitfall 1: Confusing Rank with Number of Columns

**Wrong**: "A $3 \times 4$ matrix has rank 4"
**Correct**: "Rank $\leq \min(3, 4) = 3$"

### Pitfall 2: Non-Unique Bases

**Misconception**: "There is only one basis for a vector space"
**Reality**: Infinitely many bases exist, but all have the same dimension

**Example**:
$\mathbb{R}^2$ bases:
- $\{(1,0), (0,1)\}$ ✓
- $\{(1,1), (1,-1)\}$ ✓
- $\{(2,3), (-1,4)\}$ ✓

### Pitfall 3: Dimension of Span

**Wrong**: "Span of 5 vectors in $\mathbb{R}^3$ has dimension 5"
**Correct**: "$\dim(\text{span}) \leq \min(5, 3) = 3$"

### Pitfall 4: Rank-Nullity Confusion

**Common Error**: Using $m$ (rows) instead of $n$ (columns) in rank-nullity theorem

**Remember**: $\text{rank}(A) + \text{nullity}(A) = n$ (number of **columns**)

### Pitfall 5: Assuming Independence

**Wrong**: "These vectors look different, so they're independent"
**Correct**: "Check by forming matrix and computing rank"

```python
# Visual inspection can be misleading
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 7])
v3 = np.array([3, 6, 10])  # = v1 + v2 (dependent!)

# Proper check
vectors = np.column_stack([v1, v2, v3])
rank = np.linalg.matrix_rank(vectors)
print(f"Rank: {rank}, Number of vectors: 3")
print(f"Independent: {rank == 3}")  # False!
```

---

## Practice Problems

### Basic Level

1. **Standard Basis**: Write down the standard basis for $\mathbb{R}^4$ and verify it satisfies the basis properties.

2. **Span**: Determine if vector $v = \begin{bmatrix} 3 \\ 6 \end{bmatrix}$ is in the span of $\left\{\begin{bmatrix} 1 \\ 2 \end{bmatrix}, \begin{bmatrix} 2 \\ 3 \end{bmatrix}\right\}$.

3. **Dimension**: Find the dimension of the subspace of $\mathbb{R}^3$ consisting of vectors $\begin{bmatrix} a \\ b \\ c \end{bmatrix}$ where $a + b + c = 0$.

4. **Rank**: Compute the rank of $A = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 1 & 4 \\ 0 & 0 & 0 \end{bmatrix}$.

5. **Nullity**: For the matrix in problem 4, find the nullity and verify the rank-nullity theorem.

### Intermediate Level

6. **Find a Basis**: Find a basis for the column space of:
   $$A = \begin{bmatrix} 1 & 2 & 0 & 1 \\ 2 & 4 & 1 & 3 \\ 1 & 2 & 1 & 2 \end{bmatrix}$$

7. **Dimension of Intersection**: Let $U = \text{span}\left\{\begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}\right\}$ and $W = \text{span}\left\{\begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}\right\}$. Find $\dim(U \cap W)$.

8. **Basis Extension**: Given independent vectors $v_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, v_2 = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}$ in $\mathbb{R}^3$, extend to a basis for $\mathbb{R}^3$.

9. **Null Space Basis**: Find a basis for the null space of:
   $$B = \begin{bmatrix} 1 & 2 & 3 & 4 \\ 2 & 4 & 6 & 8 \end{bmatrix}$$

10. **Rank Properties**: Prove that $\text{rank}(A) = \text{rank}(A^TA)$ for any matrix $A$.

### Advanced Level

11. **Dimension Formula**: Prove that for subspaces $U$ and $W$:
    $$\dim(U + W) = \dim(U) + \dim(W) - \dim(U \cap W)$$

12. **Rank Inequality**: Show that $\text{rank}(A + B) \leq \text{rank}(A) + \text{rank}(B)$.

13. **Coordinate Vectors**: Given basis $B = \{v_1, v_2, v_3\}$ for $\mathbb{R}^3$ where:
    $$v_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, v_2 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, v_3 = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}$$
    Find the coordinate vector $[u]_B$ for $u = \begin{bmatrix} 3 \\ 2 \\ 1 \end{bmatrix}$.

14. **Matrix Rank**: Given $\text{rank}(A) = r$ for an $m \times n$ matrix, prove that there exist invertible matrices $P$ and $Q$ such that:
    $$PAQ = \begin{bmatrix} I_r & 0 \\ 0 & 0 \end{bmatrix}$$

15. **Change of Basis**: Find the change-of-basis matrix from basis $B_1 = \{(1,0), (0,1)\}$ to $B_2 = \{(1,1), (1,-1)\}$ in $\mathbb{R}^2$.

---

## Self-Assessment Checklist

- [ ] Can you define what a basis is and explain its two key properties?
- [ ] Can you find a basis for a given vector space or subspace?
- [ ] Do you understand the relationship between span and basis?
- [ ] Can you compute the rank of a matrix using row reduction?
- [ ] Can you find the dimension of a vector space?
- [ ] Do you understand and can apply the rank-nullity theorem?
- [ ] Can you find a basis for the null space of a matrix?
- [ ] Can you extend a linearly independent set to a basis?
- [ ] Do you understand how basis and dimension relate to feature selection in ML?
- [ ] Can you apply these concepts to real data science problems?

---

## Key Takeaways

1. **Basis provides coordinates**: Every vector can be uniquely expressed in terms of basis vectors
2. **Dimension is intrinsic**: All bases have the same number of vectors
3. **Rank measures information**: Higher rank means more independent columns/rows
4. **Rank-nullity connects spaces**: Column space dimension + null space dimension = number of columns
5. **Applications everywhere**: Feature selection, PCA, matrix factorization all use these concepts
6. **Computational tools available**: NumPy and SciPy make calculations efficient

---

## References

- **Textbook**: Linear Algebra (IIT Madras), Chapters 4-5
- **Videos**: Week 4 lectures - Basis and Dimension
- **Python Libraries**: NumPy linear algebra documentation
- **Additional**: Strang, G. - *Introduction to Linear Algebra* (MIT)

---

## Connection to Next Week

Week 5 builds on basis concepts by introducing:
- **Norms**: Measuring lengths of basis vectors
- **Inner Products**: Angles between basis vectors
- **Orthogonal Bases**: Special bases where vectors are perpendicular (simplifies computations!)

Understanding basis and dimension is crucial for Week 5's orthogonality concepts!

---

**Last Updated**: 2025-11-22
**Next Week**: Vector Norms and Inner Products
