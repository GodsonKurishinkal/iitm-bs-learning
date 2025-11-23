# Week 6: Gram-Schmidt Process and QR Decomposition

---
Date: 2025-11-22
Course: BSMA1003 - Mathematics for Data Science II
Level: Foundation
Week: 6 of 11
Source: IIT Madras Mathematics II Week 6
Topic Area: Linear Algebra - Orthogonalization
Tags: #BSMA1003 #LinearAlgebra #Week6 #GramSchmidt #QR #Orthogonalization #Foundation
---

## Topics Covered

1. **Orthogonal and Orthonormal Vectors**
2. **Orthogonal Matrices**
3. **Gram-Schmidt Orthogonalization Process**
4. **Modified Gram-Schmidt**
5. **QR Decomposition**
6. **Applications to Least Squares**
7. **Computational Considerations**
8. **Machine Learning Applications**

---

## Key Concepts

### 1. Orthonormal Vectors

**Definition**: Vectors $\{u_1, u_2, \ldots, u_k\}$ are **orthonormal** if:
1. **Orthogonal**: $u_i \perp u_j$ for $i \neq j$ (i.e., $\langle u_i, u_j \rangle = 0$)
2. **Unit length**: $\|u_i\| = 1$ for all $i$

**Mathematical notation**:
$$\langle u_i, u_j \rangle = \delta_{ij} = \begin{cases} 1 & \text{if } i = j \\ 0 & \text{if } i \neq j \end{cases}$$

**Why orthonormal bases are special**:
- Simplify calculations dramatically
- Coordinates: $c_i = \langle v, u_i \rangle$ (no division needed!)
- Preserve lengths and angles
- Used in QR decomposition, eigenvalue algorithms, SVD

#### Example 1: Standard Basis is Orthonormal

The standard basis for $\mathbb{R}^3$:
$$e_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, \quad e_2 = \begin{bmatrix} 0 \\ 1 \\ 0 \end{bmatrix}, \quad e_3 = \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$

**Verification**:
- $\langle e_1, e_2 \rangle = 0$ ✓ (orthogonal)
- $\|e_1\| = 1$ ✓ (unit length)
- Similar for all pairs

```python
import numpy as np

e1 = np.array([1, 0, 0])
e2 = np.array([0, 1, 0])
e3 = np.array([0, 0, 1])

# Check orthonormality
print(f"<e1, e2> = {np.dot(e1, e2)}")  # 0
print(f"||e1|| = {np.linalg.norm(e1)}")  # 1.0
```

### 2. Gram-Schmidt Orthogonalization Process

The **Gram-Schmidt process** converts any basis into an orthogonal (or orthonormal) basis.

**Algorithm** (for vectors $v_1, v_2, \ldots, v_n$):

**Step 1**: Keep first vector
$$u_1 = v_1$$

**Step 2**: For each subsequent vector, subtract projections onto previous orthogonal vectors
$$u_2 = v_2 - \frac{\langle v_2, u_1 \rangle}{\langle u_1, u_1 \rangle} u_1$$

$$u_3 = v_3 - \frac{\langle v_3, u_1 \rangle}{\langle u_1, u_1 \rangle} u_1 - \frac{\langle v_3, u_2 \rangle}{\langle u_2, u_2 \rangle} u_2$$

**General formula**:
$$u_k = v_k - \sum_{j=1}^{k-1} \frac{\langle v_k, u_j \rangle}{\langle u_j, u_j \rangle} u_j$$

**Step 3**: Normalize to get orthonormal basis (optional)
$$e_k = \frac{u_k}{\|u_k\|}$$

#### Example 2: Gram-Schmidt in $\mathbb{R}^2$

Orthogonalize $v_1 = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$ and $v_2 = \begin{bmatrix} 2 \\ 2 \end{bmatrix}$

**Solution**:

**Step 1**: $u_1 = v_1 = \begin{bmatrix} 3 \\ 1 \end{bmatrix}$

**Step 2**: Compute projection coefficient
$$\frac{\langle v_2, u_1 \rangle}{\langle u_1, u_1 \rangle} = \frac{(2)(3) + (2)(1)}{3^2 + 1^2} = \frac{8}{10} = 0.8$$

Subtract projection:
$$u_2 = v_2 - 0.8 u_1 = \begin{bmatrix} 2 \\ 2 \end{bmatrix} - 0.8\begin{bmatrix} 3 \\ 1 \end{bmatrix} = \begin{bmatrix} -0.4 \\ 1.2 \end{bmatrix}$$

**Step 3** (normalize):
$$e_1 = \frac{u_1}{\|u_1\|} = \frac{1}{\sqrt{10}}\begin{bmatrix} 3 \\ 1 \end{bmatrix}$$
$$e_2 = \frac{u_2}{\|u_2\|} = \frac{1}{\sqrt{1.6}}\begin{bmatrix} -0.4 \\ 1.2 \end{bmatrix}$$

```python
v1 = np.array([3, 1], dtype=float)
v2 = np.array([2, 2], dtype=float)

# Gram-Schmidt
u1 = v1
u2 = v2 - (np.dot(v2, u1) / np.dot(u1, u1)) * u1

# Normalize
e1 = u1 / np.linalg.norm(u1)
e2 = u2 / np.linalg.norm(u2)

# Verify orthogonality
print(f"<e1, e2> = {np.dot(e1, e2):.10f}")  # ~0
print(f"||e1|| = {np.linalg.norm(e1)}")  # 1.0
print(f"||e2|| = {np.linalg.norm(e2)}")  # 1.0
```

#### Example 3: Gram-Schmidt in $\mathbb{R}^3$

Orthogonalize:
$$v_1 = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}, \quad v_3 = \begin{bmatrix} 0 \\ 1 \\ 1 \end{bmatrix}$$

**Solution**:

**u1**: $u_1 = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$

**u2**: 
$$\frac{\langle v_2, u_1 \rangle}{\|u_1\|^2} = \frac{2}{3}$$
$$u_2 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix} - \frac{2}{3}\begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1/3 \\ -2/3 \\ 1/3 \end{bmatrix}$$

**u3**: 
$$u_3 = v_3 - \frac{\langle v_3, u_1 \rangle}{\|u_1\|^2}u_1 - \frac{\langle v_3, u_2 \rangle}{\|u_2\|^2}u_2$$

(Computation yields orthogonal $u_3$)

```python
def gram_schmidt(vectors):
    """Gram-Schmidt orthogonalization"""
    basis = []
    for v in vectors:
        u = v.copy()
        for b in basis:
            u -= np.dot(v, b) / np.dot(b, b) * b
        if np.linalg.norm(u) > 1e-10:  # Avoid zero vectors
            basis.append(u / np.linalg.norm(u))  # Normalize
    return np.array(basis).T

v1 = np.array([1, 1, 1], dtype=float)
v2 = np.array([1, 0, 1], dtype=float)
v3 = np.array([0, 1, 1], dtype=float)

Q = gram_schmidt([v1, v2, v3])
print(f"Orthonormal basis:\n{Q}")

# Verify: Q^T Q = I
print(f"\nQ^T Q =\n{Q.T @ Q}")
```

### 3. QR Decomposition

Every matrix $A$ with linearly independent columns can be factored as:
$$A = QR$$

Where:
- **Q**: Orthogonal matrix (columns form orthonormal basis)
- **R**: Upper triangular matrix

**Properties**:
- $Q^TQ = I$ (orthogonal matrix property)
- $R$ has diagonal entries equal to norms of Gram-Schmidt vectors
- Unique if diagonal of $R$ is positive

#### Example 4: QR Decomposition

Find QR decomposition of:
$$A = \begin{bmatrix} 1 & 1 \\ 1 & 0 \\ 0 & 1 \end{bmatrix}$$

**Solution**:

**Step 1**: Apply Gram-Schmidt to columns
$$v_1 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \quad v_2 = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$$

**Step 2**: Get Q (orthonormal columns)

**Step 3**: Compute $R = Q^TA$

```python
from scipy.linalg import qr

A = np.array([[1, 1],
              [1, 0],
              [0, 1]], dtype=float)

# QR decomposition
Q, R = qr(A)

print(f"Q =\n{Q}")
print(f"\nR =\n{R}")
print(f"\nVerify A = QR:\n{Q @ R}")
print(f"\nVerify Q^T Q = I:\n{Q.T @ Q}")
```

### 4. Applications to Least Squares

**Problem**: Solve overdetermined system $Ax = b$ (more equations than unknowns)

**Solution using QR**:

If $A = QR$, then:
$$Ax = b \Rightarrow QRx = b$$
$$Rx = Q^Tb$$

Since $R$ is upper triangular, solve by back-substitution!

**Advantages**:
- More numerically stable than normal equations
- Avoids computing $A^TA$ (condition number squared)
- Direct algorithm

#### Example 5: Least Squares via QR

Fit line $y = mx + c$ to points: $(0, 1), (1, 2), (2, 4)$

**Solution**:

Design matrix:
$$A = \begin{bmatrix} 0 & 1 \\ 1 & 1 \\ 2 & 1 \end{bmatrix}, \quad b = \begin{bmatrix} 1 \\ 2 \\ 4 \end{bmatrix}$$

```python
# Data points
x_data = np.array([0, 1, 2])
y_data = np.array([1, 2, 4])

# Design matrix for y = mx + c
A = np.column_stack([x_data, np.ones_like(x_data)])

# QR decomposition
Q, R = qr(A)

# Solve Rx = Q^T b
coeffs = np.linalg.solve(R, Q.T @ y_data)

m, c = coeffs
print(f"Best fit line: y = {m:.2f}x + {c:.2f}")

# Compare with normal equations
coeffs_normal = np.linalg.lstsq(A, y_data, rcond=None)[0]
print(f"Normal equations: y = {coeffs_normal[0]:.2f}x + {coeffs_normal[1]:.2f}")
```

#### Example 6: Polynomial Fitting

Fit quadratic $y = ax^2 + bx + c$ to noisy data

```python
# Generate noisy quadratic data
np.random.seed(42)
x = np.linspace(0, 10, 20)
y_true = 2*x**2 - 3*x + 1
y_noisy = y_true + 5*np.random.randn(20)

# Design matrix for quadratic
A = np.column_stack([x**2, x, np.ones_like(x)])

# QR least squares
Q, R = qr(A)
coeffs = np.linalg.solve(R, Q.T @ y_noisy)

print(f"Fitted: y = {coeffs[0]:.2f}x² + {coeffs[1]:.2f}x + {coeffs[2]:.2f}")
print(f"True:   y = 2.00x² - 3.00x + 1.00")
```

### 5. Modified Gram-Schmidt

**Problem**: Classical Gram-Schmidt is numerically unstable for nearly dependent vectors

**Solution**: Modified Gram-Schmidt algorithm

**Difference**: Subtract projections immediately after computing each new vector

```python
def modified_gram_schmidt(A):
    """More numerically stable Gram-Schmidt"""
    m, n = A.shape
    Q = A.copy().astype(float)
    R = np.zeros((n, n))
    
    for j in range(n):
        # Norm of current column
        R[j, j] = np.linalg.norm(Q[:, j])
        Q[:, j] = Q[:, j] / R[j, j]
        
        # Subtract projection from remaining columns
        for k in range(j+1, n):
            R[j, k] = np.dot(Q[:, j], Q[:, k])
            Q[:, k] = Q[:, k] - R[j, k] * Q[:, j]
    
    return Q, R

# Test
A = np.array([[1, 1, 0],
              [1, 0, 1],
              [0, 1, 1]], dtype=float)

Q, R = modified_gram_schmidt(A)
print(f"Q:\n{Q}")
print(f"\nR:\n{R}")
print(f"\nError ||A - QR||: {np.linalg.norm(A - Q @ R)}")
```

### 6. Orthogonal Matrices

A square matrix $Q$ is **orthogonal** if:
$$Q^TQ = QQ^T = I$$

**Properties**:
- Columns form orthonormal basis
- Rows form orthonormal basis
- Preserves lengths: $\|Qx\| = \|x\|$
- Preserves angles: $\langle Qx, Qy \rangle = \langle x, y \rangle$
- $\det(Q) = \pm 1$

#### Example 7: Rotation Matrix (Orthogonal)

$$Q = \begin{bmatrix} \cos\theta & -\sin\theta \\ \sin\theta & \cos\theta \end{bmatrix}$$

```python
theta = np.pi / 4  # 45 degrees
Q = np.array([[np.cos(theta), -np.sin(theta)],
              [np.sin(theta),  np.cos(theta)]])

# Verify orthogonal
print(f"Q^T Q =\n{Q.T @ Q}")  # Identity matrix
print(f"det(Q) = {np.linalg.det(Q)}")  # 1

# Preserves length
v = np.array([3, 4])
print(f"||v|| = {np.linalg.norm(v)}")
print(f"||Qv|| = {np.linalg.norm(Q @ v)}")  # Same!
```

---

## Important Formulas

### Gram-Schmidt Process
$$u_k = v_k - \sum_{j=1}^{k-1} \frac{\langle v_k, u_j \rangle}{\langle u_j, u_j \rangle} u_j$$

### Normalization
$$e_k = \frac{u_k}{\|u_k\|}$$

### QR Decomposition
$$A = QR$$
where $Q^TQ = I$ and $R$ is upper triangular

### Least Squares via QR
$$x = R^{-1}Q^Tb$$

### Orthogonal Matrix
$$Q^TQ = I \iff Q^T = Q^{-1}$$

---

## Theorems & Proofs

### Theorem 1: Gram-Schmidt Produces Orthogonal Basis

**Statement**: The Gram-Schmidt process applied to linearly independent vectors produces an orthogonal basis.

**Proof** (by induction):
- Base case: $u_1 = v_1$ is orthogonal to itself
- Inductive step: Assume $u_1, \ldots, u_{k-1}$ are orthogonal
- Show $u_k$ is orthogonal to each $u_j$ for $j < k$:
  $$\langle u_k, u_j \rangle = \langle v_k - \sum_{i=1}^{k-1} \frac{\langle v_k, u_i \rangle}{\|u_i\|^2}u_i, u_j \rangle = 0$$

### Theorem 2: QR Decomposition Existence

**Statement**: Every matrix $A$ with linearly independent columns has a QR decomposition.

**Proof**: Apply Gram-Schmidt to columns of $A$ to get $Q$. Then $R = Q^TA$.

### Theorem 3: Orthogonal Matrices Preserve Inner Products

**Statement**: If $Q$ is orthogonal, then $\langle Qx, Qy \rangle = \langle x, y \rangle$

**Proof**:
$$\langle Qx, Qy \rangle = (Qx)^T(Qy) = x^TQ^TQy = x^TIy = x^Ty = \langle x, y \rangle$$

---

## Data Science Applications

### 1. Principal Component Analysis (PCA)

PCA uses QR decomposition in the algorithm:

```python
def pca_via_qr(X, n_components):
    """PCA using QR decomposition"""
    # Center data
    X_centered = X - np.mean(X, axis=0)
    
    # QR decomposition
    Q, R = qr(X_centered.T)
    
    # Principal components are columns of Q
    components = Q[:, :n_components]
    
    # Project data
    X_transformed = X_centered @ components
    
    return X_transformed, components

# Usage
X = np.random.randn(100, 5)
X_pca, pcs = pca_via_qr(X, n_components=2)
print(f"Reduced from {X.shape[1]} to {X_pca.shape[1]} dimensions")
```

### 2. Linear Regression (Numerical Stability)

```python
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Generate ill-conditioned data
X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)

# Method 1: Normal equations (can be unstable)
# coeffs = np.linalg.inv(X.T @ X) @ X.T @ y  # Don't do this!

# Method 2: QR decomposition (stable)
Q, R = qr(X)
coeffs_qr = np.linalg.solve(R, Q.T @ y)

# Method 3: SVD (most stable, but slower)
coeffs_svd = np.linalg.lstsq(X, y, rcond=None)[0]

print(f"QR coefficients: {coeffs_qr[:3]}")
print(f"SVD coefficients: {coeffs_svd[:3]}")
print(f"Difference: {np.linalg.norm(coeffs_qr - coeffs_svd)}")
```

### 3. Orthogonal Feature Extraction

```python
def orthogonalize_features(X):
    """Create orthogonal features using Gram-Schmidt"""
    Q = gram_schmidt(X.T)
    return Q.T

# Original features (possibly correlated)
X_original = np.random.randn(100, 5)
X_original[:, 1] = X_original[:, 0] + 0.5*np.random.randn(100)  # Correlated

# Orthogonalize
X_ortho = orthogonalize_features(X_original)

# Check correlation
print("Original correlation:\n", np.corrcoef(X_original.T))
print("\nOrthogonalized correlation:\n", np.corrcoef(X_ortho.T))
```

### 4. QR Algorithm for Eigenvalues

```python
def qr_algorithm(A, max_iter=100):
    """Find eigenvalues using QR algorithm"""
    Ak = A.copy()
    for i in range(max_iter):
        Q, R = qr(Ak)
        Ak = R @ Q  # Key step!
    
    # Diagonal contains eigenvalues
    eigenvalues = np.diag(Ak)
    return eigenvalues

# Symmetric matrix
A = np.array([[4, 1], [1, 3]], dtype=float)
eigs_qr = qr_algorithm(A)
eigs_true = np.linalg.eigvals(A)

print(f"QR algorithm: {np.sort(eigs_qr)}")
print(f"True eigenvalues: {np.sort(eigs_true)}")
```

### 5. Orthogonal Regression (Total Least Squares)

```python
def orthogonal_distance_regression(X, y):
    """Orthogonal regression minimizes perpendicular distances"""
    # Combine X and y
    data = np.column_stack([X, y])
    
    # Center data
    data_centered = data - np.mean(data, axis=0)
    
    # QR decomposition
    Q, R = qr(data_centered.T)
    
    # Last column of Q is perpendicular to best-fit hyperplane
    normal = Q[:, -1]
    
    # Coefficients
    coeffs = -normal[:-1] / normal[-1]
    intercept = np.mean(y) - coeffs @ np.mean(X, axis=0)
    
    return coeffs, intercept

# Usage
X = np.random.randn(100, 2)
y = 3*X[:, 0] - 2*X[:, 1] + np.random.randn(100)

coeffs, intercept = orthogonal_distance_regression(X, y)
print(f"Coefficients: {coeffs}")
print(f"Intercept: {intercept}")
```

---

## Common Pitfalls

### Pitfall 1: Numerical Instability in Classical Gram-Schmidt

**Problem**: Rounding errors accumulate
**Solution**: Use modified Gram-Schmidt or QR from library

### Pitfall 2: Not Checking Linear Independence

**Problem**: Gram-Schmidt fails if vectors are dependent
**Solution**: Check rank before applying

```python
def safe_gram_schmidt(vectors):
    rank = np.linalg.matrix_rank(np.column_stack(vectors))
    if rank < len(vectors):
        print(f"Warning: Only {rank} independent vectors out of {len(vectors)}")
    return gram_schmidt(vectors)
```

### Pitfall 3: Confusing Q and R

**Wrong**: Thinking R is always square
**Correct**: For $m \times n$ matrix with $m > n$, R is $n \times n$

### Pitfall 4: Normal Equations vs QR

**Wrong**: Always using normal equations $(A^TA)x = A^Tb$
**Better**: Use QR for numerical stability

**Condition number**: Normal equations square the condition number!

### Pitfall 5: Forgetting to Normalize

**Problem**: Gram-Schmidt gives orthogonal vectors, not orthonormal
**Solution**: Divide by norm at each step

---

## Practice Problems

### Basic Level

1. Verify that $\begin{bmatrix} 1 \\ 1 \\ -1 \\ -1 \end{bmatrix}$ and $\begin{bmatrix} 1 \\ -1 \\ 1 \\ -1 \end{bmatrix}$ are orthogonal

2. Normalize the vector $v = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$ to create a unit vector

3. Apply Gram-Schmidt to $v_1 = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$ and $v_2 = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$

4. Find the QR decomposition of $A = \begin{bmatrix} 1 & 0 \\ 0 & 1 \\ 0 & 0 \end{bmatrix}$

5. Verify that a rotation matrix is orthogonal

### Intermediate Level

6. Orthogonalize the vectors:
   $$v_1 = \begin{bmatrix} 1 \\ 0 \\ 0 \end{bmatrix}, v_2 = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, v_3 = \begin{bmatrix} 1 \\ 1 \\ 1 \end{bmatrix}$$

7. Use QR decomposition to solve the least squares problem for $Ax = b$ where:
   $$A = \begin{bmatrix} 1 & 1 \\ 1 & 2 \\ 1 & 3 \end{bmatrix}, b = \begin{bmatrix} 1 \\ 2 \\ 2 \end{bmatrix}$$

8. Show that the product of two orthogonal matrices is orthogonal

9. Find an orthonormal basis for the subspace spanned by:
   $$\begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}, \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$$

10. Prove that if $Q$ is orthogonal, then $\det(Q) = \pm 1$

### Advanced Level

11. Implement modified Gram-Schmidt and compare numerical stability with classical version

12. Prove that QR decomposition is unique if we require diagonal entries of R to be positive

13. Show that for orthogonal matrix $Q$: $\|Qx\|_2 = \|x\|_2$ (isometry)

14. Use QR decomposition to find eigenvalues of a symmetric matrix (QR algorithm)

15. Prove the connection between Gram-Schmidt and QR decomposition

---

## Self-Assessment Checklist

- [ ] Can you explain what orthonormal vectors are?
- [ ] Can you apply Gram-Schmidt process to a set of vectors?
- [ ] Can you compute QR decomposition?
- [ ] Do you understand why QR is more stable than normal equations?
- [ ] Can you use QR for least squares problems?
- [ ] Do you understand modified Gram-Schmidt?
- [ ] Can you recognize orthogonal matrices and their properties?
- [ ] Can you apply these concepts to ML problems?

---

## Key Takeaways

1. **Orthonormal bases simplify computation**: Coordinates obtained by simple dot products
2. **Gram-Schmidt creates orthogonal bases**: From any linearly independent set
3. **QR decomposition factors matrices**: $A = QR$ with orthogonal Q and triangular R
4. **QR is numerically stable**: Better than normal equations for least squares
5. **Orthogonal matrices preserve geometry**: Lengths, angles, and inner products
6. **Modified Gram-Schmidt is more stable**: Use in practice over classical version

---

## References

- **Textbook**: Linear Algebra (IIT Madras), Chapter 7
- **Videos**: Week 6 lectures - Gram-Schmidt and QR
- **Additional**: Trefethen & Bau - *Numerical Linear Algebra*

---

## Connection to Next Week

Week 7 transitions to calculus:
- **Single Variable Calculus**: Derivatives and optimization
- **Gradient descent**: Uses derivatives to find minima
- **Connection**: QR decomposition used in optimization algorithms

Understanding orthogonal decompositions helps in optimization!

---

**Last Updated**: 2025-11-22
**Next Week**: Single Variable Calculus - Derivatives and Optimization
