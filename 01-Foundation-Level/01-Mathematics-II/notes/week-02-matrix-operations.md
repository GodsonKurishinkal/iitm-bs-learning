---
Date: 2025-11-22
Course: BSMA1003 - Mathematics for Data Science II
Level: Foundation
Week: 2 of 11
Source: IIT Madras Mathematics for Data Science II Week 2
Topic Area: Linear Algebra - Matrix Operations and Systems of Equations
Tags: #BSMA1003 #LinearAlgebra #Matrices #Determinants #CramersRule #MatrixInverse #GaussianElimination #Week2 #Foundation
---

# Week 2: Matrix Operations and Solving Linear Systems

## Topics Covered

1. Determinants (Advanced Properties)
2. Cramer's Rule
3. Matrix Inverse
4. Echelon and Reduced Echelon Forms
5. Gaussian Elimination
6. Applications to Systems of Linear Equations

---

## Introduction

Matrices are fundamental data structures in data science, representing datasets, transformations, and relationships. This week builds on basic matrix operations to explore powerful techniques for solving linear systems—a core task in machine learning algorithms like linear regression, optimization, and neural networks.

**Why This Matters**: Every time you fit a linear regression model, the algorithm solves a system of linear equations. Understanding these solution methods gives you insight into computational complexity, numerical stability, and when certain ML algorithms will succeed or fail.

---

## 1. Determinants - Advanced Properties

### Definition Recap

For a square matrix $A$ of size $n \times n$, the **determinant** $\det(A)$ or $|A|$ is a scalar value with important geometric and algebraic properties.

**For $2 \times 2$ matrices:**
$$\det\begin{pmatrix} a & b \\ c & d \end{pmatrix} = ad - bc$$

**For $3 \times 3$ matrices** (cofactor expansion along first row):
$$\det\begin{pmatrix} a & b & c \\ d & e & f \\ g & h & i \end{pmatrix} = a(ei - fh) - b(di - fg) + c(dh - eg)$$

### Key Properties of Determinants

**Property 1: Row Operations**
- Swapping two rows changes sign: $\det(A') = -\det(A)$
- Multiplying a row by scalar $k$: $\det(A') = k \cdot \det(A)$
- Adding multiple of one row to another: $\det(A') = \det(A)$ (unchanged!)

**Property 2: Product Property**
$$\det(AB) = \det(A) \cdot \det(B)$$

**Property 3: Transpose**
$$\det(A^T) = \det(A)$$

**Property 4: Inverse**
If $A$ is invertible:
$$\det(A^{-1}) = \frac{1}{\det(A)}$$

**Property 5: Triangular Matrices**
For upper or lower triangular matrices, the determinant is the product of diagonal elements:
$$\det(A) = a_{11} \cdot a_{22} \cdot \ldots \cdot a_{nn}$$

### Geometric Interpretation

For a $2 \times 2$ matrix with column vectors $\mathbf{v}_1$ and $\mathbf{v}_2$:
$$|\det(A)| = \text{Area of parallelogram spanned by } \mathbf{v}_1 \text{ and } \mathbf{v}_2$$

For $3 \times 3$ matrices:
$$|\det(A)| = \text{Volume of parallelepiped spanned by column vectors}$$

**Sign of determinant**: Indicates orientation (positive = preserves orientation, negative = reverses it)

### Singularity and Invertibility

**Critical Theorem**: A matrix $A$ is invertible if and only if $\det(A) \neq 0$

- $\det(A) = 0$ ⟹ Matrix is **singular** (non-invertible)
- $\det(A) \neq 0$ ⟹ Matrix is **non-singular** (invertible)

---

## 2. Cramer's Rule

Cramer's Rule provides an explicit formula for solving systems of linear equations using determinants.

### The Rule

For system $A\mathbf{x} = \mathbf{b}$ where $A$ is $n \times n$ and $\det(A) \neq 0$:

$$x_i = \frac{\det(A_i)}{\det(A)}$$

where $A_i$ is the matrix formed by replacing the $i$-th column of $A$ with vector $\mathbf{b}$.

### Example: Solving 2×2 System

**Problem**: Solve the system:
$$\begin{cases}
2x + 3y = 8 \\
x - y = -1
\end{cases}$$

**Solution**:

**Step 1**: Write in matrix form $A\mathbf{x} = \mathbf{b}$
$$A = \begin{pmatrix} 2 & 3 \\ 1 & -1 \end{pmatrix}, \quad \mathbf{b} = \begin{pmatrix} 8 \\ -1 \end{pmatrix}$$

**Step 2**: Calculate $\det(A)$
$$\det(A) = 2(-1) - 3(1) = -2 - 3 = -5$$

Since $\det(A) \neq 0$, the system has a unique solution.

**Step 3**: Calculate $x$ using Cramer's Rule
$$A_1 = \begin{pmatrix} 8 & 3 \\ -1 & -1 \end{pmatrix}$$
$$\det(A_1) = 8(-1) - 3(-1) = -8 + 3 = -5$$
$$x = \frac{\det(A_1)}{\det(A)} = \frac{-5}{-5} = 1$$

**Step 4**: Calculate $y$
$$A_2 = \begin{pmatrix} 2 & 8 \\ 1 & -1 \end{pmatrix}$$
$$\det(A_2) = 2(-1) - 8(1) = -2 - 8 = -10$$
$$y = \frac{\det(A_2)}{\det(A)} = \frac{-10}{-5} = 2$$

**Answer**: $(x, y) = (1, 2)$

**Verification**:
- $2(1) + 3(2) = 2 + 6 = 8$ ✓
- $1 - 2 = -1$ ✓

### Limitations of Cramer's Rule

**Computational Complexity**:
- Computing determinants for $n \times n$ matrix: $O(n!)$ operations
- Impractical for large systems (n > 4)

**Better for**:
- Small systems (2×2, 3×3)
- Theoretical analysis
- Symbolic computation

**Not recommended for**:
- Large systems
- Numerical computation (use Gaussian elimination instead)

---

## 3. Matrix Inverse

### Definition

The **inverse** of an $n \times n$ matrix $A$, denoted $A^{-1}$, satisfies:
$$AA^{-1} = A^{-1}A = I$$

where $I$ is the identity matrix.

### Existence Condition

$A^{-1}$ exists if and only if $\det(A) \neq 0$

### Finding the Inverse (2×2 Case)

For $A = \begin{pmatrix} a & b \\ c & d \end{pmatrix}$:

$$A^{-1} = \frac{1}{\det(A)} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix} = \frac{1}{ad-bc} \begin{pmatrix} d & -b \\ -c & a \end{pmatrix}$$

**Recipe**:
1. Swap positions of $a$ and $d$
2. Negate $b$ and $c$
3. Multiply by $\frac{1}{\det(A)}$

### Example: 2×2 Inverse

**Problem**: Find the inverse of $A = \begin{pmatrix} 3 & 1 \\ 5 & 2 \end{pmatrix}$

**Solution**:

**Step 1**: Calculate determinant
$$\det(A) = 3(2) - 1(5) = 6 - 5 = 1$$

Since $\det(A) = 1 \neq 0$, the inverse exists.

**Step 2**: Apply formula
$$A^{-1} = \frac{1}{1} \begin{pmatrix} 2 & -1 \\ -5 & 3 \end{pmatrix} = \begin{pmatrix} 2 & -1 \\ -5 & 3 \end{pmatrix}$$

**Step 3**: Verify
$$AA^{-1} = \begin{pmatrix} 3 & 1 \\ 5 & 2 \end{pmatrix} \begin{pmatrix} 2 & -1 \\ -5 & 3 \end{pmatrix}$$
$$= \begin{pmatrix} 3(2)+1(-5) & 3(-1)+1(3) \\ 5(2)+2(-5) & 5(-1)+2(3) \end{pmatrix}$$
$$= \begin{pmatrix} 6-5 & -3+3 \\ 10-10 & -5+6 \end{pmatrix} = \begin{pmatrix} 1 & 0 \\ 0 & 1 \end{pmatrix}$$ ✓

### Finding Inverse Using Augmented Matrix (General Method)

For any invertible matrix $A$, we can find $A^{-1}$ using row operations:

**Procedure**:
1. Form augmented matrix $[A | I]$
2. Apply row operations to transform left side to $I$
3. The right side becomes $A^{-1}$: $[I | A^{-1}]$

### Example: 3×3 Inverse via Row Reduction

**Problem**: Find inverse of $A = \begin{pmatrix} 1 & 2 & 3 \\ 0 & 1 & 4 \\ 5 & 6 & 0 \end{pmatrix}$

**Solution**:

**Step 1**: Form $[A | I]$
$$\left[\begin{array}{ccc|ccc} 1 & 2 & 3 & 1 & 0 & 0 \\ 0 & 1 & 4 & 0 & 1 & 0 \\ 5 & 6 & 0 & 0 & 0 & 1 \end{array}\right]$$

**Step 2**: $R_3 \leftarrow R_3 - 5R_1$
$$\left[\begin{array}{ccc|ccc} 1 & 2 & 3 & 1 & 0 & 0 \\ 0 & 1 & 4 & 0 & 1 & 0 \\ 0 & -4 & -15 & -5 & 0 & 1 \end{array}\right]$$

**Step 3**: $R_3 \leftarrow R_3 + 4R_2$
$$\left[\begin{array}{ccc|ccc} 1 & 2 & 3 & 1 & 0 & 0 \\ 0 & 1 & 4 & 0 & 1 & 0 \\ 0 & 0 & 1 & -5 & 4 & 1 \end{array}\right]$$

**Step 4**: $R_2 \leftarrow R_2 - 4R_3$
$$\left[\begin{array}{ccc|ccc} 1 & 2 & 3 & 1 & 0 & 0 \\ 0 & 1 & 0 & 20 & -15 & -4 \\ 0 & 0 & 1 & -5 & 4 & 1 \end{array}\right]$$

**Step 5**: $R_1 \leftarrow R_1 - 3R_3$
$$\left[\begin{array}{ccc|ccc} 1 & 2 & 0 & 16 & -12 & -3 \\ 0 & 1 & 0 & 20 & -15 & -4 \\ 0 & 0 & 1 & -5 & 4 & 1 \end{array}\right]$$

**Step 6**: $R_1 \leftarrow R_1 - 2R_2$
$$\left[\begin{array}{ccc|ccc} 1 & 0 & 0 & -24 & 18 & 5 \\ 0 & 1 & 0 & 20 & -15 & -4 \\ 0 & 0 & 1 & -5 & 4 & 1 \end{array}\right]$$

**Answer**:
$$A^{-1} = \begin{pmatrix} -24 & 18 & 5 \\ 20 & -15 & -4 \\ -5 & 4 & 1 \end{pmatrix}$$

### Properties of Matrix Inverse

1. $(A^{-1})^{-1} = A$
2. $(AB)^{-1} = B^{-1}A^{-1}$ (reverse order!)
3. $(A^T)^{-1} = (A^{-1})^T$
4. $\det(A^{-1}) = \frac{1}{\det(A)}$

---

## 4. Echelon Forms

### Row Echelon Form (REF)

A matrix is in **row echelon form** if:
1. All nonzero rows are above any rows of all zeros
2. The first nonzero entry (pivot) of each row is to the right of the pivot of the row above
3. All entries in a column below a pivot are zeros

**Example**:
$$\begin{pmatrix} 1 & 2 & 3 & 4 \\ 0 & 1 & 5 & 6 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

### Reduced Row Echelon Form (RREF)

A matrix is in **reduced row echelon form** if it's in REF and additionally:
4. Each pivot equals 1 (leading 1)
5. Each pivot is the only nonzero entry in its column

**Example**:
$$\begin{pmatrix} 1 & 0 & 3 & 0 \\ 0 & 1 & 5 & 0 \\ 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \end{pmatrix}$$

**Uniqueness**: Every matrix has a **unique** RREF, but many possible REFs.

---

## 5. Gaussian Elimination

**Gaussian elimination** is a systematic algorithm for solving systems of linear equations by transforming the augmented matrix to row echelon form.

### Elementary Row Operations

Three types of operations that preserve solution set:

1. **Row Swap**: Exchange two rows ($R_i \leftrightarrow R_j$)
2. **Row Scaling**: Multiply row by nonzero constant ($R_i \leftarrow kR_i$, $k \neq 0$)
3. **Row Addition**: Add multiple of one row to another ($R_i \leftarrow R_i + kR_j$)

### Algorithm Steps

**Forward Elimination** (to REF):
1. Identify leftmost nonzero column (pivot column)
2. If needed, swap rows to get nonzero pivot
3. Use pivot to eliminate entries below it
4. Repeat for submatrix below current row

**Back Substitution** (to RREF):
5. Start from bottom, work upward
6. Make each pivot = 1
7. Eliminate entries above each pivot

### Example: Solving 3×3 System

**Problem**: Solve the system:
$$\begin{cases}
x + 2y + 3z = 9 \\
2x + y + z = 8 \\
3x + y + 2z = 11
\end{cases}$$

**Solution**:

**Step 1**: Form augmented matrix
$$\left[\begin{array}{ccc|c} 1 & 2 & 3 & 9 \\ 2 & 1 & 1 & 8 \\ 3 & 1 & 2 & 11 \end{array}\right]$$

**Step 2**: Eliminate below first pivot
$R_2 \leftarrow R_2 - 2R_1$:
$$\left[\begin{array}{ccc|c} 1 & 2 & 3 & 9 \\ 0 & -3 & -5 & -10 \\ 3 & 1 & 2 & 11 \end{array}\right]$$

$R_3 \leftarrow R_3 - 3R_1$:
$$\left[\begin{array}{ccc|c} 1 & 2 & 3 & 9 \\ 0 & -3 & -5 & -10 \\ 0 & -5 & -7 & -16 \end{array}\right]$$

**Step 3**: Eliminate below second pivot
$R_3 \leftarrow R_3 - \frac{5}{3}R_2$:
$$\left[\begin{array}{ccc|c} 1 & 2 & 3 & 9 \\ 0 & -3 & -5 & -10 \\ 0 & 0 & \frac{4}{3} & \frac{2}{3} \end{array}\right]$$

**Step 4**: Back substitution
From Row 3: $\frac{4}{3}z = \frac{2}{3}$ ⟹ $z = \frac{1}{2}$

From Row 2: $-3y - 5(\frac{1}{2}) = -10$ ⟹ $-3y = -10 + \frac{5}{2} = -\frac{15}{2}$ ⟹ $y = \frac{5}{2}$

From Row 1: $x + 2(\frac{5}{2}) + 3(\frac{1}{2}) = 9$ ⟹ $x + 5 + \frac{3}{2} = 9$ ⟹ $x = \frac{5}{2}$

**Answer**: $(x, y, z) = (\frac{5}{2}, \frac{5}{2}, \frac{1}{2})$

### Computational Complexity

For $n \times n$ system:
- **Gaussian elimination**: $O(n^3)$ operations
- **Cramer's rule**: $O(n! \cdot n)$ operations

Gaussian elimination is **vastly more efficient** for large systems!

---

## 6. Types of Solutions

### Case 1: Unique Solution

**Condition**: $\det(A) \neq 0$ (matrix is invertible)

**RREF**: Identity matrix on left side
$$\left[\begin{array}{ccc|c} 1 & 0 & 0 & a \\ 0 & 1 & 0 & b \\ 0 & 0 & 1 & c \end{array}\right]$$

### Case 2: No Solution (Inconsistent)

**Condition**: RREF has row like $[0 \, 0 \, 0 \, | \, k]$ where $k \neq 0$

**Example**:
$$\left[\begin{array}{ccc|c} 1 & 2 & 3 & 4 \\ 0 & 1 & 2 & 3 \\ 0 & 0 & 0 & 1 \end{array}\right]$$

Row 3 says: $0 = 1$ (contradiction!)

### Case 3: Infinitely Many Solutions

**Condition**: Free variables exist (more variables than pivot positions)

**Example**:
$$\left[\begin{array}{ccc|c} 1 & 2 & 0 & 3 \\ 0 & 0 & 1 & 2 \\ 0 & 0 & 0 & 0 \end{array}\right]$$

$x_2$ is a free variable. Solution: $x_1 = 3 - 2t$, $x_2 = t$, $x_3 = 2$ (where $t \in \mathbb{R}$)

---

## Data Science Applications

### 1. Linear Regression

**Problem**: Given data points $(x_1, y_1), \ldots, (x_n, y_n)$, find best-fit line $y = mx + b$.

**Matrix formulation**:
$$\begin{pmatrix} x_1 & 1 \\ x_2 & 1 \\ \vdots & \vdots \\ x_n & 1 \end{pmatrix} \begin{pmatrix} m \\ b \end{pmatrix} = \begin{pmatrix} y_1 \\ y_2 \\ \vdots \\ y_n \end{pmatrix}$$

Solving using **normal equations**: $(X^TX)\boldsymbol{\beta} = X^T\mathbf{y}$

This requires computing $(X^TX)^{-1}$ (matrix inverse!).

### 2. Image Processing

**Problem**: Solve $Ax = b$ where:
- $A$ represents blur operation (convolution matrix)
- $b$ is blurred image
- $x$ is original sharp image (deblurring/deconvolution)

Gaussian elimination helps recover original image.

### 3. Network Flow Problems

**Problem**: Determine flow through network nodes.

Example: Internet traffic routing, supply chain optimization

**Formulation**: System of linear equations where:
- Variables = flows through edges
- Equations = conservation of flow at nodes

### 4. Portfolio Optimization

**Problem**: Allocate investment across $n$ assets to achieve target returns with constraints.

**Matrix equation**: $A\mathbf{x} = \mathbf{r}$ where:
- $\mathbf{x}$ = allocation weights
- $A$ = return matrix
- $\mathbf{r}$ = target return vector

Plus constraints: $\sum x_i = 1$, $x_i \geq 0$

### 5. Multivariate Regression

For dataset with $p$ features and $n$ samples:

$$X\boldsymbol{\beta} = \mathbf{y}$$

where $X$ is $n \times (p+1)$, $\boldsymbol{\beta}$ is $(p+1) \times 1$, $\mathbf{y}$ is $n \times 1$

Solved using $(X^TX)^{-1}X^T\mathbf{y}$ (requires matrix inverse!).

---

## Python Implementation

```python
import numpy as np

# Example 1: Determinant
A = np.array([[3, 1], [5, 2]])
det_A = np.linalg.det(A)
print(f"Determinant: {det_A}")

# Example 2: Matrix Inverse
A_inv = np.linalg.inv(A)
print(f"Inverse:\n{A_inv}")

# Verify: A @ A_inv = I
I = A @ A_inv
print(f"A @ A^(-1):\n{np.round(I, 10)}")  # Round to handle floating point errors

# Example 3: Solving Linear System (A x = b)
A = np.array([[2, 3], [1, -1]])
b = np.array([8, -1])

# Method 1: Using solve (most efficient)
x = np.linalg.solve(A, b)
print(f"Solution using solve: {x}")

# Method 2: Using matrix inverse (less efficient)
x_inv = np.linalg.inv(A) @ b
print(f"Solution using inverse: {x_inv}")

# Method 3: Cramer's Rule (manual)
det_A = np.linalg.det(A)
A1 = np.array([[8, 3], [-1, -1]])
A2 = np.array([[2, 8], [1, -1]])
x_cramer = np.array([np.linalg.det(A1) / det_A, np.linalg.det(A2) / det_A])
print(f"Solution using Cramer's Rule: {x_cramer}")

# Example 4: Gaussian Elimination (manual)
from scipy.linalg import lu

A = np.array([[1, 2, 3], [2, 1, 1], [3, 1, 2]], dtype=float)
b = np.array([9, 8, 11], dtype=float)

# Form augmented matrix
augmented = np.column_stack([A, b])
print(f"Augmented matrix:\n{augmented}")

# Use NumPy's row operations (manual example)
# Create reduced row echelon form
def rref(matrix):
    """Compute reduced row echelon form"""
    M = matrix.copy()
    rows, cols = M.shape

    for i in range(min(rows, cols)):
        # Find pivot
        max_row = i + np.argmax(np.abs(M[i:, i]))
        M[[i, max_row]] = M[[max_row, i]]  # Swap rows

        if abs(M[i, i]) < 1e-10:
            continue

        # Scale pivot to 1
        M[i] = M[i] / M[i, i]

        # Eliminate column
        for j in range(rows):
            if j != i:
                M[j] = M[j] - M[j, i] * M[i]

    return M

rref_matrix = rref(augmented)
print(f"RREF:\n{np.round(rref_matrix, 6)}")

# Example 5: Linear Regression using Normal Equations
from sklearn.datasets import make_regression
from sklearn.linear_model import LinearRegression

# Generate synthetic data
X, y = make_regression(n_samples=100, n_features=3, noise=10, random_state=42)

# Add intercept column
X_with_intercept = np.column_stack([X, np.ones(len(X))])

# Solve using normal equations: beta = (X^T X)^(-1) X^T y
XtX = X_with_intercept.T @ X_with_intercept
Xty = X_with_intercept.T @ y
beta = np.linalg.solve(XtX, Xty)  # More stable than inv(XtX) @ Xty

print(f"Coefficients using normal equations: {beta}")

# Compare with scikit-learn
model = LinearRegression()
model.fit(X, y)
print(f"Coefficients from sklearn: {np.append(model.coef_, model.intercept_)}")
```

---

## Common Pitfalls and Misconceptions

### Pitfall 1: Using Cramer's Rule for Large Systems
**Misconception**: "Cramer's rule is the best way to solve systems."

**Reality**: Cramer's rule is $O(n!)$, impractical for $n > 4$. Use Gaussian elimination ($O(n^3)$) instead.

### Pitfall 2: Computing Inverse When Unnecessary
**Misconception**: "To solve $Ax = b$, first compute $A^{-1}$, then $x = A^{-1}b$."

**Reality**: `np.linalg.solve(A, b)` is faster and more numerically stable than `np.linalg.inv(A) @ b`.

### Pitfall 3: Assuming All Systems Have Unique Solutions
**Misconception**: "Every system $Ax = b$ has exactly one solution."

**Reality**: Three cases exist:
- Unique solution ($\det(A) \neq 0$)
- No solution (inconsistent system)
- Infinitely many solutions (underdetermined system)

### Pitfall 4: Ignoring Numerical Stability
**Misconception**: "All methods give the same numerical result."

**Reality**: Gaussian elimination with partial pivoting is more stable than direct inverse computation. Floating-point errors accumulate differently.

### Pitfall 5: Forgetting Reverse Order in $(AB)^{-1}$
**Misconception**: "$(AB)^{-1} = A^{-1}B^{-1}$"

**Reality**: $(AB)^{-1} = B^{-1}A^{-1}$ (reverse order!)

Think of putting on shoes then socks: to undo, remove socks first, then shoes.

### Pitfall 6: Treating Determinant as Distance
**Misconception**: "$\det(A)$ measures size of matrix."

**Reality**: Determinant measures **signed volume** of transformation. Negative determinants indicate orientation reversal.

### Pitfall 7: Confusing REF and RREF
**Misconception**: "Any row echelon form is unique."

**Reality**: REF is **not unique** (many possibilities), but RREF is **unique** for any matrix.

---

## Practice Problems

### Basic Level

1. **Determinant Practice**: Calculate determinants:
   - (a) $\begin{vmatrix} 5 & 2 \\ 1 & 3 \end{vmatrix}$
   - (b) $\begin{vmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 0 & 0 & 6 \end{vmatrix}$
   - (c) $\begin{vmatrix} 2 & -1 & 0 \\ 1 & 0 & 3 \\ 4 & 2 & 1 \end{vmatrix}$

2. **Matrix Inverse (2×2)**: Find inverses:
   - (a) $\begin{pmatrix} 4 & 3 \\ 3 & 2 \end{pmatrix}$
   - (b) $\begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$

3. **Cramer's Rule**: Solve using Cramer's Rule:
   $$\begin{cases} 3x + 2y = 7 \\ x - y = 1 \end{cases}$$

4. **Echelon Form Recognition**: Which matrices are in REF? RREF?
   - (a) $\begin{pmatrix} 1 & 2 & 3 \\ 0 & 1 & 4 \\ 0 & 0 & 0 \end{pmatrix}$
   - (b) $\begin{pmatrix} 1 & 0 & 2 \\ 0 & 1 & 3 \\ 0 & 0 & 1 \end{pmatrix}$

5. **Row Operations**: Apply $R_2 \leftarrow R_2 - 3R_1$ to:
   $$\begin{pmatrix} 1 & 2 & 3 \\ 3 & 5 & 7 \end{pmatrix}$$

### Intermediate Level

6. **System Solving**: Solve using Gaussian elimination:
   $$\begin{cases}
   x + y + z = 6 \\
   2x - y + z = 3 \\
   x + 2y - z = 4
   \end{cases}$$

7. **Matrix Inverse (3×3)**: Find the inverse using augmented matrix method:
   $$A = \begin{pmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{pmatrix}$$

8. **Determinant Properties**: If $\det(A) = 5$ and $A$ is $3 \times 3$, find:
   - (a) $\det(2A)$
   - (b) $\det(A^{-1})$
   - (c) $\det(A^T)$

9. **Solution Types**: Determine if each system has unique, infinite, or no solutions:
   - (a) $\begin{cases} x + y = 1 \\ 2x + 2y = 3 \end{cases}$
   - (b) $\begin{cases} x + y = 1 \\ 2x + 2y = 2 \end{cases}$

10. **RREF Computation**: Reduce to RREF:
    $$\begin{pmatrix} 1 & 2 & 1 & 3 \\ 2 & 4 & 3 & 7 \\ 1 & 2 & 2 & 5 \end{pmatrix}$$

### Advanced Level

11. **Inverse via Adjugate**: For $A = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix}$:
    - (a) Find cofactor matrix
    - (b) Find adjugate matrix (transpose of cofactor matrix)
    - (c) Verify $A^{-1} = \frac{1}{\det(A)} \text{adj}(A)$

12. **Parametric Systems**: Solve for all values of $k$:
    $$\begin{cases}
    x + 2y + z = 3 \\
    2x + 5y + 3z = 7 \\
    x + y + kz = 2
    \end{cases}$$
    For which values of $k$ does the system have:
    - Unique solution?
    - Infinitely many solutions?
    - No solution?

13. **Linear Regression**: Given data points $(1, 2)$, $(2, 4)$, $(3, 5)$:
    - Set up matrix equation $X\beta = y$ for line $y = mx + b$
    - Solve normal equations $(X^TX)\beta = X^Ty$
    - Find best-fit line

14. **Computational Efficiency**: For a $1000 \times 1000$ system:
    - Estimate operations for Cramer's Rule
    - Estimate operations for Gaussian Elimination
    - Which is faster and by what factor?

15. **Singular Matrix**: Given $A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \\ 1 & 1 & k \end{pmatrix}$:
    - For which value of $k$ is $A$ singular?
    - For that $k$, solve $Ax = 0$ (null space)

---

## Summary and Key Takeaways

1. **Determinants encode invertibility**: $\det(A) \neq 0 \Leftrightarrow A$ is invertible

2. **Cramer's Rule is theoretically elegant but computationally inefficient**: Use for small systems or symbolic computation only

3. **Matrix inverse exists iff $\det(A) \neq 0$**: Use augmented matrix method for computation

4. **Gaussian elimination is the workhorse**: $O(n^3)$ complexity, numerically stable, practical for large systems

5. **Three types of solutions**: unique (invertible), none (inconsistent), infinite (underdetermined)

6. **Row operations preserve solutions**: Build intuition for REF and RREF

7. **Avoid computing inverses unnecessarily**: `solve()` is faster than `inv() @ b`

8. **Linear systems are everywhere in ML**: regression, optimization, neural networks all rely on these techniques

---

## Connection to Next Week

Week 3 explores **vector spaces** and their structure:
- Vector space axioms and subspaces
- Linear independence (foundation for basis and dimension)
- Connection to solution spaces of $Ax = 0$

Understanding how to solve $Ax = 0$ (this week) prepares you for understanding **null spaces** and **column spaces** (next week)!

---

## References

- **Textbook**: Linear Algebra (IIT Madras), Chapters 3-4
- **Video Lectures**: Week 2 playlist
- **Strang, G.**: *Introduction to Linear Algebra*, Chapter 2
- **NumPy Documentation**: `numpy.linalg` module
- **SciPy Documentation**: `scipy.linalg` for advanced operations

---

**Last Updated**: 2025-11-22
**Next Topic**: Week 3 - Vector Spaces and Linear Independence
