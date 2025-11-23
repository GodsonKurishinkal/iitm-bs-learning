---
Date: 2025-11-22
Course: BSMA1003 - Mathematics for Data Science II
Level: Foundation
Week: 3 of 11
Source: IIT Madras Mathematics for Data Science II Week 3
Topic Area: Linear Algebra - Vector Spaces and Linear Independence
Tags: #BSMA1003 #LinearAlgebra #VectorSpaces #LinearIndependence #Subspaces #Week3 #Foundation
---

# Week 3: Vector Spaces and Linear Independence

## Topics Covered

1. Definition of Vector Spaces
2. Vector Space Axioms
3. Properties of Vector Spaces
4. Subspaces
5. Linear Dependence and Independence
6. Testing for Linear Independence

---

## Introduction

Vector spaces are the foundation of linear algebra and provide the mathematical framework for data science. Every dataset can be viewed as a collection of vectors in a high-dimensional space. Understanding vector spaces helps you grasp dimensionality reduction (PCA), feature spaces in machine learning, and the geometry underlying optimization algorithms.

**Why This Matters**: When you train a neural network or cluster data points, you're working in vector spaces. Linear independence determines if your features are redundant, and subspaces represent the span of possible model predictions.

---

## 1. Vector Spaces - Definition

A **vector space** $V$ over a field $\mathbb{F}$ (usually $\mathbb{R}$ or $\mathbb{C}$) is a set equipped with two operations:
- **Vector addition**: $\mathbf{u} + \mathbf{v} \in V$
- **Scalar multiplication**: $c\mathbf{v} \in V$ for $c \in \mathbb{F}$

satisfying certain axioms (see next section).

### Common Examples

**Example 1: $\mathbb{R}^n$**
The set of all $n$-tuples of real numbers: $\mathbb{R}^n = \{(x_1, x_2, \ldots, x_n) : x_i \in \mathbb{R}\}$

Operations:
- Addition: $(x_1, \ldots, x_n) + (y_1, \ldots, y_n) = (x_1+y_1, \ldots, x_n+y_n)$
- Scalar multiplication: $c(x_1, \ldots, x_n) = (cx_1, \ldots, cx_n)$

**Example 2: Matrix Space $M_{m \times n}(\mathbb{R})$**
All $m \times n$ matrices with real entries.

**Example 3: Polynomial Space $P_n$**
All polynomials of degree at most $n$: $P_n = \{a_0 + a_1x + \cdots + a_nx^n : a_i \in \mathbb{R}\}$

**Example 4: Function Space $C[a,b]$**
All continuous real-valued functions on interval $[a, b]$.

**Example 5: Solution Space**
Set of all solutions to homogeneous system $A\mathbf{x} = \mathbf{0}$ forms a vector space.

---

## 2. Vector Space Axioms

A set $V$ is a **vector space** if it satisfies these 10 axioms:

### Addition Axioms

**A1. Closure under addition**: If $\mathbf{u}, \mathbf{v} \in V$, then $\mathbf{u} + \mathbf{v} \in V$

**A2. Commutativity**: $\mathbf{u} + \mathbf{v} = \mathbf{v} + \mathbf{u}$

**A3. Associativity**: $(\mathbf{u} + \mathbf{v}) + \mathbf{w} = \mathbf{u} + (\mathbf{v} + \mathbf{w})$

**A4. Zero vector**: There exists $\mathbf{0} \in V$ such that $\mathbf{v} + \mathbf{0} = \mathbf{v}$ for all $\mathbf{v} \in V$

**A5. Additive inverse**: For each $\mathbf{v} \in V$, there exists $-\mathbf{v} \in V$ such that $\mathbf{v} + (-\mathbf{v}) = \mathbf{0}$

### Scalar Multiplication Axioms

**S1. Closure under scalar multiplication**: If $\mathbf{v} \in V$ and $c \in \mathbb{F}$, then $c\mathbf{v} \in V$

**S2. Distributivity (vector)**: $c(\mathbf{u} + \mathbf{v}) = c\mathbf{u} + c\mathbf{v}$

**S3. Distributivity (scalar)**: $(c + d)\mathbf{v} = c\mathbf{v} + d\mathbf{v}$

**S4. Associativity**: $(cd)\mathbf{v} = c(d\mathbf{v})$

**S5. Identity**: $1\mathbf{v} = \mathbf{v}$

### Verification Example

**Problem**: Verify $\mathbb{R}^2$ is a vector space.

**Solution**: Check all 10 axioms with $\mathbf{u} = (u_1, u_2)$, $\mathbf{v} = (v_1, v_2)$

**A1**: $(u_1, u_2) + (v_1, v_2) = (u_1+v_1, u_2+v_2) \in \mathbb{R}^2$ ✓

**A2**: Component-wise addition is commutative ✓

**A3**: Component-wise addition is associative ✓

**A4**: $\mathbf{0} = (0, 0)$ satisfies $(u_1, u_2) + (0, 0) = (u_1, u_2)$ ✓

**A5**: $-\mathbf{v} = (-v_1, -v_2)$ gives $(v_1, v_2) + (-v_1, -v_2) = (0, 0)$ ✓

**S1**: $c(v_1, v_2) = (cv_1, cv_2) \in \mathbb{R}^2$ ✓

**S2-S5**: Follow from properties of real number multiplication ✓

---

## 3. Properties of Vector Spaces

From the axioms, we can derive useful properties:

**Property 1: Uniqueness of zero vector**
The zero vector $\mathbf{0}$ is unique in any vector space.

**Proof**: Suppose $\mathbf{0}'$ is another zero vector. Then:
$$\mathbf{0} = \mathbf{0} + \mathbf{0}' = \mathbf{0}'$$
(First equality uses $\mathbf{0}'$ as zero, second uses $\mathbf{0}$ as zero)

**Property 2: Uniqueness of additive inverse**
For each $\mathbf{v}$, the additive inverse $-\mathbf{v}$ is unique.

**Property 3: Zero scalar multiplication**
$$0\mathbf{v} = \mathbf{0} \text{ for any } \mathbf{v} \in V$$

**Proof**:
$$0\mathbf{v} = (0+0)\mathbf{v} = 0\mathbf{v} + 0\mathbf{v}$$
Adding $-(0\mathbf{v})$ to both sides:
$$\mathbf{0} = 0\mathbf{v}$$

**Property 4: Scalar times zero vector**
$$c\mathbf{0} = \mathbf{0} \text{ for any scalar } c$$

**Property 5: Negative one**
$$(-1)\mathbf{v} = -\mathbf{v}$$

**Property 6: Zero product implies zero factor**
If $c\mathbf{v} = \mathbf{0}$, then either $c = 0$ or $\mathbf{v} = \mathbf{0}$

---

## 4. Subspaces

A **subspace** $W$ of vector space $V$ is a subset of $V$ that is itself a vector space under the same operations.

### Subspace Test Theorem

$W \subseteq V$ is a subspace if and only if:
1. $\mathbf{0} \in W$ (contains zero vector)
2. $\mathbf{u} + \mathbf{v} \in W$ for all $\mathbf{u}, \mathbf{v} \in W$ (closed under addition)
3. $c\mathbf{v} \in W$ for all $\mathbf{v} \in W$ and scalar $c$ (closed under scalar multiplication)

**Simplified Test**: $W$ is a subspace iff $c\mathbf{u} + d\mathbf{v} \in W$ for all $\mathbf{u}, \mathbf{v} \in W$ and scalars $c, d$ (closed under linear combinations)

### Examples of Subspaces

**Example 1: Trivial Subspaces**
- $\{\mathbf{0}\}$ is always a subspace (trivial subspace)
- $V$ itself is always a subspace

**Example 2: Lines Through Origin**
In $\mathbb{R}^2$, the set $W = \{(x, y) : y = mx\} = \{t(1, m) : t \in \mathbb{R}\}$ is a subspace (line through origin).

**Example 3: Planes Through Origin**
In $\mathbb{R}^3$, the set $W = \{(x, y, z) : ax + by + cz = 0\}$ is a subspace (plane through origin).

**Example 4: Solution Space (Null Space)**
For matrix $A$, the set $\text{Null}(A) = \{\mathbf{x} : A\mathbf{x} = \mathbf{0}\}$ is a subspace.

**Example 5: Column Space**
$\text{Col}(A) = \{\mathbf{y} : \mathbf{y} = A\mathbf{x} \text{ for some } \mathbf{x}\}$ is a subspace (span of column vectors).

### Verifying Subspace

**Problem**: Is $W = \{(x, y, z) : x + y + z = 0\}$ a subspace of $\mathbb{R}^3$?

**Solution**:

**Test 1: Zero vector**
$(0, 0, 0)$ satisfies $0 + 0 + 0 = 0$ ✓

**Test 2: Closure under addition**
Let $\mathbf{u} = (u_1, u_2, u_3)$ and $\mathbf{v} = (v_1, v_2, v_3)$ be in $W$.
Then $u_1 + u_2 + u_3 = 0$ and $v_1 + v_2 + v_3 = 0$.

$\mathbf{u} + \mathbf{v} = (u_1+v_1, u_2+v_2, u_3+v_3)$

Check: $(u_1+v_1) + (u_2+v_2) + (u_3+v_3) = (u_1+u_2+u_3) + (v_1+v_2+v_3) = 0 + 0 = 0$ ✓

**Test 3: Closure under scalar multiplication**
Let $\mathbf{v} = (v_1, v_2, v_3) \in W$, so $v_1 + v_2 + v_3 = 0$.

$c\mathbf{v} = (cv_1, cv_2, cv_3)$

Check: $cv_1 + cv_2 + cv_3 = c(v_1 + v_2 + v_3) = c(0) = 0$ ✓

**Conclusion**: $W$ is a subspace of $\mathbb{R}^3$ ✓

### Non-Examples

**Example 6: Line NOT Through Origin**
$W = \{(x, y) : y = 2x + 1\}$ is **NOT** a subspace because $(0, 0) \notin W$ (fails zero vector test).

**Example 7: First Quadrant**
$W = \{(x, y) : x \geq 0, y \geq 0\}$ is **NOT** a subspace because $(-1)(1, 1) = (-1, -1) \notin W$ (not closed under scalar multiplication).

---

## 5. Linear Dependence and Independence

### Linear Combinations

A **linear combination** of vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ is:
$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k$$
where $c_1, \ldots, c_k$ are scalars.

The **span** of vectors is the set of all linear combinations:
$$\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\} = \{c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k : c_i \in \mathbb{F}\}$$

**Key Fact**: $\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is always a subspace.

### Linear Dependence

Vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are **linearly dependent** if there exist scalars $c_1, \ldots, c_k$ (not all zero) such that:
$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$$

**Interpretation**: At least one vector can be written as a linear combination of others.

### Linear Independence

Vectors $\mathbf{v}_1, \ldots, \mathbf{v}_k$ are **linearly independent** if the only solution to:
$$c_1\mathbf{v}_1 + c_2\mathbf{v}_2 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$$
is $c_1 = c_2 = \cdots = c_k = 0$ (trivial solution).

**Interpretation**: No vector can be written as a linear combination of others.

### Geometric Intuition

**In $\mathbb{R}^2$**:
- 2 vectors are independent iff they don't lie on the same line through origin
- Dependent vectors are parallel (scalar multiples)

**In $\mathbb{R}^3$**:
- 2 vectors are independent iff non-parallel
- 3 vectors are independent iff they don't lie in the same plane
- Dependent vectors are coplanar

---

## 6. Testing for Linear Independence

### Method 1: Definition (Small Sets)

**Example 1**: Are $\mathbf{v}_1 = (1, 2)$ and $\mathbf{v}_2 = (2, 3)$ linearly independent?

**Solution**: Set up equation $c_1\mathbf{v}_1 + c_2\mathbf{v}_2 = \mathbf{0}$:
$$c_1(1, 2) + c_2(2, 3) = (0, 0)$$
$$(c_1 + 2c_2, 2c_1 + 3c_2) = (0, 0)$$

System:
$$\begin{cases}
c_1 + 2c_2 = 0 \\
2c_1 + 3c_2 = 0
\end{cases}$$

From first equation: $c_1 = -2c_2$

Substitute into second: $2(-2c_2) + 3c_2 = -4c_2 + 3c_2 = -c_2 = 0$

So $c_2 = 0$, which means $c_1 = 0$.

**Conclusion**: Only trivial solution exists. Vectors are **linearly independent** ✓

### Method 2: Matrix Rank (General)

Form matrix $A$ with vectors as columns. Vectors are linearly independent iff $\text{rank}(A) = $ number of vectors.

**Example 2**: Test independence of $\mathbf{v}_1 = (1, 2, 3)$, $\mathbf{v}_2 = (2, 4, 6)$, $\mathbf{v}_3 = (1, 0, 1)$

**Solution**: Form matrix:
$$A = \begin{pmatrix} 1 & 2 & 1 \\ 2 & 4 & 0 \\ 3 & 6 & 1 \end{pmatrix}$$

Row reduce:
$$\begin{pmatrix} 1 & 2 & 1 \\ 2 & 4 & 0 \\ 3 & 6 & 1 \end{pmatrix} \sim \begin{pmatrix} 1 & 2 & 1 \\ 0 & 0 & -2 \\ 0 & 0 & -2 \end{pmatrix} \sim \begin{pmatrix} 1 & 2 & 1 \\ 0 & 0 & 1 \\ 0 & 0 & 0 \end{pmatrix}$$

Rank = 2 < 3 (number of vectors)

**Conclusion**: Vectors are **linearly dependent** ✗

In fact, $\mathbf{v}_2 = 2\mathbf{v}_1$ (second vector is multiple of first).

### Method 3: Determinant (Square Case)

For $n$ vectors in $\mathbb{R}^n$, form $n \times n$ matrix $A$ with vectors as columns.

**Vectors are linearly independent** ⟺ $\det(A) \neq 0$

**Example 3**: Are $\mathbf{v}_1 = (1, 0, 1)$, $\mathbf{v}_2 = (0, 1, 0)$, $\mathbf{v}_3 = (1, 1, 2)$ independent?

**Solution**:
$$\det\begin{pmatrix} 1 & 0 & 1 \\ 0 & 1 & 1 \\ 1 & 0 & 2 \end{pmatrix} = 1 \cdot \det\begin{pmatrix} 1 & 1 \\ 0 & 2 \end{pmatrix} - 0 + 1 \cdot \det\begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$$
$$= 1(2) + 1(-1) = 2 - 1 = 1 \neq 0$$

**Conclusion**: Vectors are **linearly independent** ✓

### Important Theorems

**Theorem 1**: Any set containing the zero vector is linearly dependent.

**Theorem 2**: A set with exactly one nonzero vector is linearly independent.

**Theorem 3**: Two vectors are dependent iff one is a scalar multiple of the other.

**Theorem 4**: If a set is linearly independent, any subset is also independent.

**Theorem 5**: If a set is linearly dependent, any superset is also dependent.

**Theorem 6**: In $\mathbb{R}^n$, any set of more than $n$ vectors is linearly dependent.

---

## Data Science Applications

### 1. Feature Selection

In ML, features (columns of data matrix) should be linearly independent to avoid multicollinearity.

**Problem**: Redundant features don't add information and can hurt model performance.

**Solution**: Test feature vectors for linear independence. Remove dependent features.

```python
import numpy as np

# Feature matrix (rows = samples, columns = features)
X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9],
              [10, 11, 12]])

# Check if features are independent
rank = np.linalg.matrix_rank(X.T)  # Transpose to get features as rows
n_features = X.shape[1]

if rank < n_features:
    print(f"Features are DEPENDENT (rank {rank} < {n_features} features)")
else:
    print("Features are independent")
```

### 2. Dimensionality Reduction (PCA)

PCA finds linearly independent directions (principal components) that capture maximum variance.

**Key Idea**: Original features may be dependent. PCA constructs new independent features.

### 3. Rank of Data Matrix

The rank of data matrix $X$ (samples × features) tells us:
- **Effective dimensionality** of dataset
- Number of linearly independent features
- Whether linear regression solution is unique

**Example**: If $\text{rank}(X) < $ number of features, features are redundant.

### 4. Null Space and Model Solutions

For linear model $X\boldsymbol{\beta} = \mathbf{y}$:
- If columns of $X$ are independent ⟹ unique solution
- If columns are dependent ⟹ infinitely many solutions (underdetermined)

### 5. Orthogonal Vectors

In data science, we often want **orthogonal** vectors (perpendicular, dot product = 0).

Orthogonal vectors are always linearly independent (unless one is zero).

---

## Python Implementation

```python
import numpy as np
from scipy.linalg import null_space

# Example 1: Testing linear independence using rank
def are_independent(vectors):
    """
    Test if vectors (as columns) are linearly independent
    vectors: list of 1D numpy arrays or 2D array with vectors as columns
    """
    if isinstance(vectors, list):
        A = np.column_stack(vectors)
    else:
        A = vectors

    rank = np.linalg.matrix_rank(A)
    n_vectors = A.shape[1]

    return rank == n_vectors

# Test vectors
v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])  # Dependent on v1 (2*v1)
v3 = np.array([1, 0, 1])

print(f"v1, v2 independent? {are_independent([v1, v2])}")  # False
print(f"v1, v3 independent? {are_independent([v1, v3])}")  # True
print(f"v1, v2, v3 independent? {are_independent([v1, v2, v3])}")  # False

# Example 2: Finding linear dependence relation
A = np.column_stack([v1, v2, v3])
rank = np.linalg.matrix_rank(A)
print(f"\nMatrix rank: {rank} (expected 2 since v2 = 2*v1)")

# Find null space (dependence relations)
ns = null_space(A)
print(f"Null space:\n{ns}")
# Columns of ns give coefficients for dependence relations

# Example 3: Checking if set spans R^n
def spans_space(vectors, n):
    """Check if vectors span R^n"""
    A = np.column_stack(vectors) if isinstance(vectors, list) else vectors
    return np.linalg.matrix_rank(A) == n

v1 = np.array([1, 0])
v2 = np.array([0, 1])
v3 = np.array([1, 1])

print(f"\nv1, v2 span R^2? {spans_space([v1, v2], 2)}")  # True
print(f"v1 alone spans R^2? {spans_space([v1], 2)}")  # False

# Example 4: Testing subspace conditions
def is_subspace_of_Rn(vectors_in_subspace, test_vector):
    """
    Check if test_vector can be written as linear combination
    of vectors_in_subspace (i.e., if it's in their span)
    """
    A = np.column_stack(vectors_in_subspace)
    try:
        # Solve A @ coeffs = test_vector
        coeffs = np.linalg.lstsq(A, test_vector, rcond=None)[0]
        # Check if solution is exact
        reconstruction = A @ coeffs
        return np.allclose(reconstruction, test_vector)
    except:
        return False

# Subspace spanned by v1, v2
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
test1 = np.array([2, 3, 0])  # In span
test2 = np.array([1, 1, 1])  # Not in span

print(f"\ntest1 in span(v1, v2)? {is_subspace_of_Rn([v1, v2], test1)}")  # True
print(f"test2 in span(v1, v2)? {is_subspace_of_Rn([v1, v2], test2)}")  # False

# Example 5: Gram-Schmidt preview (orthogonalization)
def gram_schmidt(vectors):
    """Convert linearly independent vectors to orthonormal vectors"""
    basis = []
    for v in vectors:
        w = v.copy()
        # Subtract projections onto existing basis vectors
        for b in basis:
            w = w - np.dot(v, b) * b
        # Normalize
        if np.linalg.norm(w) > 1e-10:
            basis.append(w / np.linalg.norm(w))
    return np.array(basis).T

v1 = np.array([1.0, 1.0, 0.0])
v2 = np.array([1.0, 0.0, 1.0])

orthonormal = gram_schmidt([v1, v2])
print(f"\nOrthonormal basis:\n{orthonormal}")

# Verify orthonormality
print(f"Dot products (should be I):\n{orthonormal.T @ orthonormal}")
```

---

## Common Pitfalls and Misconceptions

### Pitfall 1: Confusing Linear Dependence with "Unrelated"
**Misconception**: "Independent means vectors have nothing in common."

**Reality**: Independence is a precise mathematical concept about linear combinations, not intuitive "unrelatedness."

### Pitfall 2: Thinking Zero Vector is Independent
**Misconception**: "A single vector is always independent."

**Reality**: The zero vector is **always dependent** (since $1 \cdot \mathbf{0} = \mathbf{0}$).

### Pitfall 3: Ignoring the Origin in Subspaces
**Misconception**: "Any line or plane is a subspace."

**Reality**: Subspaces MUST contain the origin. Line $y = x + 1$ is NOT a subspace of $\mathbb{R}^2$.

### Pitfall 4: More Vectors = Bigger Span
**Misconception**: "Adding more vectors always increases the span."

**Reality**: Adding dependent vectors doesn't change the span. $\text{span}\{\mathbf{v}_1\} = \text{span}\{\mathbf{v}_1, 2\mathbf{v}_1\}$

### Pitfall 5: Rank Confusion
**Misconception**: "Rank equals number of rows or columns."

**Reality**: Rank is the maximum number of linearly independent rows (or columns). Can be less than both dimensions.

### Pitfall 6: Assuming Geometric Intuition Works in High Dimensions
**Misconception**: "I can visualize independence in $\mathbb{R}^{100}$."

**Reality**: Use algebraic tests (rank, determinant) for high-dimensional spaces. Geometric intuition fails beyond $\mathbb{R}^3$.

---

## Practice Problems

### Basic Level

1. **Vector Space Verification**: Is $\mathbb{R}^+$ (positive reals) with standard addition and multiplication a vector space? Why or why not?

2. **Subspace Test**: Determine if $W = \{(x, y, z) : 2x - y + z = 0\}$ is a subspace of $\mathbb{R}^3$.

3. **Linear Combinations**: Express $(7, 11)$ as a linear combination of $(1, 2)$ and $(3, 4)$.

4. **Independence (2D)**: Are $(1, 3)$ and $(2, 5)$ linearly independent?

5. **Span**: Does $\text{span}\{(1, 0, 1), (0, 1, 0)\}$ contain $(2, 3, 2)$?

### Intermediate Level

6. **Subspace Proof**: Prove that the intersection of two subspaces is a subspace.

7. **Independence Test**: Test if $\{(1, 2, 3), (2, 3, 4), (3, 4, 5)\}$ is linearly independent.

8. **Span and Dependence**: If $\mathbf{v}_3 \in \text{span}\{\mathbf{v}_1, \mathbf{v}_2\}$, prove $\{\mathbf{v}_1, \mathbf{v}_2, \mathbf{v}_3\}$ is linearly dependent.

9. **Null Space**: Find all vectors in null space of $A = \begin{pmatrix} 1 & 2 & 3 \\ 2 & 4 & 6 \end{pmatrix}$.

10. **Polynomial Space**: Are $1$, $x$, $x^2$ linearly independent in $P_2$ (polynomials of degree ≤ 2)?

### Advanced Level

11. **Dependence Relation**: Given dependent vectors $\{(1, 2, 1), (2, 4, 3), (1, 2, 2)\}$, find the dependence relation.

12. **Maximal Independent Set**: From $\{(1, 0, 1), (2, 1, 3), (0, 1, 1), (1, 1, 2)\}$, find a maximal linearly independent subset.

13. **Dimension Counting**: Prove that if $V$ has dimension $n$, any set of $n+1$ vectors is linearly dependent.

14. **Matrix Rank**: If $A$ is $m \times n$ with rank $r$, prove that any $r+1$ columns are linearly dependent.

15. **Feature Matrix**: Given data matrix $X$ (100 samples × 5 features), what does $\text{rank}(X^TX) < 5$ tell you about the features?

---

## Summary and Key Takeaways

1. **Vector spaces are sets with addition and scalar multiplication** satisfying 10 axioms

2. **Subspaces must pass three tests**: contain zero, closed under addition, closed under scalar multiplication

3. **Linear independence means no redundancy**: no vector can be written as combination of others

4. **Testing independence**: use rank, determinant, or solve $c_1\mathbf{v}_1 + \cdots + c_k\mathbf{v}_k = \mathbf{0}$

5. **Span creates subspaces**: $\text{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ is smallest subspace containing these vectors

6. **In $\mathbb{R}^n$, at most $n$ vectors can be independent**: more than $n$ must be dependent

7. **Rank measures independence**: rank = maximum number of independent columns (or rows)

8. **Applications in ML**: feature selection, dimensionality, unique solutions to regression

---

## Connection to Next Week

Week 4 introduces **basis and dimension**:
- A basis is a maximal linearly independent set that spans the space
- Dimension = number of vectors in any basis
- Every basis of a vector space has the same size (dimension theorem)

Understanding independence (this week) is prerequisite for understanding bases (next week)!

---

## References

- **Textbook**: Linear Algebra (IIT Madras), Chapter 4
- **Video Lectures**: Week 3 playlist
- **Strang, G.**: *Introduction to Linear Algebra*, Chapter 3
- **Axler, S.**: *Linear Algebra Done Right*, Chapter 2
- **3Blue1Brown**: "Essence of Linear Algebra" series

---

**Last Updated**: 2025-11-22
**Next Topic**: Week 4 - Basis and Dimension
