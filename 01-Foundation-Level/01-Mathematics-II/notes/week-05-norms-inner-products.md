# Week 5: Vector Norms and Inner Products

---
Date: 2025-11-22
Course: BSMA1003 - Mathematics for Data Science II
Level: Foundation
Week: 5 of 11
Source: IIT Madras Mathematics II Week 5
Topic Area: Linear Algebra - Norms and Inner Products
Tags: #BSMA1003 #LinearAlgebra #Week5 #Norms #InnerProducts #Orthogonality #Foundation
---

## Topics Covered

1. **Vector Norms (L1, L2, Infinity)**
2. **Properties of Norms**
3. **Inner Products and Dot Products**
4. **Angles Between Vectors**
5. **Orthogonality**
6. **Orthogonal Projections**
7. **Cauchy-Schwarz Inequality**
8. **Applications to Machine Learning**

---

## Key Concepts

### 1. Vector Norms

A **norm** is a function that assigns a non-negative length or size to a vector.

**General Definition**: A norm $\\|\\cdot\\|$ on vector space $V$ satisfies:
1. **Positivity**: $\\|v\\| \\geq 0$ and $\\|v\\| = 0 \\iff v = 0$
2. **Homogeneity**: $\\|\\alpha v\\| = |\\alpha|\\|v\\|$ for scalar $\\alpha$
3. **Triangle Inequality**: $\\|u + v\\| \\leq \\|u\\| + \\|v\\|$

#### L1 Norm (Manhattan Distance)

$$\\|x\\|_1 = \\sum_{i=1}^n |x_i| = |x_1| + |x_2| + \\cdots + |x_n|$$

**Geometric Interpretation**: Distance when traveling along grid lines (like Manhattan streets)

**Applications**: 
- Sparse models (Lasso regression)
- Taxicab geometry
- Feature selection in ML

#### Example 1: Computing L1 Norm

For $v = \\begin{bmatrix} 3 \\\\ -4 \\\\ 2 \\end{bmatrix}$:

$$\\|v\\|_1 = |3| + |-4| + |2| = 3 + 4 + 2 = 9$$

```python
import numpy as np

v = np.array([3, -4, 2])

# L1 norm
l1_norm = np.linalg.norm(v, ord=1)
# Or manually:
l1_manual = np.sum(np.abs(v))

print(f"L1 norm: {l1_norm}")  # Output: 9.0
```

#### L2 Norm (Euclidean Distance)

$$\\|x\\|_2 = \\sqrt{\\sum_{i=1}^n x_i^2} = \\sqrt{x_1^2 + x_2^2 + \\cdots + x_n^2}$$

**Geometric Interpretation**: Straight-line distance (as the crow flies)

**Applications**:
- Standard distance metric
- Ridge regression
- Most common in ML

#### Example 2: Computing L2 Norm

For $v = \\begin{bmatrix} 3 \\\\ -4 \\\\ 2 \\end{bmatrix}$:

$$\\|v\\|_2 = \\sqrt{3^2 + (-4)^2 + 2^2} = \\sqrt{9 + 16 + 4} = \\sqrt{29} \\approx 5.385$$

```python
# L2 norm (Euclidean)
l2_norm = np.linalg.norm(v, ord=2)
# Or default (L2 is default):
l2_default = np.linalg.norm(v)

print(f"L2 norm: {l2_norm:.3f}")  # Output: 5.385
```

#### L∞ Norm (Maximum Norm)

$$\\|x\\|_\\infty = \\max_{i=1,\\ldots,n} |x_i|$$

**Geometric Interpretation**: Maximum absolute component

**Applications**:
- Worst-case analysis
- Minimax problems
- Chebyshev distance

#### Example 3: Computing L∞ Norm

For $v = \\begin{bmatrix} 3 \\\\ -4 \\\\ 2 \\end{bmatrix}$:

$$\\|v\\|_\\infty = \\max\\{|3|, |-4|, |2|\\} = 4$$

```python
# L-infinity norm
linf_norm = np.linalg.norm(v, ord=np.inf)

print(f"L∞ norm: {linf_norm}")  # Output: 4.0
```

### 2. Inner Products

An **inner product** (or dot product in $\\mathbb{R}^n$) measures the similarity between vectors.

**Definition** (standard inner product):
$$\\langle u, v \\rangle = u \\cdot v = \\sum_{i=1}^n u_i v_i = u_1v_1 + u_2v_2 + \\cdots + u_nv_n$$

**Properties**:
1. **Symmetry**: $\\langle u, v \\rangle = \\langle v, u \\rangle$
2. **Linearity**: $\\langle au + bw, v \\rangle = a\\langle u, v \\rangle + b\\langle w, v \\rangle$
3. **Positive Definite**: $\\langle v, v \\rangle \\geq 0$ and $\\langle v, v \\rangle = 0 \\iff v = 0$

#### Example 4: Computing Inner Product

For $u = \\begin{bmatrix} 1 \\\\ 2 \\\\ 3 \\end{bmatrix}$ and $v = \\begin{bmatrix} 4 \\\\ -1 \\\\ 2 \\end{bmatrix}$:

$$\\langle u, v \\rangle = (1)(4) + (2)(-1) + (3)(2) = 4 - 2 + 6 = 8$$

```python
u = np.array([1, 2, 3])
v = np.array([4, -1, 2])

# Inner product (dot product)
inner_product = np.dot(u, v)
# Or:
inner_product = u @ v

print(f"Inner product: {inner_product}")  # Output: 8
```

### 3. Angles Between Vectors

The **angle** $\\theta$ between vectors $u$ and $v$ is given by:

$$\\cos(\\theta) = \\frac{\\langle u, v \\rangle}{\\|u\\|_2 \\|v\\|_2}$$

Therefore:
$$\\theta = \\arccos\\left(\\frac{\\langle u, v \\rangle}{\\|u\\|_2 \\|v\\|_2}\\right)$$

**Range**: $0 \\leq \\theta \\leq \\pi$ (0° to 180°)

#### Example 5: Computing Angle Between Vectors

For $u = \\begin{bmatrix} 1 \\\\ 0 \\end{bmatrix}$ and $v = \\begin{bmatrix} 1 \\\\ 1 \\end{bmatrix}$:

**Step 1**: Compute inner product
$$\\langle u, v \\rangle = (1)(1) + (0)(1) = 1$$

**Step 2**: Compute norms
$$\\|u\\|_2 = 1, \\quad \\|v\\|_2 = \\sqrt{2}$$

**Step 3**: Compute angle
$$\\cos(\\theta) = \\frac{1}{1 \\cdot \\sqrt{2}} = \\frac{1}{\\sqrt{2}}$$
$$\\theta = \\arccos\\left(\\frac{1}{\\sqrt{2}}\\right) = 45° = \\frac{\\pi}{4}$$

```python
u = np.array([1, 0])
v = np.array([1, 1])

# Compute angle
cos_theta = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
theta_rad = np.arccos(cos_theta)
theta_deg = np.degrees(theta_rad)

print(f"Angle: {theta_deg:.1f}°")  # Output: 45.0°
```

### 4. Orthogonality

Vectors $u$ and $v$ are **orthogonal** (perpendicular) if their inner product is zero.

**Definition**:
$$u \\perp v \\iff \\langle u, v \\rangle = 0$$

**Geometric Interpretation**: Vectors at right angles ($\\theta = 90°$)

#### Example 6: Checking Orthogonality

Are $u = \\begin{bmatrix} 3 \\\\ -2 \\end{bmatrix}$ and $v = \\begin{bmatrix} 2 \\\\ 3 \\end{bmatrix}$ orthogonal?

$$\\langle u, v \\rangle = (3)(2) + (-2)(3) = 6 - 6 = 0$$

**Yes! They are orthogonal** ✓

```python
u = np.array([3, -2])
v = np.array([2, 3])

is_orthogonal = np.abs(np.dot(u, v)) < 1e-10
print(f"Orthogonal: {is_orthogonal}")  # True
```

### 5. Orthogonal Projections

The **projection** of vector $v$ onto vector $u$ is:

$$\\text{proj}_u(v) = \\frac{\\langle v, u \\rangle}{\\langle u, u \\rangle} u = \\frac{\\langle v, u \\rangle}{\\|u\\|^2} u$$

**Geometric Interpretation**: Shadow of $v$ cast onto the line containing $u$

**Component in direction of $u$**:
$$\\text{comp}_u(v) = \\frac{\\langle v, u \\rangle}{\\|u\\|}$$

#### Example 7: Computing Projection

Project $v = \\begin{bmatrix} 5 \\\\ 3 \\end{bmatrix}$ onto $u = \\begin{bmatrix} 2 \\\\ 0 \\end{bmatrix}$:

**Step 1**: Compute inner product
$$\\langle v, u \\rangle = (5)(2) + (3)(0) = 10$$

**Step 2**: Compute $\\|u\\|^2$
$$\\|u\\|^2 = 2^2 + 0^2 = 4$$

**Step 3**: Compute projection
$$\\text{proj}_u(v) = \\frac{10}{4}\\begin{bmatrix} 2 \\\\ 0 \\end{bmatrix} = \\begin{bmatrix} 5 \\\\ 0 \\end{bmatrix}$$

```python
v = np.array([5, 3])
u = np.array([2, 0])

# Projection of v onto u
proj = (np.dot(v, u) / np.dot(u, u)) * u
print(f"Projection: {proj}")  # Output: [5. 0.]
```

---

## Important Formulas

### Norms
$$\\|x\\|_1 = \\sum_{i=1}^n |x_i|$$
$$\\|x\\|_2 = \\sqrt{\\sum_{i=1}^n x_i^2}$$
$$\\|x\\|_\\infty = \\max_{i=1,\\ldots,n} |x_i|$$

### Inner Product
$$\\langle u, v \\rangle = \\sum_{i=1}^n u_i v_i$$

### Angle Between Vectors
$$\\cos(\\theta) = \\frac{\\langle u, v \\rangle}{\\|u\\|_2 \\|v\\|_2}$$

### Projection
$$\\text{proj}_u(v) = \\frac{\\langle v, u \\rangle}{\\|u\\|^2} u$$

### Cauchy-Schwarz Inequality
$$|\\langle u, v \\rangle| \\leq \\|u\\|_2 \\|v\\|_2$$

---

## Theorems & Proofs

### Theorem 1: Cauchy-Schwarz Inequality

**Statement**: For any vectors $u, v \\in \\mathbb{R}^n$:
$$|\\langle u, v \\rangle| \\leq \\|u\\|_2 \\|v\\|_2$$

Equality holds if and only if $u$ and $v$ are linearly dependent.

**Proof Sketch**:
1. If $v = 0$, inequality is trivially true
2. For $v \\neq 0$, consider $w = u - \\text{proj}_v(u)$
3. Then $w \\perp v$ (orthogonal decomposition)
4. By Pythagorean theorem: $\\|u\\|^2 = \\|\\text{proj}_v(u)\\|^2 + \\|w\\|^2 \\geq \\|\\text{proj}_v(u)\\|^2$
5. Simplifying gives the Cauchy-Schwarz inequality

### Theorem 2: Triangle Inequality

**Statement**: For any norm and vectors $u, v$:
$$\\|u + v\\| \\leq \\|u\\| + \\|v\\|$$

**Application**: The direct path is always shortest or equal to detour paths.

### Theorem 3: Pythagorean Theorem

**Statement**: If $u \\perp v$ (orthogonal), then:
$$\\|u + v\\|^2 = \\|u\\|^2 + \\|v\\|^2$$

**Proof**:
$$\\|u + v\\|^2 = \\langle u + v, u + v \\rangle = \\langle u, u \\rangle + 2\\langle u, v \\rangle + \\langle v, v \\rangle$$

Since $u \\perp v$, we have $\\langle u, v \\rangle = 0$:
$$\\|u + v\\|^2 = \\|u\\|^2 + \\|v\\|^2$$

---

## Data Science Applications

### 1. Distance Metrics in Machine Learning

```python
# Comparing different distance metrics
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances

# Sample data points
X = np.array([[1, 2], [4, 5], [7, 8]])

# Euclidean distance (L2)
dist_l2 = euclidean_distances(X)
print("Euclidean distances:\\n", dist_l2)

# Manhattan distance (L1)
dist_l1 = manhattan_distances(X)
print("\\nManhattan distances:\\n", dist_l1)

# Impact on K-Nearest Neighbors
from sklearn.neighbors import KNeighborsClassifier

# Different norms change which neighbors are "nearest"
knn_l1 = KNeighborsClassifier(n_neighbors=3, metric='manhattan')
knn_l2 = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
```

### 2. Cosine Similarity for Text Analysis

```python
# Document similarity using inner products
from sklearn.metrics.pairwise import cosine_similarity

# TF-IDF vectors for documents
doc1 = np.array([1, 2, 0, 1, 0])  # Word frequencies
doc2 = np.array([0, 1, 1, 1, 2])
doc3 = np.array([1, 2, 0, 2, 0])

documents = np.array([doc1, doc2, doc3])

# Cosine similarity matrix
similarity = cosine_similarity(documents)
print("Document similarities:\\n", similarity)

# Note: cosine_similarity = <u,v> / (||u|| ||v||)
# Documents with similar content have high cosine similarity
```

### 3. Feature Normalization

```python
# Normalizing features using different norms
from sklearn.preprocessing import Normalizer

X = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

# L2 normalization (most common)
normalizer_l2 = Normalizer(norm='l2')
X_l2_norm = normalizer_l2.transform(X)

# L1 normalization
normalizer_l1 = Normalizer(norm='l1')
X_l1_norm = normalizer_l1.transform(X)

print("Original:\\n", X)
print("\\nL2 normalized:\\n", X_l2_norm)
print("\\nL1 normalized:\\n", X_l1_norm)
```

### 4. Regularization in Linear Models

```python
# L1 (Lasso) vs L2 (Ridge) regularization
from sklearn.linear_model import Lasso, Ridge
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=20, noise=10, random_state=42)

# Ridge regression (L2 penalty)
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)
print("Ridge non-zero coefficients:", np.sum(ridge.coef_ != 0))

# Lasso regression (L1 penalty) - promotes sparsity
lasso = Lasso(alpha=1.0)
lasso.fit(X, y)
print("Lasso non-zero coefficients:", np.sum(lasso.coef_ != 0))
# L1 typically produces sparser solutions!
```

### 5. Orthogonal Projections in PCA

```python
# PCA uses orthogonal projections onto principal components
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate 2D data
np.random.seed(42)
X = np.dot(np.random.randn(100, 2), [[2, 0.5], [0.5, 1]])

# Fit PCA
pca = PCA(n_components=1)
X_pca = pca.fit_transform(X)

# First principal component (direction)
pc1 = pca.components_[0]

# Project points onto first PC
projections = np.outer(X_pca.flatten(), pc1)

print(f"First PC direction: {pc1}")
print(f"Points are projected onto this direction")
```

---

## Common Pitfalls

### Pitfall 1: Confusing Norms

**Wrong**: Using L1 norm when Euclidean distance is needed
**Correct**: L2 norm for straight-line distance, L1 for grid-based distance

### Pitfall 2: Orthogonality vs Independence

**Misconception**: "Orthogonal implies linearly independent"
**Truth**: True, BUT only for non-zero vectors
- Zero vector is orthogonal to everything but not independent

### Pitfall 3: Projection Direction

**Wrong**: $\\text{proj}_v(u)$ vs $\\text{proj}_u(v)$ are the same
**Correct**: These are different! Order matters.

### Pitfall 4: Angle Computation

**Common Error**: Forgetting to check for division by zero when vectors have zero norm

```python
def safe_angle(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)
    if norm_u == 0 or norm_v == 0:
        return None  # Undefined
    cos_theta = np.dot(u, v) / (norm_u * norm_v)
    # Clamp to [-1, 1] due to numerical errors
    cos_theta = np.clip(cos_theta, -1, 1)
    return np.arccos(cos_theta)
```

### Pitfall 5: Norm Choice in ML

**Wrong**: Always using L2 norm
**Better**: Choose based on problem:
- L1: Sparse solutions, outlier robust
- L2: Smooth solutions, differentiable
- L∞: Worst-case optimization

---

## Practice Problems

### Basic Level

1. Compute $\\|v\\|_1$, $\\|v\\|_2$, and $\\|v\\|_\\infty$ for $v = \\begin{bmatrix} -2 \\\\ 3 \\\\ -1 \\end{bmatrix}$

2. Find the inner product of $u = \\begin{bmatrix} 1 \\\\ 2 \\\\ 3 \\end{bmatrix}$ and $v = \\begin{bmatrix} 4 \\\\ 5 \\\\ 6 \\end{bmatrix}$

3. Determine if vectors $u = \\begin{bmatrix} 1 \\\\ -1 \\end{bmatrix}$ and $v = \\begin{bmatrix} 1 \\\\ 1 \\end{bmatrix}$ are orthogonal

4. Compute the angle between vectors $u = \\begin{bmatrix} 1 \\\\ 0 \\end{bmatrix}$ and $v = \\begin{bmatrix} 0 \\\\ 1 \\end{bmatrix}$

5. Project $v = \\begin{bmatrix} 3 \\\\ 4 \\end{bmatrix}$ onto $u = \\begin{bmatrix} 1 \\\\ 0 \\end{bmatrix}$

### Intermediate Level

6. Prove that $\\|v\\|_2 \\leq \\|v\\|_1$ for any vector $v \\in \\mathbb{R}^n$

7. Find all vectors in $\\mathbb{R}^3$ orthogonal to both $u = \\begin{bmatrix} 1 \\\\ 1 \\\\ 0 \\end{bmatrix}$ and $v = \\begin{bmatrix} 1 \\\\ 0 \\\\ 1 \\end{bmatrix}$

8. Decompose $v = \\begin{bmatrix} 5 \\\\ 5 \\end{bmatrix}$ into components parallel and perpendicular to $u = \\begin{bmatrix} 3 \\\\ 4 \\end{bmatrix}$

9. Verify the Cauchy-Schwarz inequality for $u = \\begin{bmatrix} 1 \\\\ 2 \\\\ 3 \\end{bmatrix}$ and $v = \\begin{bmatrix} 4 \\\\ 5 \\\\ 6 \\end{bmatrix}$

10. Find the distance from point $P = (2, 3)$ to the line through origin with direction $v = \\begin{bmatrix} 1 \\\\ 1 \\end{bmatrix}$

### Advanced Level

11. Prove the triangle inequality for the L2 norm using Cauchy-Schwarz

12. Show that if $u \\perp v$, then $\\|u + v\\|^2 = \\|u\\|^2 + \\|v\\|^2$ (Pythagorean theorem)

13. Given orthogonal vectors $u_1, u_2, \\ldots, u_k$, prove:
    $$\\left\\|\\sum_{i=1}^k u_i\\right\\|^2 = \\sum_{i=1}^k \\|u_i\\|^2$$

14. Find the matrix $P$ that projects vectors onto the subspace spanned by $u = \\begin{bmatrix} 1 \\\\ 1 \\end{bmatrix}$

15. Prove that for unit vectors $u$ and $v$: $\\|u - v\\|^2 = 2 - 2\\langle u, v \\rangle$

---

## Self-Assessment Checklist

- [ ] Can you compute L1, L2, and L∞ norms?
- [ ] Do you understand geometric interpretations of different norms?
- [ ] Can you calculate inner products and angles between vectors?
- [ ] Can you determine if vectors are orthogonal?
- [ ] Can you compute orthogonal projections?
- [ ] Do you understand the Cauchy-Schwarz inequality?
- [ ] Can you apply norms to ML problems (regularization, distance metrics)?
- [ ] Can you choose appropriate norms for different applications?

---

## Key Takeaways

1. **Norms measure vector magnitude**: Different norms capture different notions of size
2. **Inner products measure similarity**: Orthogonal vectors have zero inner product
3. **Projections decompose vectors**: Parallel and perpendicular components
4. **L1 promotes sparsity**: Used in Lasso regression for feature selection
5. **L2 is most common**: Euclidean distance, Ridge regression
6. **Orthogonality simplifies computation**: Orthogonal bases are easier to work with

---

## References

- **Textbook**: Linear Algebra (IIT Madras), Chapter 6
- **Videos**: Week 5 lectures - Norms and Inner Products
- **Additional**: Strang, G. - *Introduction to Linear Algebra*, Chapter 3

---

## Connection to Next Week

Week 6 extends these concepts with:
- **Gram-Schmidt Process**: Creating orthogonal bases
- **QR Decomposition**: Factoring matrices into orthogonal and triangular parts
- **Orthonormal Bases**: Combining orthogonality with unit length

Understanding norms and inner products is essential for Week 6!

---

**Last Updated**: 2025-11-22
**Next Week**: Gram-Schmidt Orthogonalization and QR Decomposition
