# Week 1: Vectors, Matrices, and Systems of Linear Equations

**Date**: 2025-11-16  
**Course**: Mathematics for Data Science II (BSMA1003)

## Topics Covered

1. Vectors and Vector Operations
2. Matrices and Matrix Operations
3. Systems of Linear Equations
4. Determinants (Part 1: 2×2 and 3×3)
5. Determinants (Part 2: Properties and Applications)

---

## Key Concepts

### 1. Vectors

A **vector** is an ordered list of numbers that represents magnitude and direction in n-dimensional space.

#### Vector Notation
- **Column vector**: $\vec{v} = \begin{bmatrix} v_1 \\ v_2 \\ \vdots \\ v_n \end{bmatrix}$
- **Row vector**: $\vec{v}^T = \begin{bmatrix} v_1 & v_2 & \cdots & v_n \end{bmatrix}$
- **Component form**: $\vec{v} = \langle v_1, v_2, \ldots, v_n \rangle$

#### Vector Operations

**1. Vector Addition**
$$\vec{u} + \vec{v} = \begin{bmatrix} u_1 \\ u_2 \end{bmatrix} + \begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} u_1 + v_1 \\ u_2 + v_2 \end{bmatrix}$$

**Properties**:
- Commutative: $\vec{u} + \vec{v} = \vec{v} + \vec{u}$
- Associative: $(\vec{u} + \vec{v}) + \vec{w} = \vec{u} + (\vec{v} + \vec{w})$
- Identity: $\vec{v} + \vec{0} = \vec{v}$

**2. Scalar Multiplication**
$$c\vec{v} = c\begin{bmatrix} v_1 \\ v_2 \end{bmatrix} = \begin{bmatrix} cv_1 \\ cv_2 \end{bmatrix}$$

**Properties**:
- Distributive: $c(\vec{u} + \vec{v}) = c\vec{u} + c\vec{v}$
- Distributive: $(c + d)\vec{v} = c\vec{v} + d\vec{v}$
- Associative: $(cd)\vec{v} = c(d\vec{v})$

**3. Dot Product (Inner Product)**
$$\vec{u} \cdot \vec{v} = u_1v_1 + u_2v_2 + \cdots + u_nv_n = \sum_{i=1}^{n} u_iv_i$$

**Geometric interpretation**: $\vec{u} \cdot \vec{v} = |\vec{u}||\vec{v}|\cos\theta$

**Properties**:
- Commutative: $\vec{u} \cdot \vec{v} = \vec{v} \cdot \vec{u}$
- Distributive: $\vec{u} \cdot (\vec{v} + \vec{w}) = \vec{u} \cdot \vec{v} + \vec{u} \cdot \vec{w}$
- $\vec{u} \cdot \vec{u} = |\vec{u}|^2$

**4. Vector Magnitude (Norm)**
$$|\vec{v}| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2} = \sqrt{\vec{v} \cdot \vec{v}}$$

**5. Unit Vector**
$$\hat{v} = \frac{\vec{v}}{|\vec{v}|}$$

**Why important for DS**: 
- Feature vectors in machine learning
- Word embeddings in NLP
- Similarity measures (cosine similarity)
- Gradient descent directions

### 2. Matrices

A **matrix** is a rectangular array of numbers arranged in rows and columns.

#### Matrix Notation
$$A = \begin{bmatrix} a_{11} & a_{12} & \cdots & a_{1n} \\ a_{21} & a_{22} & \cdots & a_{2n} \\ \vdots & \vdots & \ddots & \vdots \\ a_{m1} & a_{m2} & \cdots & a_{mn} \end{bmatrix}$$

- **Dimensions**: $m \times n$ (m rows, n columns)
- **Element**: $a_{ij}$ is element in row i, column j
- **Square matrix**: m = n

#### Special Matrices

**1. Zero Matrix (O)**
$$O = \begin{bmatrix} 0 & 0 & \cdots & 0 \\ 0 & 0 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 0 \end{bmatrix}$$

**2. Identity Matrix (I)**
$$I_n = \begin{bmatrix} 1 & 0 & \cdots & 0 \\ 0 & 1 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & 1 \end{bmatrix}$$

**3. Diagonal Matrix**
$$D = \begin{bmatrix} d_1 & 0 & \cdots & 0 \\ 0 & d_2 & \cdots & 0 \\ \vdots & \vdots & \ddots & \vdots \\ 0 & 0 & \cdots & d_n \end{bmatrix}$$

**4. Upper Triangular Matrix**
All elements below main diagonal are zero.

**5. Lower Triangular Matrix**
All elements above main diagonal are zero.

**6. Symmetric Matrix**
$A = A^T$ (equal to its transpose)

**7. Transpose**
$$A^T: (A^T)_{ij} = A_{ji}$$
Rows become columns and vice versa.

#### Matrix Operations

**1. Matrix Addition**
- Only defined for matrices of same dimensions
- Add corresponding elements

$$A + B = \begin{bmatrix} a_{11} + b_{11} & a_{12} + b_{12} \\ a_{21} + b_{21} & a_{22} + b_{22} \end{bmatrix}$$

**2. Scalar Multiplication**
$$cA = \begin{bmatrix} ca_{11} & ca_{12} \\ ca_{21} & ca_{22} \end{bmatrix}$$

**3. Matrix Multiplication**
- $(AB)_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$
- **Requirement**: Number of columns in A = number of rows in B
- Result dimensions: If $A$ is $m \times n$ and $B$ is $n \times p$, then $AB$ is $m \times p$

**Properties**:
- **NOT commutative**: $AB \neq BA$ (in general)
- Associative: $(AB)C = A(BC)$
- Distributive: $A(B + C) = AB + AC$
- Identity: $AI = IA = A$

**Why important for DS**:
- Neural network computations
- Data transformations
- Principal Component Analysis (PCA)
- Recommendation systems

### 3. Systems of Linear Equations

A system of m linear equations with n unknowns:

$$\begin{cases}
a_{11}x_1 + a_{12}x_2 + \cdots + a_{1n}x_n = b_1 \\
a_{21}x_1 + a_{22}x_2 + \cdots + a_{2n}x_n = b_2 \\
\vdots \\
a_{m1}x_1 + a_{m2}x_2 + \cdots + a_{mn}x_n = b_m
\end{cases}$$

#### Matrix Form
$$A\vec{x} = \vec{b}$$

Where:
- $A$ is the coefficient matrix ($m \times n$)
- $\vec{x}$ is the variable vector ($n \times 1$)
- $\vec{b}$ is the constant vector ($m \times 1$)

#### Types of Solutions

**1. Unique Solution**
- Exactly one solution
- System is consistent and determined
- For square systems: $\det(A) \neq 0$

**2. No Solution**
- System is inconsistent
- Equations are contradictory

**3. Infinitely Many Solutions**
- System is consistent and underdetermined
- Equations are dependent
- For square systems: $\det(A) = 0$

#### Augmented Matrix
$$[A|\vec{b}] = \left[\begin{array}{ccc|c} a_{11} & a_{12} & a_{13} & b_1 \\ a_{21} & a_{22} & a_{23} & b_2 \\ a_{31} & a_{32} & a_{33} & b_3 \end{array}\right]$$

**Why important for DS**:
- Linear regression (solving for coefficients)
- Optimization problems
- Computer graphics transformations
- Network flow problems

### 4. Determinants

The **determinant** is a scalar value that can be computed from a square matrix.

#### 2×2 Determinant
$$\det(A) = |A| = \begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc$$

#### 3×3 Determinant (Expansion by First Row)
$$\det(A) = \begin{vmatrix} a_{11} & a_{12} & a_{13} \\ a_{21} & a_{22} & a_{23} \\ a_{31} & a_{32} & a_{33} \end{vmatrix}$$

$$= a_{11}\begin{vmatrix} a_{22} & a_{23} \\ a_{32} & a_{33} \end{vmatrix} - a_{12}\begin{vmatrix} a_{21} & a_{23} \\ a_{31} & a_{33} \end{vmatrix} + a_{13}\begin{vmatrix} a_{21} & a_{22} \\ a_{31} & a_{32} \end{vmatrix}$$

$$= a_{11}(a_{22}a_{33} - a_{23}a_{32}) - a_{12}(a_{21}a_{33} - a_{23}a_{31}) + a_{13}(a_{21}a_{32} - a_{22}a_{31})$$

#### Cofactor Expansion
- **Minor** $M_{ij}$: Determinant of matrix obtained by deleting row i and column j
- **Cofactor** $C_{ij} = (-1)^{i+j}M_{ij}$
- Can expand along any row or column

#### Geometric Interpretation
- 2D: Area of parallelogram formed by two vectors
- 3D: Volume of parallelepiped formed by three vectors
- Sign indicates orientation (positive = right-handed)

### 5. Properties of Determinants

**Property 1: Transpose**
$$\det(A^T) = \det(A)$$

**Property 2: Row/Column Swap**
Swapping two rows (or columns) changes sign:
$$\det(A) = -\det(B)$$ where B is A with two rows swapped

**Property 3: Scalar Multiplication**
$$\det(cA) = c^n\det(A)$$ for $n \times n$ matrix

**Property 4: Row Addition**
Adding a multiple of one row to another doesn't change determinant:
If $B$ is obtained from $A$ by adding $k$(row i) to row j, then $\det(B) = \det(A)$

**Property 5: Zero Row/Column**
If any row or column is all zeros, $\det(A) = 0$

**Property 6: Identical Rows/Columns**
If two rows (or columns) are identical, $\det(A) = 0$

**Property 7: Product Rule**
$$\det(AB) = \det(A) \cdot \det(B)$$

**Property 8: Triangular Matrices**
For upper or lower triangular matrices:
$$\det(A) = \prod_{i=1}^{n} a_{ii}$$ (product of diagonal elements)

**Property 9: Invertibility**
- $A$ is invertible $\Longleftrightarrow$ $\det(A) \neq 0$
- If invertible: $\det(A^{-1}) = \frac{1}{\det(A)}$

---

## Definitions

- **Vector**: An ordered list of n numbers representing magnitude and direction in n-dimensional space
- **Magnitude/Norm**: Length of a vector, calculated as $|\vec{v}| = \sqrt{\sum v_i^2}$
- **Unit Vector**: A vector with magnitude 1, obtained by $\hat{v} = \vec{v}/|\vec{v}|$
- **Dot Product**: Scalar result of multiplying corresponding components: $\vec{u} \cdot \vec{v} = \sum u_iv_i$
- **Matrix**: A rectangular array of numbers with m rows and n columns ($m \times n$ matrix)
- **Transpose**: Matrix $A^T$ obtained by swapping rows and columns of A
- **Identity Matrix**: Square matrix with 1s on diagonal and 0s elsewhere, acts as multiplicative identity
- **Determinant**: Scalar value computed from square matrix, indicates invertibility and geometric properties
- **Singular Matrix**: Square matrix with determinant 0 (not invertible)
- **Non-singular Matrix**: Square matrix with non-zero determinant (invertible)
- **Augmented Matrix**: Matrix $[A|\vec{b}]$ combining coefficient matrix with constant vector
- **Linear System**: Set of linear equations that can be written as $A\vec{x} = \vec{b}$
- **Consistent System**: System with at least one solution
- **Inconsistent System**: System with no solution

---

## Important Formulas

### Vector Operations
- **Magnitude**: $|\vec{v}| = \sqrt{v_1^2 + v_2^2 + \cdots + v_n^2}$
- **Dot Product**: $\vec{u} \cdot \vec{v} = u_1v_1 + u_2v_2 + \cdots + u_nv_n$
- **Angle Between Vectors**: $\cos\theta = \frac{\vec{u} \cdot \vec{v}}{|\vec{u}||\vec{v}|}$
- **Projection**: $\text{proj}_{\vec{u}}\vec{v} = \frac{\vec{v} \cdot \vec{u}}{|\vec{u}|^2}\vec{u}$

### Matrix Operations
- **Matrix Multiplication**: $(AB)_{ij} = \sum_{k=1}^{n} a_{ik}b_{kj}$
- **Transpose Properties**: $(A^T)^T = A$, $(AB)^T = B^TA^T$

### Determinants
- **2×2 Matrix**: $\begin{vmatrix} a & b \\ c & d \end{vmatrix} = ad - bc$
- **3×3 Matrix**: Expand by cofactors along any row/column
- **Product**: $\det(AB) = \det(A)\det(B)$
- **Inverse**: $\det(A^{-1}) = \frac{1}{\det(A)}$
- **Transpose**: $\det(A^T) = \det(A)$
- **Scalar Multiplication**: $\det(cA) = c^n\det(A)$ for $n \times n$ matrix

---

## Theorems & Proofs

### Theorem 1: Determinant and Invertibility
**Statement**: A square matrix $A$ is invertible if and only if $\det(A) \neq 0$.

**Proof Outline**: 
- If $\det(A) = 0$, the columns are linearly dependent
- This means $A\vec{x} = \vec{0}$ has non-trivial solutions
- Therefore, $A$ cannot have an inverse (non-injective)
- Conversely, if $\det(A) \neq 0$, $A$ has full rank and is invertible

**Significance**: This theorem is fundamental in determining whether a system of linear equations has a unique solution.

### Theorem 2: Determinant Product Rule
**Statement**: For square matrices $A$ and $B$ of the same size, $\det(AB) = \det(A)\det(B)$.

**Significance**: This property is useful in computing determinants of matrix products without explicitly multiplying the matrices first.

### Theorem 3: Transpose Determinant
**Statement**: $\det(A^T) = \det(A)$ for any square matrix $A$.

**Proof**: The cofactor expansion along a row of $A$ equals the cofactor expansion along the corresponding column of $A^T$.

---

## Examples (Worked Problems)

### Example 1: Vector Operations
**Problem**: Given $\vec{u} = \begin{bmatrix} 3 \\ -2 \\ 1 \end{bmatrix}$ and $\vec{v} = \begin{bmatrix} 1 \\ 4 \\ -2 \end{bmatrix}$, find:
(a) $\vec{u} + \vec{v}$
(b) $3\vec{u} - 2\vec{v}$
(c) $\vec{u} \cdot \vec{v}$
(d) $|\vec{u}|$

**Solution**:
(a) $\vec{u} + \vec{v} = \begin{bmatrix} 3+1 \\ -2+4 \\ 1+(-2) \end{bmatrix} = \begin{bmatrix} 4 \\ 2 \\ -1 \end{bmatrix}$

(b) $3\vec{u} - 2\vec{v} = 3\begin{bmatrix} 3 \\ -2 \\ 1 \end{bmatrix} - 2\begin{bmatrix} 1 \\ 4 \\ -2 \end{bmatrix} = \begin{bmatrix} 9 \\ -6 \\ 3 \end{bmatrix} - \begin{bmatrix} 2 \\ 8 \\ -4 \end{bmatrix} = \begin{bmatrix} 7 \\ -14 \\ 7 \end{bmatrix}$

(c) $\vec{u} \cdot \vec{v} = (3)(1) + (-2)(4) + (1)(-2) = 3 - 8 - 2 = -7$

(d) $|\vec{u}| = \sqrt{3^2 + (-2)^2 + 1^2} = \sqrt{9 + 4 + 1} = \sqrt{14} \approx 3.74$

### Example 2: Matrix Multiplication
**Problem**: Compute $AB$ where $A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ and $B = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$

**Solution**:
$$AB = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}\begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$$

Element $(1,1)$: $(1)(5) + (2)(7) = 5 + 14 = 19$
Element $(1,2)$: $(1)(6) + (2)(8) = 6 + 16 = 22$
Element $(2,1)$: $(3)(5) + (4)(7) = 15 + 28 = 43$
Element $(2,2)$: $(3)(6) + (4)(8) = 18 + 32 = 50$

$$AB = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}$$

### Example 3: System of Linear Equations
**Problem**: Solve the system:
$$\begin{cases}
2x + y = 7 \\
x - y = 2
\end{cases}$$

**Solution Method 1: Substitution**
From equation 2: $x = y + 2$
Substitute into equation 1: $2(y + 2) + y = 7$
$2y + 4 + y = 7$
$3y = 3$
$y = 1$

Then $x = 1 + 2 = 3$

**Solution**: $(x, y) = (3, 1)$

**Verification**: 
- $2(3) + 1 = 7$ ✓
- $3 - 1 = 2$ ✓

**Solution Method 2: Matrix Form**
$$\begin{bmatrix} 2 & 1 \\ 1 & -1 \end{bmatrix}\begin{bmatrix} x \\ y \end{bmatrix} = \begin{bmatrix} 7 \\ 2 \end{bmatrix}$$

### Example 4: 2×2 Determinant
**Problem**: Find $\det(A)$ for $A = \begin{bmatrix} 3 & 4 \\ 2 & 5 \end{bmatrix}$

**Solution**:
$$\det(A) = (3)(5) - (4)(2) = 15 - 8 = 7$$

Since $\det(A) = 7 \neq 0$, the matrix $A$ is invertible.

### Example 5: 3×3 Determinant
**Problem**: Find $\det(B)$ for $B = \begin{bmatrix} 1 & 2 & 3 \\ 0 & 4 & 5 \\ 0 & 0 & 6 \end{bmatrix}$

**Solution**: 
Since $B$ is upper triangular, the determinant is the product of diagonal elements:
$$\det(B) = (1)(4)(6) = 24$$

### Example 6: 3×3 Determinant by Expansion
**Problem**: Find $\det(C)$ for $C = \begin{bmatrix} 2 & 1 & 3 \\ 1 & -1 & 2 \\ 3 & 2 & -1 \end{bmatrix}$

**Solution**: Expand along first row:
$$\det(C) = 2\begin{vmatrix} -1 & 2 \\ 2 & -1 \end{vmatrix} - 1\begin{vmatrix} 1 & 2 \\ 3 & -1 \end{vmatrix} + 3\begin{vmatrix} 1 & -1 \\ 3 & 2 \end{vmatrix}$$

$$= 2[(-1)(-1) - (2)(2)] - 1[(1)(-1) - (2)(3)] + 3[(1)(2) - (-1)(3)]$$

$$= 2[1 - 4] - 1[-1 - 6] + 3[2 + 3]$$

$$= 2(-3) - 1(-7) + 3(5)$$

$$= -6 + 7 + 15 = 16$$

### Example 7: Checking Linear Independence with Determinant
**Problem**: Are the vectors $\vec{v}_1 = \begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$, $\vec{v}_2 = \begin{bmatrix} 4 \\ 5 \\ 6 \end{bmatrix}$, $\vec{v}_3 = \begin{bmatrix} 7 \\ 8 \\ 9 \end{bmatrix}$ linearly independent?

**Solution**: Form matrix with vectors as columns:
$$A = \begin{bmatrix} 1 & 4 & 7 \\ 2 & 5 & 8 \\ 3 & 6 & 9 \end{bmatrix}$$

Compute $\det(A)$:
$$\det(A) = 1\begin{vmatrix} 5 & 8 \\ 6 & 9 \end{vmatrix} - 4\begin{vmatrix} 2 & 8 \\ 3 & 9 \end{vmatrix} + 7\begin{vmatrix} 2 & 5 \\ 3 & 6 \end{vmatrix}$$

$$= 1(45-48) - 4(18-24) + 7(12-15)$$

$$= 1(-3) - 4(-6) + 7(-3) = -3 + 24 - 21 = 0$$

Since $\det(A) = 0$, the vectors are **linearly dependent** (not independent).

---

## Data Science Applications

### Why Linear Algebra Matters in Data Science

Linear algebra is the mathematical foundation of modern data science and machine learning.

#### 1. **Data Representation**
- **Dataset as Matrix**: Each row is a data point, each column is a feature
  ```
  Dataset (100 customers × 5 features):
  [age, income, purchases, clicks, time_on_site]
  ```
- Feature vectors represent individual data points
- Matrix operations enable batch processing

#### 2. **Linear Regression**
Fitting a line $y = mx + b$ is actually solving:
$$X\vec{\beta} = \vec{y}$$

Where $X$ is the design matrix, $\vec{\beta}$ are coefficients.

**Solution**: $\vec{\beta} = (X^TX)^{-1}X^T\vec{y}$ (normal equation)

Requires: Matrix multiplication, transpose, inverse (determinant $\neq$ 0)

#### 3. **Principal Component Analysis (PCA)**
- Finds directions of maximum variance in data
- Uses eigenvectors and eigenvalues of covariance matrix
- Dimensionality reduction: 1000 features → 10 principal components
- Relies heavily on matrix operations

#### 4. **Neural Networks**
Each layer performs: $\vec{y} = f(W\vec{x} + \vec{b})$
- $W$: Weight matrix
- $\vec{x}$: Input vector
- $\vec{b}$: Bias vector
- $f$: Activation function

Deep learning = stacking many matrix multiplications!

#### 5. **Recommendation Systems**
- User-item matrix: rows = users, columns = items
- Matrix factorization: $R \approx UV^T$
- Collaborative filtering uses matrix operations

#### 6. **Computer Graphics & Image Processing**
- Images are matrices (pixels)
- Transformations (rotation, scaling) are matrix multiplications
- Convolutions in CNNs use matrix operations

#### 7. **Natural Language Processing**
- Word embeddings: words as vectors
- Document-term matrix: TF-IDF
- Cosine similarity: $\text{similarity} = \frac{\vec{u} \cdot \vec{v}}{|\vec{u}||\vec{v}|}$

### Real-World Example: Customer Similarity

```python
# Customer feature vectors
customer1 = [age: 25, income: 50k, purchases: 10]
customer2 = [age: 27, income: 55k, purchases: 12]

# Cosine similarity (dot product / magnitudes)
similarity = (25*27 + 50*55 + 10*12) / (||customer1|| * ||customer2||)
```

High similarity → recommend similar products!

---

## Practice Problems

### Basic Level

1. **Vector Operations**: Given $\vec{u} = \begin{bmatrix} 2 \\ -1 \\ 3 \end{bmatrix}$ and $\vec{v} = \begin{bmatrix} -1 \\ 2 \\ 1 \end{bmatrix}$, find:
   - (a) $\vec{u} + \vec{v}$
   - (b) $2\vec{u} - 3\vec{v}$
   - (c) $\vec{u} \cdot \vec{v}$
   - (d) $|\vec{u}|$ and $|\vec{v}|$

2. **Matrix Addition**: Compute $A + B$ where:
   $$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}, \quad B = \begin{bmatrix} 7 & 8 & 9 \\ 10 & 11 & 12 \end{bmatrix}$$

3. **Transpose**: Find $A^T$ for $A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{bmatrix}$

4. **2×2 Determinant**: Find $\det(A)$ for:
   - (a) $A = \begin{bmatrix} 2 & 3 \\ 1 & 4 \end{bmatrix}$
   - (b) $A = \begin{bmatrix} 5 & 2 \\ 10 & 4 \end{bmatrix}$

5. **System of Equations**: Solve:
   $$\begin{cases} x + 2y = 5 \\ 3x - y = 4 \end{cases}$$

### Intermediate Level

6. **Matrix Multiplication**: Compute $AB$ and $BA$ where:
   $$A = \begin{bmatrix} 1 & 2 \\ 3 & 4 \\ 5 & 6 \end{bmatrix}, \quad B = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}$$
   Are they equal? Why or why not?

7. **Angle Between Vectors**: Find the angle $\theta$ between $\vec{u} = \begin{bmatrix} 1 \\ 1 \\ 0 \end{bmatrix}$ and $\vec{v} = \begin{bmatrix} 1 \\ 0 \\ 1 \end{bmatrix}$

8. **3×3 Determinant**: Calculate $\det(A)$ for:
   $$A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 10 \end{bmatrix}$$

9. **Invertibility**: Determine if the following matrix is invertible:
   $$A = \begin{bmatrix} 2 & 1 & 3 \\ 0 & -1 & 2 \\ 1 & 4 & 1 \end{bmatrix}$$

10. **System with Parameters**: For what value of $k$ does the system have no unique solution?
    $$\begin{cases} x + 2y = 3 \\ 2x + ky = 6 \end{cases}$$

### Advanced Level

11. **Properties of Determinants**: Prove that if $A$ is a $3 \times 3$ matrix, then $\det(2A) = 8\det(A)$

12. **Matrix Powers**: If $A = \begin{bmatrix} 1 & 1 \\ 0 & 1 \end{bmatrix}$, find $A^2$, $A^3$, and guess the pattern for $A^n$

13. **Projection**: Find the projection of $\vec{v} = \begin{bmatrix} 3 \\ 4 \end{bmatrix}$ onto $\vec{u} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$

14. **Linear Independence**: Determine if the vectors are linearly independent:
    $$\vec{v}_1 = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix}, \vec{v}_2 = \begin{bmatrix} 2 \\ 1 \\ 3 \end{bmatrix}, \vec{v}_3 = \begin{bmatrix} 1 \\ -1 \\ 2 \end{bmatrix}$$

15. **Geometric Interpretation**: Show that the area of the parallelogram formed by vectors $\vec{u} = \begin{bmatrix} 2 \\ 1 \end{bmatrix}$ and $\vec{v} = \begin{bmatrix} 1 \\ 3 \end{bmatrix}$ equals $|\det(A)|$ where $A = [\vec{u} \, \vec{v}]$

### Challenge Problems

16. **Block Matrix**: If $A = \begin{bmatrix} 2 & 0 \\ 0 & 3 \end{bmatrix}$ and $B = \begin{bmatrix} 1 & 1 \\ 1 & 1 \end{bmatrix}$, find $\det\begin{bmatrix} A & 0 \\ 0 & B \end{bmatrix}$

17. **Vandermonde Determinant**: Show that:
    $$\begin{vmatrix} 1 & x & x^2 \\ 1 & y & y^2 \\ 1 & z & z^2 \end{vmatrix} = (y-x)(z-x)(z-y)$$

18. **Data Science Application**: A dataset has 100 samples and 20 features. After applying PCA, we keep components that explain 95% of variance, reducing to 5 features. Express the transformation using matrix notation and explain the role of determinants in this process.

---

## Questions/Doubts

- [ ] What's the geometric interpretation of matrix multiplication?
- [ ] When can we compute $AB$ but not $BA$?
- [ ] Why is determinant zero related to linear dependence?
- [ ] How does cofactor expansion work for $n \times n$ matrices?
- [ ] What's the connection between determinants and volumes?
- [ ] Why do we need different methods to solve linear systems?

---

## Action Items

- [x] Review lecture slides on vectors and matrices
- [ ] Complete practice problems 1-5 (Basic Level)
- [ ] Work through notebook examples (Week 1 Practice)
- [ ] Watch 3Blue1Brown "Essence of Linear Algebra" (Chapters 1-5)
- [ ] Solve textbook exercises: Linear Algebra chapter 1
- [ ] Implement matrix operations in Python (NumPy)
- [ ] Visualize vector operations geometrically

---

## Key Takeaways

1. **Vectors represent data**: Each feature vector is a point in n-dimensional space
2. **Matrices are transformations**: They transform input vectors to output vectors
3. **Dot product measures similarity**: Used extensively in ML (cosine similarity)
4. **Determinant indicates invertibility**: Zero determinant = no unique solution
5. **Matrix multiplication is NOT commutative**: Order matters! $AB \neq BA$
6. **Linear systems are everywhere**: Regression, neural networks, optimization
7. **Geometric intuition helps**: Think about vectors as arrows, matrices as transformations

---

## References

- **Textbook**: 
  - Strang, G. - *Introduction to Linear Algebra*, Chapters 1-3
  - Lay, D.C. - *Linear Algebra and Its Applications*, Chapters 1-2
- **Video Lectures**: 
  - IIT Madras Week 1 lectures (BSMA1003)
  - [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab)
  - MIT OpenCourseWare - Linear Algebra (Gilbert Strang)
- **Practice**: 
  - Week 1 Practice Notebook
  - Khan Academy: Linear Algebra basics
  - Paul's Online Math Notes: Vectors and Matrices

---

## Connection to Next Week

Week 2 will continue with:
- Advanced methods for solving systems (Gaussian elimination, row reduction)
- Matrix inverses and their computation
- Applications of determinants (Cramer's Rule)

The concepts of vectors, matrices, and determinants from Week 1 are essential building blocks for everything that follows in linear algebra!

---

**Last Updated**: 2025-11-16  
**Next Class**: Week 2 - Solving Linear Equations and Gaussian Elimination
