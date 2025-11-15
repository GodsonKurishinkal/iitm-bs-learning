# BSMA1003: Mathematics for Data Science II - Study Guide

**Course ID:** BSMA1003  
**Credits:** 4  
**Duration:** 11 weeks  
**Instructor:** Sarang S Sane  
**Prerequisites:** BSMA1001 - Mathematics for Data Science I

## 📚 Course Overview

This course introduces linear algebra, calculus, and optimization concepts with a focus on applications in machine learning and data science. You'll learn matrix operations, vector spaces, calculus techniques, and optimization methods essential for understanding ML algorithms.

## 🎯 Learning Objectives

By the end of this course, you will be able to:
- Manipulate matrices using matrix algebra
- Perform elementary row operations and Gaussian elimination
- Solve systems of linear equations
- Determine linear independence of vectors
- Find bases and dimensions of vector spaces
- Calculate distances and angles using norms and inner products
- Apply Gram-Schmidt orthogonalization process
- Find maxima and minima using derivatives and vector calculus
- Apply optimization techniques to machine learning problems

## 📖 Reference Materials

**Required Book (Available for Download):**
- **Linear Algebra** - [Download from course page](https://drive.google.com/file/d/1nMGSpQfLObffDsaojI-56kkl_tuo94hJ/view)

**Video Lectures:** [YouTube Playlist](https://www.youtube.com/playlist?list=PLZ2ps__7DhBboGlwPVSsWP8loAJCLrKc8)

---

## 📅 Week-by-Week Breakdown

### Week 1: Vectors and Matrices

**Topics Covered:**
- Introduction to vectors
- Vector operations (addition, scalar multiplication)
- Matrices and matrix notation
- Matrix operations (addition, subtraction, scalar multiplication)
- Matrix multiplication
- Systems of linear equations
- Determinants (Part 1): Definition and basic properties
- Determinants (Part 2): Computing determinants

**Learning Activities:**
1. **Read:** Linear Algebra book, Chapter 1-2
2. **Watch:** Week 1 video lectures
3. **Practice:** Matrix operations problems
4. **Code:** Implement matrix operations using NumPy

**Key Concepts:**
- **Vector:** An ordered list of numbers that can represent data points, features, or directions
- **Matrix:** A rectangular array of numbers used to represent linear transformations and datasets
- **Determinant:** A scalar value that indicates if a matrix is invertible

**Practice Problems:**
- Perform matrix addition, multiplication, and transposition
- Calculate 2×2 and 3×3 determinants
- Set up systems of equations as matrix equations

**Python Applications:**
```python
import numpy as np

# Creating vectors and matrices
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])

# Matrix operations
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Matrix multiplication
C = A @ B  # or np.dot(A, B)

# Determinant
det_A = np.linalg.det(A)
```

**Data Science Connection:** Matrices represent datasets where rows are samples and columns are features. Understanding matrix operations is crucial for data transformations and ML algorithms.

**Weekly Notebook:** `week-01-vectors-matrices.ipynb`

---

### Week 2: Solving Linear Equations

**Topics Covered:**
- Determinants (Part 3): Properties and applications
- Cramer's Rule for solving systems
- Solutions with invertible coefficient matrices
- Matrix inverse
- Echelon form and reduced echelon form
- Row reduction techniques
- Gaussian elimination method
- Applications to real-world problems

**Learning Activities:**
1. **Read:** Linear Algebra book, Chapter 3
2. **Watch:** Week 2 video lectures
3. **Practice:** Solve systems using different methods
4. **Code:** Implement Gaussian elimination in Python

**Key Concepts:**
- **Echelon Form:** A matrix form that makes solving systems easier
- **Gaussian Elimination:** Systematic method for solving linear systems
- **Matrix Inverse:** A^(-1) such that A·A^(-1) = I

**Practice Problems:**
- Solve 3×3 and 4×4 systems using Gaussian elimination
- Find matrix inverses
- Apply Cramer's rule
- Determine when systems have unique, infinite, or no solutions

**Python Applications:**
```python
# Solving linear systems
A = np.array([[2, 1, -1],
              [-3, -1, 2],
              [-2, 1, 2]])
b = np.array([8, -11, -3])

# Using NumPy's solver
x = np.linalg.solve(A, b)

# Finding matrix inverse
A_inv = np.linalg.inv(A)

# Manual Gaussian elimination
from scipy.linalg import lu
P, L, U = lu(A)
```

**Data Science Connection:** Solving linear systems is fundamental in regression analysis, optimization, and many ML algorithms like linear regression and neural networks.

**Weekly Notebook:** `week-02-solving-systems.ipynb`

---

### Week 3: Introduction to Vector Spaces

**Topics Covered:**
- Definition of vector spaces
- Axioms of vector spaces
- Properties of vector spaces
- Subspaces
- Linear dependence of vectors
- Linear independence (Part 1): Definition and tests
- Linear independence (Part 2): Applications and examples

**Learning Activities:**
1. **Read:** Linear Algebra book, Chapter 4
2. **Watch:** Week 3 video lectures
3. **Practice:** Determine linear independence
4. **Code:** Test for linear independence using rank

**Key Concepts:**
- **Vector Space:** A set with addition and scalar multiplication operations satisfying certain axioms
- **Linear Dependence:** Vectors are dependent if one can be written as a combination of others
- **Linear Independence:** No vector can be written as a combination of the others

**Practice Problems:**
- Verify vector space axioms
- Identify subspaces
- Determine if sets of vectors are linearly independent
- Find dependencies between vectors

**Python Applications:**
```python
# Testing linear independence
def is_linearly_independent(vectors):
    """Check if column vectors are linearly independent"""
    A = np.column_stack(vectors)
    rank = np.linalg.matrix_rank(A)
    return rank == len(vectors)

v1 = np.array([1, 2, 3])
v2 = np.array([2, 4, 6])  # Dependent on v1
v3 = np.array([1, 0, 1])

vectors = [v1, v2, v3]
print(f"Independent: {is_linearly_independent(vectors)}")
```

**Data Science Connection:** Feature selection in ML involves identifying linearly independent features to avoid redundancy and improve model performance.

**Weekly Notebook:** `week-03-vector-spaces.ipynb`

---

### Week 4: Basis and Dimension

**Topics Covered:**
- Definition of a basis for a vector space
- Properties of bases
- Finding bases for vector spaces
- Span of vectors
- Rank of a matrix
- Dimension of a vector space
- Rank-nullity theorem
- Using Gaussian elimination to find rank and dimension

**Learning Activities:**
1. **Read:** Linear Algebra book, Chapter 5
2. **Watch:** Week 4 video lectures
3. **Practice:** Find bases and dimensions
4. **Code:** Compute rank and bases using NumPy

**Key Concepts:**
- **Basis:** A linearly independent set that spans the vector space
- **Dimension:** The number of vectors in a basis
- **Rank:** The dimension of the column space (or row space) of a matrix

**Practice Problems:**
- Find bases for various vector spaces
- Calculate dimensions of subspaces
- Determine rank of matrices
- Apply rank-nullity theorem

**Python Applications:**
```python
# Finding rank and basis
A = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]])

rank = np.linalg.matrix_rank(A)
print(f"Rank: {rank}")

# Using SVD to find basis
U, s, Vt = np.linalg.svd(A)
# Columns of U corresponding to non-zero singular values form basis
```

**Data Science Connection:** Dimensionality reduction techniques like PCA rely on finding basis vectors that capture maximum variance in data.

**Weekly Notebook:** `week-04-basis-dimension.ipynb`

---

### Week 5: Norms and Inner Products

**Topics Covered:**
- Vector norms (L1, L2, infinity norms)
- Properties of norms
- Distance metrics
- Inner products (dot products)
- Angles between vectors
- Orthogonal vectors
- Orthogonal projections
- Applications in machine learning

**Learning Activities:**
1. **Read:** Linear Algebra book, Chapter 6
2. **Watch:** Week 5 video lectures
3. **Practice:** Calculate distances and angles
4. **Code:** Implement various distance metrics

**Key Concepts:**
- **Norm:** A measure of vector length/magnitude
- **Inner Product:** A generalization of dot product measuring similarity
- **Orthogonality:** Vectors at right angles (dot product = 0)

**Practice Problems:**
- Calculate different norms of vectors
- Find angles between vectors
- Determine orthogonal vectors
- Compute projections

**Python Applications:**
```python
# Different norms
v = np.array([3, 4])

# L1 norm (Manhattan distance)
l1_norm = np.linalg.norm(v, ord=1)

# L2 norm (Euclidean distance)
l2_norm = np.linalg.norm(v, ord=2)

# Infinity norm
linf_norm = np.linalg.norm(v, ord=np.inf)

# Dot product and angle
v1 = np.array([1, 0])
v2 = np.array([1, 1])
dot_product = np.dot(v1, v2)
angle = np.arccos(dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
angle_degrees = np.degrees(angle)
```

**Data Science Connection:** Distance metrics are crucial for clustering algorithms (K-means), nearest neighbor classifiers, and similarity measures.

**Weekly Notebook:** `week-05-norms-inner-products.ipynb`

---

### Week 6: Gram-Schmidt Orthogonalization

**Topics Covered:**
- Orthogonal and orthonormal vectors
- Orthogonal matrices
- QR decomposition
- Gram-Schmidt process
- Modified Gram-Schmidt process
- Applications to least squares problems
- Orthonormal bases

**Learning Activities:**
1. **Read:** Linear Algebra book, Chapter 7
2. **Watch:** Week 6 video lectures
3. **Practice:** Apply Gram-Schmidt process
4. **Code:** Implement orthogonalization algorithms

**Key Concepts:**
- **Orthonormal Basis:** Basis vectors that are perpendicular and unit length
- **Gram-Schmidt Process:** Algorithm to convert any basis to an orthonormal basis
- **QR Decomposition:** Factorization of a matrix into orthogonal (Q) and upper triangular (R) matrices

**Practice Problems:**
- Convert bases to orthonormal bases
- Perform QR decomposition
- Solve least squares problems using orthogonal projections

**Python Applications:**
```python
from scipy.linalg import qr

# Gram-Schmidt orthogonalization
def gram_schmidt(vectors):
    """Orthogonalize a set of vectors"""
    basis = []
    for v in vectors:
        w = v.copy()
        for b in basis:
            w -= np.dot(v, b) * b
        if np.linalg.norm(w) > 1e-10:
            basis.append(w / np.linalg.norm(w))
    return np.array(basis)

# Using SciPy's QR decomposition
A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
Q, R = qr(A)
print("Q (orthogonal):\n", Q)
print("R (upper triangular):\n", R)
```

**Data Science Connection:** QR decomposition is used in solving least squares regression and in eigenvalue algorithms essential for PCA.

**Weekly Notebook:** `week-06-gram-schmidt.ipynb`

---

### Week 7: Single Variable Calculus - Derivatives

**Topics Covered:**
- Review of limits and continuity
- Definition of derivative
- Differentiation rules
- Critical points
- First derivative test
- Second derivative test
- Concavity and inflection points
- Finding maxima and minima
- Optimization problems

**Learning Activities:**
1. **Read:** Linear Algebra book, Calculus section
2. **Watch:** Week 7 video lectures
3. **Practice:** Optimization problems
4. **Code:** Implement numerical optimization

**Key Concepts:**
- **Derivative:** Rate of change of a function
- **Critical Point:** Where derivative is zero or undefined
- **Optimization:** Finding maximum or minimum values

**Practice Problems:**
- Find derivatives of complex functions
- Locate and classify critical points
- Solve optimization problems
- Apply calculus to real-world scenarios

**Python Applications:**
```python
from scipy.optimize import minimize_scalar
from scipy.misc import derivative

# Define function
def f(x):
    return x**4 - 4*x**3 + 4*x**2

# Find minimum
result = minimize_scalar(f, bounds=(-10, 10), method='bounded')
print(f"Minimum at x = {result.x}, f(x) = {result.fun}")

# Numerical derivative
x_point = 2.0
slope = derivative(f, x_point, dx=1e-6)
print(f"Derivative at x={x_point}: {slope}")

# Plot function and derivative
import matplotlib.pyplot as plt
x = np.linspace(-1, 5, 1000)
y = f(x)
dy = [derivative(f, xi, dx=1e-6) for xi in x]

plt.figure(figsize=(12, 5))
plt.subplot(121)
plt.plot(x, y)
plt.title('Function f(x)')
plt.subplot(122)
plt.plot(x, dy)
plt.title("Derivative f'(x)")
plt.axhline(y=0, color='r', linestyle='--')
plt.tight_layout()
```

**Data Science Connection:** Gradient descent, the fundamental optimization algorithm in machine learning, relies on derivatives to minimize loss functions.

**Weekly Notebook:** `week-07-single-var-calculus.ipynb`

---

### Week 8: Multivariable Calculus - Partial Derivatives

**Topics Covered:**
- Functions of multiple variables
- Partial derivatives
- Gradient vectors
- Directional derivatives
- Chain rule for multivariable functions
- Applications to machine learning

**Learning Activities:**
1. **Read:** Linear Algebra book, Multivariable calculus section
2. **Watch:** Week 8 video lectures
3. **Practice:** Compute gradients
4. **Code:** Implement gradient calculations

**Key Concepts:**
- **Partial Derivative:** Derivative with respect to one variable, holding others constant
- **Gradient:** Vector of all partial derivatives
- **Directional Derivative:** Rate of change in a specific direction

**Practice Problems:**
- Calculate partial derivatives
- Find gradient vectors
- Compute directional derivatives
- Apply chain rule in multiple dimensions

**Python Applications:**
```python
from scipy.optimize import approx_fprime

# Function of two variables
def f(x):
    return x[0]**2 + x[1]**2 - 2*x[0]*x[1]

# Compute gradient at a point
point = np.array([1.0, 2.0])
gradient = approx_fprime(point, f, epsilon=1e-6)
print(f"Gradient at {point}: {gradient}")

# Visualize function and gradient
x1 = np.linspace(-3, 3, 100)
x2 = np.linspace(-3, 3, 100)
X1, X2 = np.meshgrid(x1, x2)
Z = X1**2 + X2**2 - 2*X1*X2

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(121, projection='3d')
ax.plot_surface(X1, X2, Z, cmap='viridis', alpha=0.8)
ax.set_title('3D Surface')
```

**Data Science Connection:** Gradients guide optimization in neural networks through backpropagation. Understanding partial derivatives is crucial for multivariate regression.

**Weekly Notebook:** `week-08-multivariable-calculus.ipynb`

---

### Week 9: Optimization - Finding Extrema

**Topics Covered:**
- Critical points in multiple dimensions
- Hessian matrix
- Second derivative test for multivariable functions
- Constrained optimization
- Lagrange multipliers
- Convex functions
- Global vs local extrema

**Learning Activities:**
1. **Read:** Linear Algebra book, Optimization section
2. **Watch:** Week 9 video lectures
3. **Practice:** Solve optimization problems
4. **Code:** Implement optimization algorithms

**Key Concepts:**
- **Hessian Matrix:** Matrix of second-order partial derivatives
- **Lagrange Multipliers:** Method for constrained optimization
- **Convex Optimization:** Special case where local minimum is global minimum

**Practice Problems:**
- Find and classify critical points using Hessian
- Solve constrained optimization with Lagrange multipliers
- Identify convex functions
- Apply optimization to real-world problems

**Python Applications:**
```python
from scipy.optimize import minimize

# Unconstrained optimization
def f(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

result = minimize(f, x0=[0, 0], method='BFGS')
print(f"Minimum at: {result.x}")

# Constrained optimization
from scipy.optimize import minimize

def objective(x):
    return x[0]**2 + x[1]**2

def constraint(x):
    return x[0] + x[1] - 1

constraints = {'type': 'eq', 'fun': constraint}
result = minimize(objective, x0=[0.5, 0.5], constraints=constraints)
print(f"Constrained minimum at: {result.x}")

# Compute Hessian
from scipy.optimize import approx_fprime
from numdifftools import Hessian

hess_func = Hessian(lambda x: (x[0]-1)**2 + (x[1]-2)**2)
H = hess_func([0, 0])
print(f"Hessian:\n{H}")
```

**Data Science Connection:** All machine learning training involves optimization - minimizing loss functions subject to constraints. Understanding optimization theory is key to designing better algorithms.

**Weekly Notebook:** `week-09-optimization-extrema.ipynb`

---

### Week 10: Applications to Machine Learning

**Topics Covered:**
- Linear regression as optimization
- Gradient descent algorithm
- Normal equations
- Regularization (L1 and L2)
- Principal Component Analysis (PCA)
- Eigenvalues and eigenvectors
- Applications of linear algebra in ML

**Learning Activities:**
1. **Read:** Supplementary ML materials
2. **Watch:** Week 10 video lectures
3. **Practice:** Implement ML algorithms from scratch
4. **Code:** Build linear regression with gradient descent

**Key Concepts:**
- **Gradient Descent:** Iterative optimization using gradients
- **Normal Equations:** Direct solution to linear regression
- **PCA:** Dimensionality reduction using eigenvectors
- **Regularization:** Adding constraints to prevent overfitting

**Practice Problems:**
- Implement gradient descent from scratch
- Solve linear regression using normal equations
- Apply PCA to high-dimensional data
- Compare different optimization methods

**Python Applications:**
```python
# Linear regression with gradient descent
def gradient_descent_linear_regression(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    theta = np.zeros(n)
    
    for i in range(iterations):
        predictions = X @ theta
        errors = predictions - y
        gradient = (1/m) * (X.T @ errors)
        theta -= learning_rate * gradient
    
    return theta

# Using normal equations
def normal_equations(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# PCA implementation
from sklearn.decomposition import PCA

# Create sample data
X = np.random.randn(100, 5)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)
print(f"Explained variance: {pca.explained_variance_ratio_}")
```

**Data Science Connection:** This week ties everything together, showing how linear algebra and calculus concepts directly power machine learning algorithms.

**Weekly Notebook:** `week-10-ml-applications.ipynb`

---

### Week 11: Review and Advanced Topics

**Topics Covered:**
- Comprehensive review of all concepts
- Matrix decompositions (SVD, eigendecomposition)
- Advanced optimization techniques
- Numerical stability considerations
- Computational complexity
- Integration of concepts in real projects
- Exam preparation

**Learning Activities:**
1. **Review:** All previous weeks' materials
2. **Practice:** Comprehensive problem sets
3. **Project:** Complete end-to-end ML project using course concepts
4. **Code:** Implement complex algorithms

**Project Ideas:**
1. **Image Compression with SVD:** Use singular value decomposition to compress images
2. **Custom Linear Regression:** Build from scratch with multiple regularization options
3. **PCA on Real Dataset:** Apply dimensionality reduction to high-dimensional dataset
4. **Optimization Comparison:** Compare gradient descent variants on different functions

**Python Applications:**
```python
# SVD for image compression
from PIL import Image

def compress_image_svd(image_array, k):
    """Compress image using SVD keeping k singular values"""
    U, s, Vt = np.linalg.svd(image_array, full_matrices=False)
    # Keep only k components
    compressed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
    return compressed

# Load and compress image
img = Image.open('image.jpg').convert('L')  # Convert to grayscale
img_array = np.array(img)

compressed_img = compress_image_svd(img_array, k=50)
print(f"Compression ratio: {k / min(img_array.shape):.2%}")
```

**Review Checklist:**
- [ ] Master all matrix operations
- [ ] Understand vector spaces and linear independence
- [ ] Calculate bases and dimensions fluently
- [ ] Apply Gram-Schmidt orthogonalization
- [ ] Solve optimization problems
- [ ] Compute gradients and Hessians
- [ ] Understand ML applications of linear algebra

**Weekly Notebook:** `week-11-comprehensive-review.ipynb`

---

## 🎯 Assessment Structure

- **Weekly Online Assignments:** 10-20% (varies by term)
- **Quiz 1 (In-person):** 15-20%
- **Quiz 2 (In-person):** 15-20%
- **End Term Exam (In-person):** 50-60%

**Passing Grade:** 40% overall with at least 40% in end-term exam

---

## 💡 Study Tips

1. **Master NumPy:** Practice all concepts using NumPy - it's essential for data science
2. **Visualize Everything:** Use matplotlib to visualize matrices, vectors, transformations
3. **Connect to ML:** Always ask "How is this used in machine learning?"
4. **Practice Daily:** Spend 1-2 hours daily on problems and coding
5. **Build Intuition:** Don't just memorize formulas - understand what they mean
6. **Work Through Examples:** Do all textbook examples yourself before checking solutions
7. **Form Study Groups:** Explain concepts to peers - teaching solidifies understanding
8. **Use Online Resources:** Khan Academy, 3Blue1Brown's Essence of Linear Algebra

---

## 🔗 Important Links

- **Course Page:** https://study.iitm.ac.in/ds/course_pages/BSMA1003.html
- **Video Lectures:** [YouTube Playlist](https://www.youtube.com/playlist?list=PLZ2ps__7DhBboGlwPVSsWP8loAJCLrKc8)
- **Reference Book:** [Linear Algebra PDF](https://drive.google.com/file/d/1nMGSpQfLObffDsaojI-56kkl_tuo94hJ/view)
- **Instructor Website:** [Prof. Sarang S Sane](https://home.iitm.ac.in/sarang/)

---

## 📚 Recommended Additional Resources

- **3Blue1Brown - Essence of Linear Algebra:** Visual introduction to linear algebra concepts
- **Khan Academy Linear Algebra:** Practice problems and video explanations
- **Gilbert Strang's Linear Algebra Course (MIT):** Classic comprehensive course
- **NumPy Documentation:** Master the library used throughout the course

---

## ✅ Weekly Checklist Template

```markdown
### Week X Checklist
- [ ] Watched all video lectures
- [ ] Read assigned textbook sections
- [ ] Completed all practice problems
- [ ] Created Jupyter notebook with implementations
- [ ] Tested code on different examples
- [ ] Took comprehensive notes
- [ ] Submitted online assignment
- [ ] Connected concepts to previous weeks
- [ ] Identified ML applications
```

---

**Remember:** Linear algebra is the foundation of modern machine learning and data science. Every hour spent mastering these concepts will pay dividends throughout your entire career!
