# Week 10: Applications to Machine Learning - Optimization in Action

---
Date: 2025-11-22
Course: BSMA1003 - Mathematics for Data Science II
Level: Foundation
Week: 10 of 11
Source: IIT Madras Mathematics II Week 10
Topic Area: Machine Learning Applications
Tags: #BSMA1003 #MachineLearning #GradientDescent #LinearRegression #PCA #Week10 #Foundation
---

## Overview (BLUF)

**One-sentence definition**: Machine learning is fundamentally applied optimization—every model training algorithm uses calculus and linear algebra concepts to find parameters that minimize prediction error.

**Why it matters**: Understanding the mathematics behind ML algorithms transforms you from a user of libraries to a designer of new methods. When `sklearn` doesn't work, or when you need custom loss functions for domain-specific problems, you need to know how gradient descent, normal equations, and PCA actually work under the hood.

**When to use**: Training any supervised learning model (regression, classification), reducing dimensionality with PCA, implementing custom optimization algorithms, debugging convergence issues, designing novel ML architectures, understanding why hyperparameters matter.

**Prerequisites**: Partial derivatives and gradients ([week-08-rank-nullity.md](week-08-rank-nullity.md)), optimization and Hessian matrix ([week-09-optimization-basics.md](week-09-optimization-basics.md)), eigenvalues and eigenvectors ([week-05-eigenvalues-eigenvectors.md](week-05-eigenvalues-eigenvectors.md)), matrix operations ([week-02-matrix-operations.md](week-02-matrix-operations.md)).

---

## Core Theory

### 1. Linear Regression as Optimization Problem

**The Supervised Learning Setup**:

We have training data: $\{(\mathbf{x}_1, y_1), (\mathbf{x}_2, y_2), \ldots, (\mathbf{x}_m, y_m)\}$

- $\mathbf{x}_i \in \mathbb{R}^n$: feature vector (inputs)
- $y_i \in \mathbb{R}$: target value (output)
- Goal: Find function $h(\mathbf{x})$ that predicts $y$ from $\mathbf{x}$

**Linear Hypothesis**:
$$h_{\boldsymbol{\theta}}(\mathbf{x}) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \cdots + \theta_n x_n = \boldsymbol{\theta}^T \mathbf{x}$$

where $\boldsymbol{\theta} = [\theta_0, \theta_1, \ldots, \theta_n]^T$ are **parameters** (what we optimize).

**Cost Function** (Mean Squared Error):
$$J(\boldsymbol{\theta}) = \frac{1}{2m} \sum_{i=1}^{m} (h_{\boldsymbol{\theta}}(\mathbf{x}_i) - y_i)^2$$

**Optimization Goal**:
$$\boldsymbol{\theta}^* = \arg\min_{\boldsymbol{\theta}} J(\boldsymbol{\theta})$$

**Why squared error?**:
1. Penalizes large errors more than small ones
2. Differentiable everywhere (smooth optimization)
3. Has unique global minimum (convex function)
4. Statistical interpretation: maximum likelihood under Gaussian noise

**Matrix Form**:

Let $X$ be the **design matrix** ($m \times (n+1)$):
$$X = \begin{bmatrix}
1 & x_1^{(1)} & x_2^{(1)} & \cdots & x_n^{(1)} \\
1 & x_1^{(2)} & x_2^{(2)} & \cdots & x_n^{(2)} \\
\vdots & \vdots & \vdots & \ddots & \vdots \\
1 & x_1^{(m)} & x_2^{(m)} & \cdots & x_n^{(m)}
\end{bmatrix}$$

Let $\mathbf{y} = [y_1, y_2, \ldots, y_m]^T$

Then:
$$J(\boldsymbol{\theta}) = \frac{1}{2m} \|X\boldsymbol{\theta} - \mathbf{y}\|^2 = \frac{1}{2m}(X\boldsymbol{\theta} - \mathbf{y})^T(X\boldsymbol{\theta} - \mathbf{y})$$

#### Example 1: Simple Linear Regression (1 Feature)

**Data**: House sizes and prices

| Size (sq ft) | Price ($1000s) |
|--------------|----------------|
| 1000         | 200            |
| 1500         | 250            |
| 2000         | 300            |
| 2500         | 350            |

**Model**: $h_\theta(x) = \theta_0 + \theta_1 x$

**Goal**: Find $\theta_0$ (intercept) and $\theta_1$ (slope) that minimize squared error.

**Cost function for this data**:
$$J(\theta_0, \theta_1) = \frac{1}{8}\sum_{i=1}^{4}(\theta_0 + \theta_1 x_i - y_i)^2$$

Let's try some values:

**Try 1**: $\theta_0 = 0$, $\theta_1 = 0.1$
- Predictions: $100, 150, 200, 250$
- Errors: $-100, -100, -100, -100$
- $J = \frac{1}{8}(10000 + 10000 + 10000 + 10000) = 5000$

**Try 2**: $\theta_0 = 100$, $\theta_1 = 0.1$
- Predictions: $200, 250, 300, 350$
- Errors: $0, 0, 0, 0$
- $J = 0$ ← **Perfect fit!**

```python
import numpy as np
import matplotlib.pyplot as plt

# Data
X = np.array([1000, 1500, 2000, 2500])
y = np.array([200, 250, 300, 350])

# Add intercept term (column of 1s)
X_with_intercept = np.c_[np.ones(len(X)), X]

# Compute cost for different theta values
theta_0_vals = np.linspace(0, 200, 100)
theta_1_vals = np.linspace(0, 0.2, 100)
Theta0, Theta1 = np.meshgrid(theta_0_vals, theta_1_vals)

# Compute cost for each combination
J_vals = np.zeros_like(Theta0)
for i in range(len(theta_0_vals)):
    for j in range(len(theta_1_vals)):
        theta = np.array([Theta0[j, i], Theta1[j, i]])
        predictions = X_with_intercept @ theta
        errors = predictions - y
        J_vals[j, i] = (1/(2*len(y))) * np.sum(errors**2)

# Plot cost function
fig = plt.figure(figsize=(14, 5))

# 3D surface
ax1 = fig.add_subplot(131, projection='3d')
ax1.plot_surface(Theta0, Theta1, J_vals, cmap='viridis', alpha=0.7)
ax1.scatter([100], [0.1], [0], color='red', s=100, label='Optimal')
ax1.set_xlabel('θ₀')
ax1.set_ylabel('θ₁')
ax1.set_zlabel('Cost J(θ)')
ax1.set_title('Cost Function Surface')

# Contour plot
ax2 = fig.add_subplot(132)
contours = ax2.contour(Theta0, Theta1, J_vals, levels=20, cmap='viridis')
ax2.clabel(contours, inline=True, fontsize=8)
ax2.plot(100, 0.1, 'r*', markersize=15, label='Optimal θ')
ax2.set_xlabel('θ₀')
ax2.set_ylabel('θ₁')
ax2.set_title('Cost Function Contours')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Best fit line
ax3 = fig.add_subplot(133)
ax3.scatter(X, y, color='blue', s=100, label='Data')
x_line = np.linspace(900, 2600, 100)
y_line = 100 + 0.1 * x_line
ax3.plot(x_line, y_line, 'r-', linewidth=2, label='Best fit: y=100+0.1x')
ax3.set_xlabel('Size (sq ft)')
ax3.set_ylabel('Price ($1000s)')
ax3.set_title('Linear Regression Fit')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

---

### 2. Gradient Descent Algorithm

**Intuition**: Imagine you're on a mountainside in fog and want to get to the valley. You can't see the bottom, but you can feel which direction is downhill. Take small steps in the steepest downhill direction repeatedly—eventually you'll reach a valley.

**Algorithm**:

1. **Initialize**: Start with random $\boldsymbol{\theta}^{(0)}$
2. **Repeat** until convergence:
   $$\boldsymbol{\theta}^{(t+1)} = \boldsymbol{\theta}^{(t)} - \alpha \nabla J(\boldsymbol{\theta}^{(t)})$$

where:
- $\alpha > 0$: **learning rate** (step size)
- $\nabla J$: **gradient** of cost function (direction of steepest ascent)
- $-\nabla J$: direction of steepest **descent**

**Gradient of MSE**:
$$\nabla J(\boldsymbol{\theta}) = \frac{1}{m} X^T(X\boldsymbol{\theta} - \mathbf{y})$$

**Update Rule** (vectorized):
$$\boldsymbol{\theta} := \boldsymbol{\theta} - \frac{\alpha}{m} X^T(X\boldsymbol{\theta} - \mathbf{y})$$

**For each parameter individually**:
$$\theta_j := \theta_j - \alpha \frac{\partial J}{\partial \theta_j}$$

where:
$$\frac{\partial J}{\partial \theta_j} = \frac{1}{m}\sum_{i=1}^{m}(h_{\boldsymbol{\theta}}(\mathbf{x}_i) - y_i)x_j^{(i)}$$

**Convergence**: Stop when:
- Gradient magnitude $\|\nabla J\|$ is small (e.g., $< 10^{-6}$)
- Cost change is small (e.g., $|J^{(t+1)} - J^{(t)}| < 10^{-6}$)
- Maximum iterations reached

#### Example 2: Gradient Descent Implementation

**Problem**: Find optimal $\theta_0$ and $\theta_1$ for house price data using gradient descent.

```python
def gradient_descent(X, y, learning_rate=0.01, iterations=1000, tolerance=1e-6):
    """
    Perform gradient descent to minimize cost function.

    Parameters:
    -----------
    X : array, shape (m, n+1) - design matrix with intercept column
    y : array, shape (m,) - target values
    learning_rate : float - step size α
    iterations : int - maximum number of iterations
    tolerance : float - convergence threshold

    Returns:
    --------
    theta : array, shape (n+1,) - optimal parameters
    cost_history : list - cost at each iteration
    """
    m, n = X.shape
    theta = np.zeros(n)  # Initialize parameters to zero
    cost_history = []

    for iteration in range(iterations):
        # Compute predictions
        predictions = X @ theta

        # Compute errors
        errors = predictions - y

        # Compute gradient
        gradient = (1/m) * (X.T @ errors)

        # Update parameters
        theta = theta - learning_rate * gradient

        # Compute cost
        cost = (1/(2*m)) * np.sum(errors**2)
        cost_history.append(cost)

        # Check convergence
        if iteration > 0 and abs(cost_history[-1] - cost_history[-2]) < tolerance:
            print(f"Converged at iteration {iteration}")
            break

    return theta, cost_history

# Apply to house price data
X = np.array([1000, 1500, 2000, 2500])
y = np.array([200, 250, 300, 350])

# Add intercept
X_with_intercept = np.c_[np.ones(len(X)), X]

# Run gradient descent
theta_optimal, cost_history = gradient_descent(
    X_with_intercept, y,
    learning_rate=0.0000001,  # Small learning rate for stability
    iterations=10000
)

print(f"Optimal parameters: θ₀={theta_optimal[0]:.2f}, θ₁={theta_optimal[1]:.6f}")
print(f"Final cost: {cost_history[-1]:.6f}")

# Visualize convergence
plt.figure(figsize=(14, 5))

plt.subplot(131)
plt.plot(cost_history, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cost J(θ)')
plt.title('Cost vs Iteration')
plt.grid(True, alpha=0.3)

plt.subplot(132)
plt.semilogy(cost_history, linewidth=2)
plt.xlabel('Iteration')
plt.ylabel('Cost J(θ) (log scale)')
plt.title('Cost vs Iteration (Log Scale)')
plt.grid(True, alpha=0.3)

plt.subplot(133)
plt.scatter(X, y, color='blue', s=100, label='Data')
x_line = np.linspace(900, 2600, 100)
y_line = theta_optimal[0] + theta_optimal[1] * x_line
plt.plot(x_line, y_line, 'r-', linewidth=2, label=f'Fitted: y={theta_optimal[0]:.1f}+{theta_optimal[1]:.4f}x')
plt.xlabel('Size (sq ft)')
plt.ylabel('Price ($1000s)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**Key Observations**:
1. Cost decreases monotonically (if learning rate is appropriate)
2. Convergence is asymptotic (approaches minimum gradually)
3. Learning rate affects convergence speed

---

### 3. Learning Rate Selection

**The Goldilocks Problem**: Learning rate $\alpha$ must be "just right"

**Too small** ($\alpha$ very small):
- ✅ Guaranteed to decrease cost each step
- ❌ Extremely slow convergence
- ❌ May take millions of iterations

**Too large** ($\alpha$ very large):
- ❌ Can overshoot minimum
- ❌ Cost may increase or oscillate
- ❌ May diverge (cost → ∞)

**Just right**:
- ✅ Fast convergence
- ✅ Stable decrease in cost
- ✅ Reaches minimum in reasonable time

#### Example 3: Effect of Learning Rate

```python
# Test different learning rates
learning_rates = [0.00000001, 0.0000001, 0.000001, 0.00001]
colors = ['red', 'orange', 'green', 'blue']

plt.figure(figsize=(14, 5))

for lr, color in zip(learning_rates, colors):
    theta, cost_hist = gradient_descent(
        X_with_intercept, y,
        learning_rate=lr,
        iterations=5000
    )
    plt.plot(cost_hist[:500], color=color, linewidth=2, label=f'α={lr}')

plt.xlabel('Iteration')
plt.ylabel('Cost J(θ)')
plt.title('Learning Rate Comparison')
plt.legend()
plt.grid(True, alpha=0.3)
plt.yscale('log')
plt.show()
```

**Adaptive Learning Rates**: Modern optimizers (Adam, RMSprop) automatically adjust $\alpha$ during training.

---

### 4. Normal Equations - Direct Solution

**Key Insight**: For linear regression, we can solve $\nabla J(\boldsymbol{\theta}) = \mathbf{0}$ directly!

**Derivation**:

Cost function: $J(\boldsymbol{\theta}) = \frac{1}{2m}(X\boldsymbol{\theta} - \mathbf{y})^T(X\boldsymbol{\theta} - \mathbf{y})$

Expand:
$$J(\boldsymbol{\theta}) = \frac{1}{2m}(\boldsymbol{\theta}^TX^TX\boldsymbol{\theta} - 2\boldsymbol{\theta}^TX^T\mathbf{y} + \mathbf{y}^T\mathbf{y})$$

Gradient:
$$\nabla J(\boldsymbol{\theta}) = \frac{1}{m}(X^TX\boldsymbol{\theta} - X^T\mathbf{y})$$

Set equal to zero:
$$X^TX\boldsymbol{\theta} = X^T\mathbf{y}$$

**Normal Equations**:
$$\boldsymbol{\theta}^* = (X^TX)^{-1}X^T\mathbf{y}$$

**Advantages**:
- ✅ No learning rate to tune
- ✅ No iterations needed
- ✅ Exact solution (no approximation)

**Disadvantages**:
- ❌ Requires matrix inversion: $O(n^3)$ complexity
- ❌ Slow for large $n$ (many features)
- ❌ Doesn't work if $X^TX$ is singular (non-invertible)
- ❌ Doesn't generalize to other cost functions

**When to use**:
- Use **normal equations** if $n < 10000$ (small number of features)
- Use **gradient descent** if $n$ is large or for non-linear models

#### Example 4: Normal Equations Solution

```python
def normal_equations(X, y):
    """
    Solve linear regression using normal equations.

    Returns optimal theta: (XᵀX)⁻¹Xᵀy
    """
    theta = np.linalg.inv(X.T @ X) @ X.T @ y
    return theta

# Apply to house data
theta_normal = normal_equations(X_with_intercept, y)
print(f"Normal equations solution: θ₀={theta_normal[0]:.2f}, θ₁={theta_normal[1]:.6f}")

# Compare with gradient descent
print(f"Gradient descent solution: θ₀={theta_optimal[0]:.2f}, θ₁={theta_optimal[1]:.6f}")
print(f"Difference: {np.linalg.norm(theta_normal - theta_optimal):.10f}")
```

**Result**: Both methods give (essentially) the same answer! Normal equations is exact (up to numerical precision).

---

### 5. Regularization - Preventing Overfitting

**The Problem**: With many features, model can fit training data perfectly but fail on new data (**overfitting**).

**Solution**: Add penalty for large parameter values.

**L2 Regularization (Ridge)**:
$$J(\boldsymbol{\theta}) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\boldsymbol{\theta}}(\mathbf{x}_i) - y_i)^2 + \frac{\lambda}{2m}\sum_{j=1}^{n}\theta_j^2$$

where $\lambda \geq 0$ is **regularization parameter**.

**Effect**: Shrinks parameters toward zero, preventing overfitting.

**Modified Normal Equations**:
$$\boldsymbol{\theta}^* = (X^TX + \lambda I)^{-1}X^T\mathbf{y}$$

where $I$ is $(n+1) \times (n+1)$ identity matrix (but don't regularize intercept $\theta_0$).

**L1 Regularization (Lasso)**:
$$J(\boldsymbol{\theta}) = \frac{1}{2m}\sum_{i=1}^{m}(h_{\boldsymbol{\theta}}(\mathbf{x}_i) - y_i)^2 + \frac{\lambda}{m}\sum_{j=1}^{n}|\theta_j|$$

**Effect**: Produces **sparse** solutions (many $\theta_j = 0$), useful for feature selection.

#### Example 5: Ridge Regression

```python
def ridge_regression(X, y, lambda_param=1.0):
    """Ridge regression with L2 regularization."""
    n = X.shape[1]
    I = np.eye(n)
    I[0, 0] = 0  # Don't regularize intercept
    theta = np.linalg.inv(X.T @ X + lambda_param * I) @ X.T @ y
    return theta

# Generate data with noise
np.random.seed(42)
X_train = np.random.randn(20, 10)  # 20 samples, 10 features
X_train = np.c_[np.ones(20), X_train]  # Add intercept
true_theta = np.random.randn(11)
y_train = X_train @ true_theta + 0.1 * np.random.randn(20)

# Test data
X_test = np.random.randn(10, 10)
X_test = np.c_[np.ones(10), X_test]
y_test = X_test @ true_theta + 0.1 * np.random.randn(10)

# Compare different lambda values
lambdas = [0, 0.1, 1.0, 10.0]
for lam in lambdas:
    theta = ridge_regression(X_train, y_train, lam)
    train_error = np.mean((X_train @ theta - y_train)**2)
    test_error = np.mean((X_test @ theta - y_test)**2)
    print(f"λ={lam:4.1f}: Train error={train_error:.4f}, Test error={test_error:.4f}")
```

**Typical Output**:
- λ=0: Train error low, test error high (overfitting)
- λ=1: Train error slightly higher, test error lower (good generalization)
- λ=10: Both errors higher (underfitting)

---

### 6. Principal Component Analysis (PCA)

**The Problem**: High-dimensional data ($n$ large) is:
- Computationally expensive
- Difficult to visualize
- Often has redundant features (correlated)

**Solution**: Find **lower-dimensional representation** that captures most information.

**PCA Goal**: Find $k$ orthogonal directions (principal components) that capture maximum variance in data.

**Mathematical Formulation**:

Given data $\mathbf{x}_1, \ldots, \mathbf{x}_m \in \mathbb{R}^n$:

1. **Center data**: $\tilde{\mathbf{x}}_i = \mathbf{x}_i - \boldsymbol{\mu}$ where $\boldsymbol{\mu} = \frac{1}{m}\sum_{i=1}^{m}\mathbf{x}_i$

2. **Covariance matrix**:
   $$\Sigma = \frac{1}{m}\sum_{i=1}^{m}\tilde{\mathbf{x}}_i\tilde{\mathbf{x}}_i^T = \frac{1}{m}\tilde{X}^T\tilde{X}$$

3. **Eigendecomposition**: Find eigenvectors $\mathbf{v}_1, \ldots, \mathbf{v}_n$ and eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_n$
   $$\Sigma \mathbf{v}_i = \lambda_i \mathbf{v}_i$$

4. **Select top $k$ eigenvectors**: These are principal components

5. **Project data**: $\mathbf{z}_i = U_k^T \tilde{\mathbf{x}}_i$ where $U_k = [\mathbf{v}_1, \ldots, \mathbf{v}_k]$

**Variance Explained**:
$$\text{Proportion of variance} = \frac{\sum_{i=1}^{k}\lambda_i}{\sum_{i=1}^{n}\lambda_i}$$

**Intuition**: Principal components are directions of maximum variance. First PC has most variance, second PC (orthogonal to first) has second-most, etc.

#### Example 6: PCA on Iris Dataset

```python
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
iris = load_iris()
X = iris.data  # 150 samples, 4 features
y = iris.target

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Results
print(f"Original dimensions: {X.shape}")
print(f"Reduced dimensions: {X_pca.shape}")
print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.2%}")

# Visualize
plt.figure(figsize=(14, 5))

# Original data (first 2 features)
plt.subplot(131)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.title('Original Features (2D projection)')
plt.colorbar(label='Species')

# PCA transformed data
plt.subplot(132)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA Projection (2D)')
plt.colorbar(label='Species')

# Variance explained
plt.subplot(133)
pca_full = PCA()
pca_full.fit(X_scaled)
cumsum = np.cumsum(pca_full.explained_variance_ratio_)
plt.plot(range(1, 5), cumsum, 'bo-', linewidth=2, markersize=8)
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Variance Explained')
plt.title('Scree Plot')
plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()
plt.show()
```

#### Example 7: PCA from Scratch

```python
def pca_from_scratch(X, n_components=2):
    """
    Implement PCA manually using eigendecomposition.

    Parameters:
    -----------
    X : array, shape (m, n) - data matrix
    n_components : int - number of principal components

    Returns:
    --------
    X_transformed : array, shape (m, n_components) - projected data
    components : array, shape (n_components, n) - principal components
    explained_var : array - variance explained by each component
    """
    # Step 1: Center the data
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # Step 2: Compute covariance matrix
    cov_matrix = (1 / len(X)) * (X_centered.T @ X_centered)

    # Step 3: Eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Step 4: Sort by eigenvalues (descending)
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Step 5: Select top k eigenvectors
    components = eigenvectors[:, :n_components]

    # Step 6: Project data
    X_transformed = X_centered @ components

    # Compute variance explained
    explained_var = eigenvalues[:n_components] / np.sum(eigenvalues)

    return X_transformed, components.T, explained_var

# Test on iris data
X_pca_scratch, components, var_explained = pca_from_scratch(X_scaled, n_components=2)

print("Components from scratch:")
print(components)
print("\nComponents from sklearn:")
print(pca.components_)
print("\nVariance explained (scratch):", var_explained)
print("Variance explained (sklearn):", pca.explained_variance_ratio_)
```

---

## Data Science Applications

### 1. Hyperparameter Optimization

**Problem**: Find best learning rate, regularization, batch size, etc.

**Approach**: Use gradient-free optimization (grid search, random search, Bayesian optimization).

```python
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge

# Define parameter grid
param_grid = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]}

# Grid search with cross-validation
ridge = Ridge()
grid_search = GridSearchCV(ridge, param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train[:, 1:], y_train)  # Exclude intercept column

print(f"Best alpha: {grid_search.best_params_['alpha']}")
print(f"Best CV score: {-grid_search.best_score_:.4f}")
```

### 2. Feature Engineering with PCA

**Use case**: Reduce 1000 features to 50 while keeping 95% of information.

```python
# High-dimensional data
np.random.seed(42)
X_high_dim = np.random.randn(100, 1000)

# Apply PCA
pca = PCA(n_components=0.95)  # Keep 95% variance
X_reduced = pca.fit_transform(X_high_dim)

print(f"Original: {X_high_dim.shape}")
print(f"Reduced: {X_reduced.shape}")
print(f"Reduction: {100*(1 - X_reduced.shape[1]/X_high_dim.shape[1]):.1f}%")
```

### 3. Image Compression with PCA

```python
from PIL import Image

# Load grayscale image
img = Image.open('image.jpg').convert('L')
img_array = np.array(img)

# Apply PCA
pca = PCA(n_components=50)  # Keep 50 components
img_transformed = pca.fit_transform(img_array)
img_compressed = pca.inverse_transform(img_transformed)

# Visualize
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(img_array, cmap='gray')
axes[0].set_title('Original')
axes[1].imshow(img_compressed, cmap='gray')
axes[1].set_title(f'Compressed (k={pca.n_components})')
axes[2].plot(np.cumsum(pca.explained_variance_ratio_))
axes[2].set_title('Variance Explained')
axes[2].set_xlabel('Component')
axes[2].grid(True)
plt.show()
```

### 4. Online Learning with Stochastic Gradient Descent

**Batch Gradient Descent**: Use all data for each update (slow for large datasets)

**Stochastic Gradient Descent (SGD)**: Use one sample at a time

**Mini-batch SGD**: Use small batch of samples

```python
def mini_batch_sgd(X, y, learning_rate=0.01, batch_size=32, epochs=100):
    """Mini-batch stochastic gradient descent."""
    m, n = X.shape
    theta = np.zeros(n)

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(m)
        X_shuffled = X[indices]
        y_shuffled = y[indices]

        # Process mini-batches
        for i in range(0, m, batch_size):
            X_batch = X_shuffled[i:i+batch_size]
            y_batch = y_shuffled[i:i+batch_size]

            # Compute gradient on batch
            predictions = X_batch @ theta
            errors = predictions - y_batch
            gradient = (1/len(X_batch)) * (X_batch.T @ errors)

            # Update
            theta -= learning_rate * gradient

    return theta
```

---

## Common Pitfalls and Misconceptions

### Pitfall 1: Not Standardizing Features

❌ **Wrong**: Run PCA or gradient descent on raw features with different scales.

✅ **Right**: Always standardize features (zero mean, unit variance) before PCA or when using gradient descent.

**Why**: Features with larger scales dominate gradient and PCA components.

### Pitfall 2: Using Normal Equations for Large $n$

❌ **Wrong**: Use normal equations for $n = 100000$ features.

✅ **Right**: Use gradient descent for $n > 10000$.

**Why**: Matrix inversion is $O(n^3)$—too slow for large $n$.

### Pitfall 3: Wrong Learning Rate

❌ **Wrong**: Use same learning rate for all problems.

✅ **Right**: Tune learning rate, use learning rate schedules, or use adaptive optimizers (Adam).

### Pitfall 4: PCA Before Train-Test Split

❌ **Wrong**:
```python
X_pca = pca.fit_transform(X)  # Fit on all data
X_train, X_test = train_test_split(X_pca)
```

✅ **Right**:
```python
X_train, X_test = train_test_split(X)
pca.fit(X_train)  # Fit only on training data
X_train_pca = pca.transform(X_train)
X_test_pca = pca.transform(X_test)
```

**Why**: PCA must not see test data (data leakage).

### Pitfall 5: Ignoring Convergence

❌ **Wrong**: Run fixed 1000 iterations without checking convergence.

✅ **Right**: Monitor cost function and stop when converged.

---

## Self-Assessment and Active Recall

### Concept Check Questions

1. **Explain**: Why is linear regression a convex optimization problem?

2. **Compare**: When should you use gradient descent vs normal equations?

3. **Derive**: Show that gradient of MSE is $\nabla J = \frac{1}{m}X^T(X\theta - y)$.

4. **Intuition**: Why does PCA use eigenvectors of covariance matrix?

5. **True/False**: Increasing regularization parameter λ always improves test performance. Explain.

### Coding Challenges

1. **Gradient Descent**: Implement gradient descent with momentum:
   $$v_{t+1} = \beta v_t - \alpha \nabla J(\theta_t)$$
   $$\theta_{t+1} = \theta_t + v_{t+1}$$

2. **Ridge Regression**: Implement ridge regression gradient descent with L2 penalty.

3. **PCA Reconstruction**: Implement inverse PCA transform and visualize reconstruction error vs number of components.

4. **Learning Rate Schedule**: Implement exponential decay: $\alpha_t = \alpha_0 e^{-kt}$

5. **Compare Optimizers**: Compare SGD, SGD with momentum, and Adam on same problem.

### Practice Problems

#### Basic Level

1. Find optimal parameters for $y = \theta_0 + \theta_1 x$ given data: $(1, 3), (2, 5), (3, 7)$ using:
   - (a) Normal equations
   - (b) Gradient descent (show 5 iterations)

2. Compute first principal component for data:
   $$X = \begin{bmatrix} 1 & 2 \\ 2 & 4 \\ 3 & 6 \end{bmatrix}$$

#### Intermediate Level

3. Prove that adding L2 regularization makes $(X^TX + \lambda I)$ invertible even if $X^TX$ is singular.

4. Derive gradient of ridge regression cost function.

5. Show that PCA minimizes reconstruction error.

#### Advanced Level

6. Implement Adam optimizer (adaptive learning rate + momentum).

7. Prove that for convex $J(\theta)$, gradient descent with small enough $\alpha$ converges to global minimum.

8. Derive relationship between PCA and SVD (singular value decomposition).

---

## Quick Reference Summary

### Key Algorithms

**Gradient Descent**:
$$\theta := \theta - \alpha \nabla J(\theta)$$

**Normal Equations**:
$$\theta = (X^TX)^{-1}X^Ty$$

**Ridge Regression**:
$$\theta = (X^TX + \lambda I)^{-1}X^Ty$$

**PCA**:
1. Center data: $\tilde{X} = X - \mu$
2. Covariance: $\Sigma = \frac{1}{m}\tilde{X}^T\tilde{X}$
3. Eigendecomposition: $\Sigma v = \lambda v$
4. Project: $Z = \tilde{X}U_k$

### Code Templates

```python
# Gradient descent
def gradient_descent(X, y, alpha=0.01, iterations=1000):
    theta = np.zeros(X.shape[1])
    for _ in range(iterations):
        gradient = (1/len(X)) * X.T @ (X @ theta - y)
        theta -= alpha * gradient
    return theta

# Normal equations
def normal_equations(X, y):
    return np.linalg.inv(X.T @ X) @ X.T @ y

# Ridge regression
def ridge(X, y, lambda_param=1.0):
    n = X.shape[1]
    I = np.eye(n)
    return np.linalg.inv(X.T @ X + lambda_param * I) @ X.T @ y

# PCA
from sklearn.decomposition import PCA
pca = PCA(n_components=k)
X_reduced = pca.fit_transform(X)
```

### Top 3 Things to Remember

1. **ML = Optimization**: Every model training is minimizing a cost function using calculus
2. **Choose Wisely**: Normal equations for small $n$, gradient descent for large $n$ or non-linear
3. **PCA = Maximum Variance**: Principal components are eigenvectors of covariance matrix

---

## Further Resources

### Documentation
- scikit-learn: Linear Models, PCA, Preprocessing
- NumPy: Linear algebra (`linalg` module)
- SciPy: Optimization (`scipy.optimize`)

### Papers & Books
- Goodfellow et al., "Deep Learning" - Chapter 4 (Numerical Computation) and 5 (Machine Learning Basics)
- Murphy, "Machine Learning: A Probabilistic Perspective"
- Bishop, "Pattern Recognition and Machine Learning"

### Courses
- Andrew Ng's Machine Learning (Coursera) - Week 1-2
- Stanford CS229 - Linear Algebra Review and Probability Review

### Practice
- Kaggle: House Prices, Titanic (regression/classification)
- UCI ML Repository: Try various datasets with PCA
- Implement optimizers from scratch

### Review Schedule
- **After 1 day**: Re-derive gradient of MSE
- **After 3 days**: Implement gradient descent on new dataset
- **After 1 week**: Code PCA from scratch and test
- **After 2 weeks**: Build complete ML pipeline with regularization

---

**Related Notes**:
- Previous: [week-09-optimization-basics.md](week-09-optimization-basics.md)
- Next: [week-11-advanced-topics.md](week-11-advanced-topics.md)
- Foundations: Eigenvalues ([week-05-eigenvalues-eigenvectors.md](week-05-eigenvalues-eigenvectors.md)), Partial derivatives ([week-08-rank-nullity.md](week-08-rank-nullity.md))

**Last Updated**: 2025-11-22
**Status**: Complete and comprehensive ✅
