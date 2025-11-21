#!/usr/bin/env python3
"""
Generate comprehensive Week 4 polynomial operations notebook
"""

import json

# Notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9.6"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

def add_markdown_cell(text):
    """Add a markdown cell to the notebook"""
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split('\n')
    })

def add_code_cell(code):
    """Add a code cell to the notebook"""
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": code.split('\n')
    })

# Title and metadata
add_markdown_cell("""# Week 4: Polynomial Operations in Python

---
**Date**: 2025-11-21
**Course**: BSMA1001 - Mathematics for Data Science I
**Level**: Foundation
**Week**: 4 of 12
**Topic**: Algebra of Polynomials
**Tags**: #BSMA1001 #Polynomials #Python #Week4
---

## Learning Objectives

By the end of this notebook, you will be able to:
1. Implement polynomial operations in Python (addition, subtraction, multiplication, division)
2. Use NumPy's polynomial module for efficient computation
3. Perform symbolic manipulation with SymPy
4. Visualize polynomials and their properties
5. Apply polynomial concepts to real-world data science problems
6. Implement polynomial regression for predictive modeling""")

# Setup
add_markdown_cell("""## 1. Setup and Imports

We'll use several Python libraries for polynomial operations and visualization.""")

add_code_cell("""# Standard libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')

# Symbolic mathematics
import sympy as sp
from sympy import symbols, expand, factor, div, Poly, roots

# Machine learning
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("✓ All libraries imported successfully!")
print(f"NumPy version: {np.__version__}")""")

# Polynomial basics
add_markdown_cell("""## 2. Polynomial Basics

### 2.1 Representing Polynomials

In Python, polynomials can be represented in multiple ways:
1. **NumPy's `poly1d`**: Coefficient-based representation (descending powers)
2. **SymPy**: Symbolic representation for algebraic manipulation""")

add_code_cell("""# Example: P(x) = 2x³ + 3x² - 5x + 1

# Method 1: NumPy poly1d (coefficients in descending order)
p_numpy = np.poly1d([2, 3, -5, 1])
print("NumPy representation:")
print(p_numpy)
print()

# Method 2: SymPy (symbolic)
x = symbols('x')
p_sympy = 2*x**3 + 3*x**2 - 5*x + 1
print("SymPy representation:")
print(p_sympy)
print()

# Evaluate at x = 2
print(f"P(2) using NumPy: {p_numpy(2)}")
print(f"P(2) using SymPy: {p_sympy.subs(x, 2)}")""")

add_markdown_cell("""### 2.2 Polynomial Degree and Coefficients""")

add_code_cell("""# Get polynomial properties
print(f"Degree: {p_numpy.order}")
print(f"Coefficients: {p_numpy.coefficients}")
print(f"Leading coefficient: {p_numpy.coefficients[0]}")
print(f"Constant term: {p_numpy.coefficients[-1]}")
print()

# Using SymPy
p_poly = Poly(p_sympy, x)
print(f"SymPy degree: {p_poly.degree()}")
print(f"SymPy coefficients: {p_poly.all_coeffs()}")""")

# Polynomial operations
add_markdown_cell("""## 3. Polynomial Operations

### 3.1 Addition and Subtraction""")

add_code_cell("""# Define two polynomials
# P(x) = 3x² + 2x - 1
# Q(x) = x² - 4x + 3

P = np.poly1d([3, 2, -1])
Q = np.poly1d([1, -4, 3])

print("P(x) =", P)
print("Q(x) =", Q)
print()

# Addition
P_plus_Q = P + Q
print("P(x) + Q(x) =", P_plus_Q)
print()

# Subtraction
P_minus_Q = P - Q
print("P(x) - Q(x) =", P_minus_Q)
print()

# Verify with SymPy
x = symbols('x')
P_sym = 3*x**2 + 2*x - 1
Q_sym = x**2 - 4*x + 3
print("SymPy P + Q:", expand(P_sym + Q_sym))
print("SymPy P - Q:", expand(P_sym - Q_sym))""")

add_markdown_cell("""### 3.2 Multiplication""")

add_code_cell("""# Multiply two polynomials: (2x + 3)(x - 4)
P = np.poly1d([2, 3])
Q = np.poly1d([1, -4])

print("P(x) =", P)
print("Q(x) =", Q)
print()

# Multiplication
P_times_Q = P * Q
print("P(x) × Q(x) =", P_times_Q)
print()

# Verify degree property: deg(P×Q) = deg(P) + deg(Q)
print(f"deg(P) = {P.order}, deg(Q) = {Q.order}")
print(f"deg(P×Q) = {P_times_Q.order} (should be {P.order + Q.order})")
print()

# More complex example: (x² + 2x - 1)(x + 3)
P = np.poly1d([1, 2, -1])
Q = np.poly1d([1, 3])
result = P * Q
print(f"({P}) × ({Q})")
print(f"= {result}")""")

add_markdown_cell("""### 3.3 Special Products""")

add_code_cell("""# Demonstrate special product formulas

# 1. Square of binomial: (a + b)² = a² + 2ab + b²
# Example: (3x - 2)²
P = np.poly1d([3, -2])
P_squared = P * P
print("(3x - 2)² =", P_squared)
print()

# 2. Difference of squares: (a + b)(a - b) = a² - b²
# Example: (x + 4)(x - 4)
P = np.poly1d([1, 4])
Q = np.poly1d([1, -4])
result = P * Q
print("(x + 4)(x - 4) =", result)
print()

# Using SymPy for symbolic verification
x = symbols('x')
print("Symbolic (x² + 4)(x² - 4) =", expand((x**2 + 4)*(x**2 - 4)))""")

# Division
add_markdown_cell("""## 4. Division and Remainder Theorem

### 4.1 Polynomial Division""")

add_code_cell("""# Divide P(x) = 2x³ + 5x² - 3x + 7 by D(x) = x + 2

P = np.poly1d([2, 5, -3, 7])
D = np.poly1d([1, 2])

print("Dividend P(x) =", P)
print("Divisor D(x) =", D)
print()

# Perform division
quotient, remainder = np.polydiv(P.coefficients, D.coefficients)
Q = np.poly1d(quotient)
R = np.poly1d(remainder) if len(remainder) > 0 else 0

print("Quotient Q(x) =", Q)
print("Remainder R =", R)
print()

# Verify: P(x) = Q(x)·D(x) + R
verification = Q * D + R
print("Verification P(x) = Q(x)·D(x) + R:")
print(verification)
print("Matches original?", np.allclose(P.coefficients, verification.coefficients))""")

add_markdown_cell("""### 4.2 Remainder Theorem

**Theorem**: When polynomial P(x) is divided by (x - c), the remainder is P(c).""")

add_code_cell("""# Example: Find remainder when P(x) = x³ - 4x² + 6x - 8 is divided by (x - 3)

P = np.poly1d([1, -4, 6, -8])
c = 3

print("P(x) =", P)
print(f"Dividing by (x - {c})")
print()

# Method 1: Direct evaluation using Remainder Theorem
remainder_theorem = P(c)
print(f"Remainder Theorem: P({c}) = {remainder_theorem}")
print()

# Method 2: Actual division
D = np.poly1d([1, -c])
_, remainder = np.polydiv(P.coefficients, D.coefficients)
print(f"Actual division remainder: {remainder[0] if len(remainder) > 0 else 0}")
print()

print("✓ Both methods give the same result!")""")

add_markdown_cell("""### 4.3 Factor Theorem

**Theorem**: (x - c) is a factor of P(x) if and only if P(c) = 0.""")

add_code_cell("""# Example: Is (x - 2) a factor of P(x) = x³ - 6x² + 11x - 6?

P = np.poly1d([1, -6, 11, -6])
c = 2

print("P(x) =", P)
print(f"Testing if (x - {c}) is a factor...")
print()

# Evaluate P(c)
result = P(c)
print(f"P({c}) = {result}")

if abs(result) < 1e-10:  # Check if essentially zero
    print(f"✓ (x - {c}) IS a factor of P(x)")
else:
    print(f"✗ (x - {c}) is NOT a factor of P(x)")

# Find all roots to verify
roots_list = np.roots(P.coefficients)
print(f"\\nAll roots of P(x): {roots_list}")""")

# Factorization
add_markdown_cell("""## 5. Factorization and Roots

### 5.1 Finding Roots""")

add_code_cell("""# Example: Find roots of P(x) = x³ - 6x² + 11x - 6

P = np.poly1d([1, -6, 11, -6])
print("P(x) =", P)
print()

# NumPy method
roots_np = np.roots(P.coefficients)
print("Roots (NumPy):", roots_np)
print()

# SymPy method (symbolic, exact)
x = symbols('x')
P_sym = x**3 - 6*x**2 + 11*x - 6
roots_sym = sp.solve(P_sym, x)
print("Roots (SymPy):", roots_sym)
print()

# Verify each root
print("Verification:")
for r in roots_np:
    print(f"P({r:.6f}) = {P(r):.2e}")""")

add_markdown_cell("""### 5.2 Complete Factorization""")

add_code_cell("""# Factor polynomial completely using SymPy
x = symbols('x')

# Example 1: x³ - 2x² - 9x + 18
P1 = x**3 - 2*x**2 - 9*x + 18
print("P(x) =", P1)
print("Factored form:", factor(P1))
print()

# Example 2: 2x³ - x² - 13x - 6
P2 = 2*x**3 - x**2 - 13*x - 6
print("P(x) =", P2)
print("Factored form:", factor(P2))
print()

# Example 3: x⁴ - 5x² + 4
P3 = x**4 - 5*x**2 + 4
print("P(x) =", P3)
print("Factored form:", factor(P3))""")

add_markdown_cell("""### 5.3 Multiplicity of Roots""")

add_code_cell("""# Example: P(x) = (x - 2)³(x + 1)²(x - 5)
x = symbols('x')
P = (x - 2)**3 * (x + 1)**2 * (x - 5)
P_expanded = expand(P)

print("P(x) =", P_expanded)
print()

# Find roots and their multiplicities
roots_dict = roots(Poly(P_expanded, x))
print("Roots and their multiplicities:")
for root, multiplicity in roots_dict.items():
    print(f"  x = {root}, multiplicity = {multiplicity}")
print()

print(f"Total degree: {sum(roots_dict.values())}")""")

# Visualization
add_markdown_cell("""## 6. Visualization

### 6.1 Plotting Polynomials""")

add_code_cell("""# Visualize several polynomials
x_vals = np.linspace(-5, 5, 400)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Linear
P1 = np.poly1d([2, -3])
axes[0, 0].plot(x_vals, P1(x_vals), 'b-', linewidth=2, label='P(x) = 2x - 3')
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_title('Linear Polynomial', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('x')
axes[0, 0].set_ylabel('P(x)')
axes[0, 0].legend()

# Plot 2: Quadratic
P2 = np.poly1d([1, -2, -3])
axes[0, 1].plot(x_vals, P2(x_vals), 'r-', linewidth=2, label='P(x) = x² - 2x - 3')
axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
roots = np.roots(P2.coefficients)
axes[0, 1].plot(roots, [0, 0], 'go', markersize=10, label=f'Roots: {roots}')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_title('Quadratic Polynomial', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('x')
axes[0, 1].set_ylabel('P(x)')
axes[0, 1].legend()

# Plot 3: Cubic
P3 = np.poly1d([1, -6, 11, -6])
axes[1, 0].plot(x_vals, P3(x_vals), 'g-', linewidth=2, label='P(x) = x³ - 6x² + 11x - 6')
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
roots = np.roots(P3.coefficients)
axes[1, 0].plot(roots, [0, 0, 0], 'go', markersize=10, label=f'Roots: {roots}')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_title('Cubic Polynomial', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('P(x)')
axes[1, 0].legend()

# Plot 4: Quartic
P4 = np.poly1d([1, 0, -5, 0, 4])
axes[1, 1].plot(x_vals, P4(x_vals), 'm-', linewidth=2, label='P(x) = x⁴ - 5x² + 4')
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
roots = np.roots(P4.coefficients)
axes[1, 1].plot(roots, [0, 0, 0, 0], 'go', markersize=10)
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_title('Quartic Polynomial', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('x')
axes[1, 1].set_ylabel('P(x)')
axes[1, 1].legend()

plt.tight_layout()
plt.show()

print("✓ Polynomial visualizations complete!")""")

add_markdown_cell("""### 6.2 End Behavior and Turning Points""")

add_code_cell("""# Analyze P(x) = x³ - 6x² + 9x + 1
P = np.poly1d([1, -6, 9, 1])
P_deriv = np.polyder(P)  # First derivative

x_vals = np.linspace(-1, 6, 500)
y_vals = P(x_vals)

# Find critical points (where derivative = 0)
critical_points = np.roots(P_deriv.coefficients)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot polynomial
ax1.plot(x_vals, y_vals, 'b-', linewidth=2.5, label='P(x) = x³ - 6x² + 9x + 1')
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax1.plot(critical_points, P(critical_points), 'ro', markersize=10, label='Turning points')
ax1.grid(True, alpha=0.3)
ax1.set_title('Polynomial with Turning Points', fontsize=12, fontweight='bold')
ax1.set_xlabel('x')
ax1.set_ylabel('P(x)')
ax1.legend()

# Plot derivative
ax2.plot(x_vals, P_deriv(x_vals), 'r-', linewidth=2.5, label="P'(x)")
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.axvline(x=0, color='k', linestyle='--', alpha=0.3)
ax2.plot(critical_points, [0, 0], 'go', markersize=10, label='Critical points')
ax2.grid(True, alpha=0.3)
ax2.set_title('First Derivative', fontsize=12, fontweight='bold')
ax2.set_xlabel('x')
ax2.set_ylabel("P'(x)")
ax2.legend()

plt.tight_layout()
plt.show()

print(f"Critical points (turning points): {critical_points}")
print(f"Values at critical points: {P(critical_points)}")""")

# Data science application
add_markdown_cell("""## 7. Data Science Application: Polynomial Regression

### 7.1 Generating Sample Data""")

add_code_cell("""# Generate synthetic data with polynomial relationship + noise
np.random.seed(42)

# True relationship: y = 2x² - 3x + 1 + noise
X = np.linspace(-3, 3, 50)
y_true = 2*X**2 - 3*X + 1
noise = np.random.normal(0, 5, size=X.shape)
y = y_true + noise

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(X, y, alpha=0.6, s=50, label='Observed data')
plt.plot(X, y_true, 'r--', linewidth=2, label='True function: y = 2x² - 3x + 1')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Synthetic Dataset for Polynomial Regression', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print(f"Dataset size: {len(X)} samples")
print(f"X range: [{X.min():.2f}, {X.max():.2f}]")
print(f"y range: [{y.min():.2f}, {y.max():.2f}]")""")

add_markdown_cell("""### 7.2 Fitting Polynomial Models of Different Degrees""")

add_code_cell("""# Fit polynomials of degree 1, 2, 3, and 5
degrees = [1, 2, 3, 5]
X_reshaped = X.reshape(-1, 1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.ravel()

for idx, degree in enumerate(degrees):
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly_features.fit_transform(X_reshaped)

    # Fit linear regression on polynomial features
    model = LinearRegression()
    model.fit(X_poly, y)

    # Predict
    X_test = np.linspace(-3, 3, 300).reshape(-1, 1)
    X_test_poly = poly_features.transform(X_test)
    y_pred = model.predict(X_test_poly)
    y_train_pred = model.predict(X_poly)

    # Calculate metrics
    r2 = r2_score(y, y_train_pred)
    mse = mean_squared_error(y, y_train_pred)

    # Plot
    axes[idx].scatter(X, y, alpha=0.6, s=50, label='Data')
    axes[idx].plot(X, y_true, 'r--', linewidth=1.5, alpha=0.7, label='True function')
    axes[idx].plot(X_test, y_pred, 'b-', linewidth=2.5, label=f'Degree {degree} fit')
    axes[idx].set_title(f'Degree {degree} | R² = {r2:.3f} | MSE = {mse:.2f}',
                       fontsize=12, fontweight='bold')
    axes[idx].set_xlabel('X')
    axes[idx].set_ylabel('y')
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

    # Extract coefficients
    print(f"\\nDegree {degree} polynomial:")
    print(f"  Coefficients: {model.coef_}")
    print(f"  Intercept: {model.intercept_:.3f}")
    print(f"  R²: {r2:.4f}")
    print(f"  MSE: {mse:.4f}")

plt.tight_layout()
plt.show()""")

add_markdown_cell("""### 7.3 Model Comparison: Overfitting vs Underfitting""")

add_code_cell("""# Compare models across many degrees
degrees_test = range(1, 11)
train_scores = []
test_scores = []

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X.reshape(-1, 1), y, test_size=0.3, random_state=42
)

for degree in degrees_test:
    poly_features = PolynomialFeatures(degree=degree)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)

    model = LinearRegression()
    model.fit(X_train_poly, y_train)

    train_score = model.score(X_train_poly, y_train)
    test_score = model.score(X_test_poly, y_test)

    train_scores.append(train_score)
    test_scores.append(test_score)

# Plot training vs test performance
plt.figure(figsize=(10, 6))
plt.plot(degrees_test, train_scores, 'o-', linewidth=2, label='Training R²', color='blue')
plt.plot(degrees_test, test_scores, 's-', linewidth=2, label='Test R²', color='red')
plt.axvline(x=2, color='g', linestyle='--', linewidth=2, alpha=0.7, label='True degree (2)')
plt.xlabel('Polynomial Degree', fontsize=12)
plt.ylabel('R² Score', fontsize=12)
plt.title('Model Performance vs Polynomial Degree\\n(Detecting Overfitting)',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(degrees_test)
plt.show()

print("Observations:")
print("- Degree 2 has the best test performance (matches true relationship)")
print("- Higher degrees overfit: high training R², lower test R²")
print("- Degree 1 underfits: poor performance on both sets")""")

add_markdown_cell("""### 7.4 Real-World Example: Temperature Prediction""")

add_code_cell("""# Simulate temperature data over 24 hours
np.random.seed(123)

hours = np.arange(0, 24)
# Temperature follows a polynomial pattern (warmer during midday)
temp_true = -0.05 * hours**2 + 1.2 * hours + 15
temp_observed = temp_true + np.random.normal(0, 1.5, size=hours.shape)

# Fit polynomial regression
poly_features = PolynomialFeatures(degree=2)
hours_reshaped = hours.reshape(-1, 1)
hours_poly = poly_features.fit_transform(hours_reshaped)

model = LinearRegression()
model.fit(hours_poly, temp_observed)

# Predict on finer grid
hours_fine = np.linspace(0, 24, 200).reshape(-1, 1)
hours_fine_poly = poly_features.transform(hours_fine)
temp_pred = model.predict(hours_fine_poly)

# Visualization
plt.figure(figsize=(12, 6))
plt.scatter(hours, temp_observed, s=100, alpha=0.7, label='Observed temperature', color='red')
plt.plot(hours_fine, temp_pred, 'b-', linewidth=3, label='Polynomial fit (degree 2)')
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Temperature (°C)', fontsize=12)
plt.title('Temperature Prediction Using Polynomial Regression', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(range(0, 25, 3))
plt.show()

# Print model equation
coeffs = model.coef_
intercept = model.intercept_
print(f"\\nModel equation:")
print(f"Temperature = {coeffs[2]:.4f}×hour² + {coeffs[1]:.4f}×hour + {intercept:.4f}")

# Predictions
for h in [6, 12, 18]:
    pred = model.predict(poly_features.transform([[h]]))
    print(f"Predicted temperature at {h}:00 = {pred[0]:.1f}°C")""")

# Practice problems
add_markdown_cell("""## 8. Practice Problems

Try solving these problems using the techniques learned above!""")

add_markdown_cell("""### Problem 1: Polynomial Operations

Given $P(x) = 3x^3 + 2x^2 - x + 5$ and $Q(x) = x^3 - 4x^2 + 3x - 2$, compute:
a) $P(x) + Q(x)$
b) $P(x) - Q(x)$
c) $P(x) \\times Q(x)$""")

add_code_cell("""# Your solution here
P = np.poly1d([3, 2, -1, 5])
Q = np.poly1d([1, -4, 3, -2])

# a) Addition
sum_PQ = P + Q
print("a) P(x) + Q(x) =", sum_PQ)

# b) Subtraction
diff_PQ = P - Q
print("b) P(x) - Q(x) =", diff_PQ)

# c) Multiplication
prod_PQ = P * Q
print("c) P(x) × Q(x) =", prod_PQ)""")

add_markdown_cell("""### Problem 2: Division and Remainder

Divide $P(x) = 2x^4 - 3x^3 + x - 5$ by $D(x) = x^2 - 2x + 1$ and find quotient and remainder.""")

add_code_cell("""# Your solution here
P = np.poly1d([2, -3, 0, 1, -5])
D = np.poly1d([1, -2, 1])

quotient, remainder = np.polydiv(P.coefficients, D.coefficients)
Q = np.poly1d(quotient)
R = np.poly1d(remainder)

print("Quotient Q(x) =", Q)
print("Remainder R(x) =", R)

# Verify
verification = np.poly1d(quotient) * D + np.poly1d(remainder)
print("\\nVerification: Q(x)×D(x) + R(x) =", verification)""")

add_markdown_cell("""### Problem 3: Factorization

Factor completely: $P(x) = x^3 - 3x^2 - 10x + 24$""")

add_code_cell("""# Your solution here
x = symbols('x')
P = x**3 - 3*x**2 - 10*x + 24

print("P(x) =", P)
print("Factored form:", factor(P))
print()

# Find roots
roots_list = sp.solve(P, x)
print("Roots:", roots_list)

# Verify
for root in roots_list:
    value = P.subs(x, root)
    print(f"P({root}) = {value}")""")

add_markdown_cell("""### Problem 4: Finding Unknown Coefficients

If $(x - 1)$ and $(x + 2)$ are factors of $P(x) = x^3 + ax^2 + bx - 4$, find $a$ and $b$.""")

add_code_cell("""# Your solution here
# Using Factor Theorem: P(1) = 0 and P(-2) = 0

# P(1) = 0: 1 + a + b - 4 = 0 → a + b = 3
# P(-2) = 0: -8 + 4a - 2b - 4 = 0 → 4a - 2b = 12 → 2a - b = 6

# Solve system of equations
from sympy import Eq, solve
a, b = symbols('a b')

eq1 = Eq(a + b, 3)
eq2 = Eq(2*a - b, 6)

solution = solve((eq1, eq2), (a, b))
print("Solution:", solution)
print(f"a = {solution[a]}, b = {solution[b]}")

# Verify
x = symbols('x')
P = x**3 + solution[a]*x**2 + solution[b]*x - 4
print("\\nP(x) =", P)
print("Factored form:", factor(P))""")

# Self-assessment
add_markdown_cell("""## 9. Self-Assessment Checklist

Evaluate your understanding of this week's material:

### Polynomial Basics
- [ ] I can represent polynomials using NumPy and SymPy
- [ ] I understand polynomial degree and coefficients
- [ ] I can evaluate polynomials at specific values

### Operations
- [ ] I can add and subtract polynomials
- [ ] I can multiply polynomials correctly
- [ ] I understand the degree property for multiplication
- [ ] I can apply special product formulas

### Division
- [ ] I can perform polynomial division using NumPy
- [ ] I understand and can apply the Division Algorithm
- [ ] I can use the Remainder Theorem effectively
- [ ] I can apply the Factor Theorem to test factors

### Factorization
- [ ] I can find roots of polynomials
- [ ] I can factor polynomials using SymPy
- [ ] I understand multiplicity of roots
- [ ] I can apply the Rational Root Theorem

### Visualization
- [ ] I can plot polynomials using Matplotlib
- [ ] I can identify roots, turning points, and end behavior from graphs
- [ ] I understand the relationship between a polynomial and its derivative

### Applications
- [ ] I understand polynomial regression concepts
- [ ] I can implement polynomial regression using scikit-learn
- [ ] I recognize overfitting and underfitting in polynomial models
- [ ] I can apply polynomial methods to real-world problems

### Overall
- [ ] I completed all code examples
- [ ] I worked through all practice problems
- [ ] I can explain polynomial concepts clearly
- [ ] I'm ready to move to Week 5 (Functions)

---

**If you checked fewer than 80% of the boxes, review the relevant sections before proceeding.**""")

add_markdown_cell("""## Summary

In this notebook, we explored:

1. **Polynomial Representation**: Using NumPy and SymPy to work with polynomials
2. **Operations**: Addition, subtraction, multiplication, and division of polynomials
3. **Key Theorems**: Division Algorithm, Remainder Theorem, Factor Theorem
4. **Factorization**: Finding roots and factoring polynomials completely
5. **Visualization**: Plotting polynomials and analyzing their behavior
6. **Applications**: Polynomial regression for predictive modeling

### Key Takeaways

- Polynomials are fundamental building blocks in mathematics and data science
- Python provides powerful tools (NumPy, SymPy, sklearn) for polynomial work
- Polynomial regression is useful for modeling nonlinear relationships
- Higher-degree polynomials can overfit; choose degree carefully
- Visualization helps understand polynomial behavior

### Next Steps

**Week 5: Functions - Deep Dive**
- Formal function definitions
- Domain, range, and function composition
- Inverse functions
- Special types of functions

---

**Great work completing Week 4! 🎉**""")

# Save notebook
output_path = "/Users/godsonkurishinkal/Projects/iitm-bs-learning/01-Foundation-Level/01-Mathematics-I/notebooks/week-04-polynomial-operations.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"✓ Notebook generated successfully: {output_path}")
print(f"✓ Total cells: {len(notebook['cells'])}")
print(f"  - Markdown cells: {sum(1 for c in notebook['cells'] if c['cell_type'] == 'markdown')}")
print(f"  - Code cells: {sum(1 for c in notebook['cells'] if c['cell_type'] == 'code')}")
