#!/usr/bin/env python3
"""
Generate comprehensive Week 5 functions notebook
"""

import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.9.6"}
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

def add_md(text):
    notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": text.split('\n')})

def add_code(code):
    notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code.split('\n')})

# Title
add_md("""# Week 5: Functions in Python

---
**Date**: 2025-11-21
**Course**: BSMA1001 - Mathematics for Data Science I
**Level**: Foundation
**Week**: 5 of 12
**Topic**: Functions - Definition, Properties, and Applications
**Tags**: #BSMA1001 #Functions #Python #Week5
---

## Learning Objectives

1. Implement and analyze functions in Python
2. Visualize domain, range, and function behavior
3. Perform function composition and find inverses
4. Classify functions (one-to-one, onto, bijective)
5. Apply function transformations
6. Implement data science applications (activation functions, transforms)""")

# Setup
add_md("""## 1. Setup and Imports""")

add_code("""import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sympy as sp
from sympy import symbols, solve, simplify, lambdify
from scipy import optimize
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

print("✓ All libraries imported successfully!")""")

# Function basics
add_md("""## 2. Defining Functions in Python

### 2.1 Basic Function Definition""")

add_code("""# Define a simple function
def f(x):
    \"\"\"Square function: f(x) = x^2\"\"\"
    return x**2

# Test the function
test_values = [-2, 0, 1, 3]
print("f(x) = x²")
for val in test_values:
    print(f"f({val}) = {f(val)}")
print()

# Using lambda functions (anonymous functions)
g = lambda x: 2*x + 3
print("g(x) = 2x + 3")
for val in test_values:
    print(f"g({val}) = {g(val)}")""")

add_md("""### 2.2 Vectorized Functions with NumPy""")

add_code("""# NumPy allows element-wise operations on arrays
def f_vectorized(x):
    \"\"\"Vectorized square function\"\"\"
    return np.array(x)**2

# Test with array
x_array = np.array([-2, -1, 0, 1, 2, 3])
y_array = f_vectorized(x_array)

print("Input:", x_array)
print("Output:", y_array)
print()

# Visualize
plt.figure(figsize=(10, 5))
plt.plot(x_array, y_array, 'bo-', markersize=10, linewidth=2, label='f(x) = x²')
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('Discrete Function Values', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()""")

# Domain and Range
add_md("""## 3. Domain and Range Analysis

### 3.1 Finding Domains""")

add_code("""# Function 1: f(x) = 1/(x-3)
def f1(x):
    return 1 / (x - 3)

print("f(x) = 1/(x-3)")
print("Domain: All real numbers except x = 3")
print("In interval notation: (-∞, 3) ∪ (3, ∞)")
print()

# Visualize with discontinuity
x_left = np.linspace(-2, 2.9, 100)
x_right = np.linspace(3.1, 8, 100)

plt.figure(figsize=(10, 6))
plt.plot(x_left, f1(x_left), 'b-', linewidth=2, label='f(x) = 1/(x-3)')
plt.plot(x_right, f1(x_right), 'b-', linewidth=2)
plt.axvline(x=3, color='r', linestyle='--', linewidth=2, alpha=0.7, label='x = 3 (not in domain)')
plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
plt.ylim(-5, 5)
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('Function with Restricted Domain', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()

# Function 2: g(x) = √(x-2)
def g(x):
    return np.sqrt(x - 2)

print("g(x) = √(x-2)")
print("Domain: x ≥ 2")
print("In interval notation: [2, ∞)")""")

add_md("""### 3.2 Determining Range""")

add_code("""# Analyze range for f(x) = x² on different domains

# Case 1: Domain = All reals
x1 = np.linspace(-3, 3, 200)
y1 = x1**2

# Case 2: Domain = [0, ∞)
x2 = np.linspace(0, 3, 150)
y2 = x2**2

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Full domain
ax1.plot(x1, y1, 'b-', linewidth=2.5)
ax1.fill_between(x1, 0, y1, alpha=0.2)
ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('f(x)', fontsize=12)
ax1.set_title('f(x) = x², Domain: ℝ\\nRange: [0, ∞)', fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Plot 2: Restricted domain
ax2.plot(x2, y2, 'r-', linewidth=2.5)
ax2.fill_between(x2, 0, y2, alpha=0.2, color='red')
ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('f(x)', fontsize=12)
ax2.set_title('f(x) = x², Domain: [0, ∞)\\nRange: [0, ∞)', fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Same function, different domains → same range!")
print("This shows domain affects which inputs are allowed, not necessarily the range.")""")

# Function types
add_md("""## 4. Function Types

### 4.1 One-to-One (Injective) Functions""")

add_code("""def is_one_to_one_numerical(func, x_min, x_max, n_samples=1000):
    \"\"\"
    Numerical test for one-to-one property using horizontal line test
    \"\"\"
    x = np.linspace(x_min, x_max, n_samples)
    y = func(x)

    # Check if any y-value appears more than once
    unique_y = len(np.unique(np.round(y, 10)))
    total_y = len(y)

    return unique_y == total_y

# Test functions
f1 = lambda x: 2*x + 3  # One-to-one
f2 = lambda x: x**2      # NOT one-to-one on full domain

print("Testing f(x) = 2x + 3 on [-5, 5]:")
print(f"One-to-one? {is_one_to_one_numerical(f1, -5, 5)}")
print()

print("Testing f(x) = x² on [-5, 5]:")
print(f"One-to-one? {is_one_to_one_numerical(f2, -5, 5)}")
print("(Not one-to-one because f(2) = f(-2) = 4)")
print()

print("Testing f(x) = x² on [0, 5] (restricted domain):")
print(f"One-to-one? {is_one_to_one_numerical(f2, 0, 5)}")

# Visualize horizontal line test
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

x = np.linspace(-3, 3, 200)

# Plot 1: One-to-one function
y1 = 2*x + 3
ax1.plot(x, y1, 'b-', linewidth=2.5, label='f(x) = 2x + 3')
for h in [1, 3, 5]:
    ax1.axhline(y=h, color='r', linestyle='--', alpha=0.5, linewidth=1.5)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('f(x)', fontsize=12)
ax1.set_title('One-to-One Function\\n(Each horizontal line intersects once)',
              fontsize=12, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot 2: NOT one-to-one
y2 = x**2
ax2.plot(x, y2, 'r-', linewidth=2.5, label='f(x) = x²')
for h in [1, 4, 9]:
    ax2.axhline(y=h, color='b', linestyle='--', alpha=0.5, linewidth=1.5)
    if h > 0:
        intersections = [-np.sqrt(h), np.sqrt(h)]
        ax2.plot(intersections, [h, h], 'go', markersize=10)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('f(x)', fontsize=12)
ax2.set_title('NOT One-to-One\\n(Horizontal lines can intersect twice)',
              fontsize=12, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.show()""")

add_md("""### 4.2 Even and Odd Functions""")

add_code("""def test_even_odd(func, x_range=(-5, 5), n_points=100):
    \"\"\"
    Test if a function is even, odd, or neither
    \"\"\"
    x = np.linspace(x_range[0], x_range[1], n_points)
    x = x[x != 0]  # Exclude zero

    fx = func(x)
    f_neg_x = func(-x)

    is_even = np.allclose(fx, f_neg_x, rtol=1e-10)
    is_odd = np.allclose(fx, -f_neg_x, rtol=1e-10)

    if is_even:
        return "EVEN"
    elif is_odd:
        return "ODD"
    else:
        return "NEITHER"

# Test various functions
functions = [
    (lambda x: x**2, "f(x) = x²"),
    (lambda x: x**3, "g(x) = x³"),
    (lambda x: x**4 - 2*x**2, "h(x) = x⁴ - 2x²"),
    (lambda x: x**3 - x, "p(x) = x³ - x"),
    (lambda x: x**2 + x, "q(x) = x² + x")
]

print("Testing Even/Odd Properties:")
print("="*50)
for func, name in functions:
    result = test_even_odd(func)
    print(f"{name:20} → {result}")

# Visualize even and odd functions
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
x = np.linspace(-3, 3, 200)

# Even: f(x) = x²
axes[0, 0].plot(x, x**2, 'b-', linewidth=2.5, label='f(x) = x²')
axes[0, 0].plot(x, x**2, 'r--', linewidth=1, alpha=0.5)
axes[0, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].set_title('EVEN: f(-x) = f(x)\\nY-axis Symmetry', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].legend()

# Odd: f(x) = x³
axes[0, 1].plot(x, x**3, 'r-', linewidth=2.5, label='f(x) = x³')
axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].set_title('ODD: f(-x) = -f(x)\\nOrigin Symmetry', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].legend()

# Even: f(x) = cos(x)
axes[1, 0].plot(x, np.cos(x), 'g-', linewidth=2.5, label='f(x) = cos(x)')
axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].set_title('EVEN: f(x) = cos(x)', fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].legend()

# Odd: f(x) = sin(x)
axes[1, 1].plot(x, np.sin(x), 'm-', linewidth=2.5, label='f(x) = sin(x)')
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].set_title('ODD: f(x) = sin(x)', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].legend()

plt.tight_layout()
plt.show()""")

# Composition
add_md("""## 5. Function Composition

### 5.1 Basic Composition""")

add_code("""# Define two functions
def f(x):
    return x**2

def g(x):
    return x + 1

# Compute compositions
def fog(x):
    \"\"\"(f ∘ g)(x) = f(g(x))\"\"\"
    return f(g(x))

def gof(x):
    \"\"\"(g ∘ f)(x) = g(f(x))\"\"\"
    return g(f(x))

# Test values
test_x = np.array([0, 1, 2, 3])

print("Given: f(x) = x², g(x) = x + 1")
print()
print("x\\t(f∘g)(x)\\t(g∘f)(x)")
print("-" * 35)
for x in test_x:
    print(f"{x}\\t{fog(x)}\\t\\t{gof(x)}")

print()
print("Observation: (f∘g)(x) ≠ (g∘f)(x) in general!")
print("Composition is NOT commutative.")

# Visualize
x = np.linspace(-2, 4, 200)

plt.figure(figsize=(12, 6))
plt.plot(x, f(x), 'b-', linewidth=2, label='f(x) = x²', alpha=0.7)
plt.plot(x, g(x), 'r-', linewidth=2, label='g(x) = x + 1', alpha=0.7)
plt.plot(x, fog(x), 'g-', linewidth=2.5, label='(f∘g)(x) = (x+1)²')
plt.plot(x, gof(x), 'm--', linewidth=2.5, label='(g∘f)(x) = x² + 1')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Function Composition Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()""")

add_md("""### 5.2 Three-Function Composition""")

add_code("""# Define three functions
f = lambda x: 2*x
g = lambda x: x + 3
h = lambda x: x**2

# Compute (f ∘ g ∘ h)(x)
def fog_h(x):
    return f(g(h(x)))

# Test associativity: ((f∘g)∘h) = (f∘(g∘h))
fg = lambda x: f(g(x))
gh = lambda x: g(h(x))

x_test = 5
result1 = fg(h(x_test))
result2 = f(gh(x_test))

print(f"Testing associativity at x = {x_test}:")
print(f"((f∘g)∘h)({x_test}) = {result1}")
print(f"(f∘(g∘h))({x_test}) = {result2}")
print(f"Equal? {result1 == result2}")
print()

# Symbolic verification with SymPy
x = symbols('x')
f_sym = 2*x
g_sym = x + 3
h_sym = x**2

composition = f_sym.subs(x, g_sym.subs(x, h_sym))
print(f"Symbolic composition: (f∘g∘h)(x) = {composition}")
print(f"Expanded: {sp.expand(composition)}")""")

# Inverse functions
add_md("""## 6. Inverse Functions

### 6.1 Finding Inverses""")

add_code("""def find_inverse_numerically(func, y_val, x_range=(-10, 10)):
    \"\"\"
    Find x such that f(x) = y_val (numerical inverse)
    \"\"\"
    objective = lambda x: func(x) - y_val
    result = optimize.brentq(objective, x_range[0], x_range[1])
    return result

# Example: f(x) = 2x + 3
def f(x):
    return 2*x + 3

def f_inverse(x):
    \"\"\"Analytical inverse: f⁻¹(x) = (x-3)/2\"\"\"
    return (x - 3) / 2

# Verify inverse property
test_values = [1, 5, 10]

print("Function: f(x) = 2x + 3")
print("Inverse: f⁻¹(x) = (x-3)/2")
print()
print("Verifying f(f⁻¹(x)) = x:")
for val in test_values:
    result = f(f_inverse(val))
    print(f"f(f⁻¹({val})) = {result}")

print()
print("Verifying f⁻¹(f(x)) = x:")
for val in test_values:
    result = f_inverse(f(val))
    print(f"f⁻¹(f({val})) = {result}")

# Visualize function and its inverse
x = np.linspace(-3, 5, 200)
y_f = f(x)
y_f_inv = f_inverse(x)

plt.figure(figsize=(10, 8))
plt.plot(x, y_f, 'b-', linewidth=2.5, label='f(x) = 2x + 3')
plt.plot(x, y_f_inv, 'r-', linewidth=2.5, label='f⁻¹(x) = (x-3)/2')
plt.plot(x, x, 'k--', linewidth=1.5, alpha=0.5, label='y = x (line of symmetry)')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Function and Its Inverse\\n(Reflection across y=x)', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.show()""")

add_md("""### 6.2 Inverse of x² with Restricted Domain""")

add_code("""# f(x) = x² on [0, ∞) has inverse f⁻¹(x) = √x

x_pos = np.linspace(0, 4, 200)
y_square = x_pos**2
y_sqrt = np.sqrt(x_pos)

plt.figure(figsize=(10, 8))
plt.plot(x_pos, y_square, 'b-', linewidth=2.5, label='f(x) = x² (domain: [0, ∞))')
plt.plot(x_pos, y_sqrt, 'r-', linewidth=2.5, label='f⁻¹(x) = √x')
plt.plot(x_pos, x_pos, 'k--', linewidth=1.5, alpha=0.5, label='y = x')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('f(x) = x² and f⁻¹(x) = √x\\n(Restricted to non-negative domain)',
          fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.axis('equal')
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.show()

# Verify inverse property
test_x = np.array([0, 1, 4, 9, 16])
print("Verifying inverse property:")
print("x\\tf(f⁻¹(x))\\tf⁻¹(f(x))")
print("-" * 35)
for x in test_x:
    f_finv = np.sqrt(x)**2
    finv_f = np.sqrt(x**2)
    print(f"{x}\\t{f_finv}\\t\\t{finv_f}")""")

# Transformations
add_md("""## 7. Function Transformations

### 7.1 Vertical and Horizontal Shifts""")

add_code("""# Base function
def f(x):
    return np.sqrt(x)

x = np.linspace(0, 10, 200)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original
axes[0, 0].plot(x, f(x), 'b-', linewidth=2.5, label='f(x) = √x')
axes[0, 0].set_title('Original Function', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(-1, 4)

# Vertical shift up
axes[0, 1].plot(x, f(x), 'b-', linewidth=1.5, alpha=0.5, label='f(x) = √x')
axes[0, 1].plot(x, f(x) + 2, 'r-', linewidth=2.5, label='f(x) + 2 = √x + 2')
axes[0, 1].set_title('Vertical Shift Up by 2', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(-1, 5)

# Horizontal shift right
x_shifted = np.linspace(3, 10, 200)
axes[1, 0].plot(x, f(x), 'b-', linewidth=1.5, alpha=0.5, label='f(x) = √x')
axes[1, 0].plot(x_shifted, f(x_shifted - 3), 'g-', linewidth=2.5, label='f(x-3) = √(x-3)')
axes[1, 0].set_title('Horizontal Shift Right by 3', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(-1, 4)

# Combined
axes[1, 1].plot(x, f(x), 'b-', linewidth=1.5, alpha=0.5, label='f(x) = √x')
axes[1, 1].plot(x_shifted, f(x_shifted - 3) + 1, 'm-', linewidth=2.5, label='f(x-3) + 1')
axes[1, 1].set_title('Combined: Right 3, Up 1', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim(-1, 4)

plt.tight_layout()
plt.show()""")

add_md("""### 7.2 Reflections and Stretches""")

add_code("""x = np.linspace(-3, 3, 200)
f_x = x**2

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original
axes[0, 0].plot(x, f_x, 'b-', linewidth=2.5, label='f(x) = x²')
axes[0, 0].set_title('Original: f(x) = x²', fontsize=12, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylim(-10, 10)

# Reflection about x-axis
axes[0, 1].plot(x, f_x, 'b-', linewidth=1.5, alpha=0.5, label='f(x) = x²')
axes[0, 1].plot(x, -f_x, 'r-', linewidth=2.5, label='-f(x) = -x²')
axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].set_title('Reflection about x-axis', fontsize=12, fontweight='bold')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylim(-10, 10)

# Vertical stretch
axes[1, 0].plot(x, f_x, 'b-', linewidth=1.5, alpha=0.5, label='f(x) = x²')
axes[1, 0].plot(x, 2*f_x, 'g-', linewidth=2.5, label='2f(x) = 2x²')
axes[1, 0].set_title('Vertical Stretch by 2', fontsize=12, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylim(-1, 20)

# Horizontal compression
axes[1, 1].plot(x, f_x, 'b-', linewidth=1.5, alpha=0.5, label='f(x) = x²')
axes[1, 1].plot(x, (2*x)**2, 'm-', linewidth=2.5, label='f(2x) = (2x)²')
axes[1, 1].set_title('Horizontal Compression by 1/2', fontsize=12, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylim(-1, 20)

plt.tight_layout()
plt.show()""")

# Data science applications
add_md("""## 8. Data Science Applications

### 8.1 Activation Functions""")

add_code("""# Common activation functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

x = np.linspace(-5, 5, 400)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Sigmoid
axes[0, 0].plot(x, sigmoid(x), 'b-', linewidth=2.5)
axes[0, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 0].axhline(y=1, color='r', linestyle='--', alpha=0.3)
axes[0, 0].set_title('Sigmoid: σ(x) = 1/(1+e⁻ˣ)\\nRange: (0, 1)',
                     fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].set_ylabel('σ(x)')

# ReLU
axes[0, 1].plot(x, relu(x), 'r-', linewidth=2.5)
axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[0, 1].set_title('ReLU: max(0, x)\\nRange: [0, ∞)',
                     fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].set_ylabel('ReLU(x)')

# Tanh
axes[1, 0].plot(x, tanh(x), 'g-', linewidth=2.5)
axes[1, 0].axhline(y=-1, color='r', linestyle='--', alpha=0.3)
axes[1, 0].axhline(y=1, color='r', linestyle='--', alpha=0.3)
axes[1, 0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 0].set_title('Tanh: (eˣ - e⁻ˣ)/(eˣ + e⁻ˣ)\\nRange: (-1, 1)',
                     fontsize=12, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].set_ylabel('tanh(x)')

# Leaky ReLU
axes[1, 1].plot(x, leaky_relu(x), 'm-', linewidth=2.5)
axes[1, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].axvline(x=0, color='k', linestyle='--', alpha=0.3)
axes[1, 1].set_title('Leaky ReLU: max(αx, x) where α=0.01\\nRange: (-∞, ∞)',
                     fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].set_ylabel('Leaky ReLU(x)')

for ax in axes.flat:
    ax.set_xlabel('x')

plt.tight_layout()
plt.show()

print("Activation Function Properties:")
print("="*60)
print("Sigmoid: Smooth, differentiable, outputs probabilities")
print("ReLU: Simple, efficient, but not differentiable at 0")
print("Tanh: Zero-centered, steeper gradient than sigmoid")
print("Leaky ReLU: Fixes 'dying ReLU' problem")""")

add_md("""### 8.2 Data Transformations""")

add_code("""# Generate skewed data
np.random.seed(42)
data = np.random.exponential(scale=2, size=1000)

# Apply various transformations
log_transform = np.log(data + 1)
sqrt_transform = np.sqrt(data)
boxcox_transform = (data**0.5 - 1) / 0.5  # Box-Cox with λ=0.5

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original data
axes[0, 0].hist(data, bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_title(f'Original Data (Skewed)\\nSkewness: {pd.Series(data).skew():.2f}',
                     fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Value')
axes[0, 0].set_ylabel('Frequency')

# Log transform
axes[0, 1].hist(log_transform, bins=50, color='lightcoral', edgecolor='black', alpha=0.7)
axes[0, 1].set_title(f'Log Transform: log(x+1)\\nSkewness: {pd.Series(log_transform).skew():.2f}',
                     fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Value')
axes[0, 1].set_ylabel('Frequency')

# Square root transform
axes[1, 0].hist(sqrt_transform, bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[1, 0].set_title(f'Square Root Transform: √x\\nSkewness: {pd.Series(sqrt_transform).skew():.2f}',
                     fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Value')
axes[1, 0].set_ylabel('Frequency')

# Box-Cox transform
axes[1, 1].hist(boxcox_transform, bins=50, color='plum', edgecolor='black', alpha=0.7)
axes[1, 1].set_title(f'Box-Cox Transform (λ=0.5)\\nSkewness: {pd.Series(boxcox_transform).skew():.2f}',
                     fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

print("Transformation reduces skewness:")
print(f"Original: {pd.Series(data).skew():.3f}")
print(f"Log: {pd.Series(log_transform).skew():.3f}")
print(f"Sqrt: {pd.Series(sqrt_transform).skew():.3f}")
print(f"Box-Cox: {pd.Series(boxcox_transform).skew():.3f}")""")

add_md("""### 8.3 Feature Scaling""")

add_code("""# Generate sample data
np.random.seed(42)
feature1 = np.random.normal(100, 15, 100)  # Mean=100, std=15
feature2 = np.random.normal(5, 2, 100)     # Mean=5, std=2

data_df = pd.DataFrame({'Feature1': feature1, 'Feature2': feature2})

# Apply different scaling methods
min_max_scaler = MinMaxScaler()
standard_scaler = StandardScaler()

data_minmax = min_max_scaler.fit_transform(data_df)
data_standard = standard_scaler.fit_transform(data_df)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original
axes[0].scatter(data_df['Feature1'], data_df['Feature2'], alpha=0.6, s=50)
axes[0].set_title('Original Data\\n(Different Scales)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Feature 1')
axes[0].set_ylabel('Feature 2')
axes[0].grid(True, alpha=0.3)

# Min-Max scaled
axes[1].scatter(data_minmax[:, 0], data_minmax[:, 1], alpha=0.6, s=50, color='red')
axes[1].set_title('Min-Max Scaling\\n(Scaled to [0, 1])', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Feature 1 (scaled)')
axes[1].set_ylabel('Feature 2 (scaled)')
axes[1].grid(True, alpha=0.3)
axes[1].set_xlim(-0.1, 1.1)
axes[1].set_ylim(-0.1, 1.1)

# Z-score scaled
axes[2].scatter(data_standard[:, 0], data_standard[:, 1], alpha=0.6, s=50, color='green')
axes[2].set_title('Z-Score Normalization\\n(Mean=0, Std=1)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Feature 1 (standardized)')
axes[2].set_ylabel('Feature 2 (standardized)')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print("Original Statistics:")
print(data_df.describe())
print()
print("After Min-Max Scaling:")
print(pd.DataFrame(data_minmax, columns=['Feature1', 'Feature2']).describe())
print()
print("After Z-Score Normalization:")
print(pd.DataFrame(data_standard, columns=['Feature1', 'Feature2']).describe())""")

# Practice problems
add_md("""## 9. Practice Problems

### Problem 1: Domain and Range""")

add_code("""# Find domain and range of f(x) = √(x²-4)

x_left = np.linspace(-5, -2, 100)
x_right = np.linspace(2, 5, 100)

y_left = np.sqrt(x_left**2 - 4)
y_right = np.sqrt(x_right**2 - 4)

plt.figure(figsize=(10, 6))
plt.plot(x_left, y_left, 'b-', linewidth=2.5, label='f(x) = √(x²-4)')
plt.plot(x_right, y_right, 'b-', linewidth=2.5)
plt.axvline(x=-2, color='r', linestyle='--', alpha=0.5, label='Domain boundary')
plt.axvline(x=2, color='r', linestyle='--', alpha=0.5)
plt.fill_between([-2, 2], -1, 5, alpha=0.2, color='red', label='Not in domain')
plt.xlabel('x', fontsize=12)
plt.ylabel('f(x)', fontsize=12)
plt.title('f(x) = √(x²-4)\\nDomain: (-∞, -2] ∪ [2, ∞)\\nRange: [0, ∞)',
          fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

print("Solution:")
print("Domain: Require x²-4 ≥ 0 → x² ≥ 4 → |x| ≥ 2")
print("Domain: (-∞, -2] ∪ [2, ∞)")
print()
print("Range: Since √(x²-4) ≥ 0 and can get arbitrarily large")
print("Range: [0, ∞)")""")

add_md("""### Problem 2: Function Composition""")

add_code("""# Given f(x) = 2x+1 and g(x) = x²-3, find (f∘h)(x) = g(x)

# We need: f(h(x)) = g(x)
# 2h(x) + 1 = x² - 3
# 2h(x) = x² - 4
# h(x) = (x² - 4)/2

def h(x):
    return (x**2 - 4) / 2

def f(x):
    return 2*x + 1

def g(x):
    return x**2 - 3

# Verify
x_test = np.array([0, 1, 2, 3])
print("Verification that (f∘h)(x) = g(x):")
print("x\\tf(h(x))\\t\\tg(x)\\t\\tMatch?")
print("-" * 50)
for x in x_test:
    fh = f(h(x))
    gx = g(x)
    print(f"{x}\\t{fh:.2f}\\t\\t{gx:.2f}\\t\\t{np.isclose(fh, gx)}")

print()
print("Solution: h(x) = (x² - 4)/2")""")

add_md("""### Problem 3: Finding Inverse""")

add_code("""# Find inverse of f(x) = (2x+1)/(x-3)

x = symbols('x')
y = symbols('y')

# Define function symbolically
f_expr = (2*x + 1)/(x - 3)

# Solve y = f(x) for x
equation = sp.Eq(y, f_expr)
x_solution = solve(equation, x)

print("Finding inverse of f(x) = (2x+1)/(x-3)")
print()
print("Step 1: Set y = (2x+1)/(x-3)")
print("Step 2: Solve for x:")
print(f"x = {x_solution[0]}")
print()
print("Step 3: Swap x and y:")
f_inverse_expr = x_solution[0].subs(y, x)
print(f"f⁻¹(x) = {f_inverse_expr}")

# Verify numerically
f_func = lambdify(x, f_expr, 'numpy')
f_inv_func = lambdify(x, f_inverse_expr, 'numpy')

test_vals = np.array([1, 4, 7])
print()
print("Verification:")
print("x\\tf(f⁻¹(x))\\t\\tf⁻¹(f(x))")
print("-" * 40)
for val in test_vals:
    if val != 3:  # Avoid division by zero
        comp1 = f_func(f_inv_func(val))
        comp2 = f_inv_func(f_func(val))
        print(f"{val}\\t{comp1:.4f}\\t\\t{comp2:.4f}")""")

# Self-assessment
add_md("""## 10. Self-Assessment Checklist

Evaluate your mastery of this week's material:

### Core Concepts
- [ ] I understand the formal definition of a function
- [ ] I can find domain and range of any function
- [ ] I can identify one-to-one, onto, and bijective functions
- [ ] I understand the difference between codomain and range

### Operations
- [ ] I can compose functions correctly (order matters!)
- [ ] I can find inverse functions when they exist
- [ ] I understand when a function has an inverse (bijective)
- [ ] I can verify inverse properties

### Special Types
- [ ] I can identify even and odd functions
- [ ] I can recognize monotonic functions
- [ ] I understand bounded functions

### Transformations
- [ ] I can apply vertical and horizontal shifts
- [ ] I can perform reflections
- [ ] I can apply vertical and horizontal stretches/compressions
- [ ] I can combine multiple transformations

### Applications
- [ ] I understand activation functions in neural networks
- [ ] I can apply data transformations (log, sqrt, Box-Cox)
- [ ] I know when to use different scaling methods
- [ ] I can connect function concepts to ML applications

### Python Skills
- [ ] I can define and visualize functions in Python
- [ ] I can implement composition and inverse operations
- [ ] I can create informative function plots
- [ ] I completed all code examples and practice problems

---

**If you checked fewer than 80% of the boxes, review the relevant sections!**""")

add_md("""## Summary

This notebook covered:

1. **Function Fundamentals**: Definition, domain, range, notation
2. **Function Types**: One-to-one, onto, bijective
3. **Even/Odd Functions**: Symmetry properties
4. **Composition**: Combining functions (non-commutative!)
5. **Inverse Functions**: Finding and verifying inverses
6. **Transformations**: Shifts, reflections, stretches
7. **Data Science Applications**:
   - Activation functions (sigmoid, ReLU, tanh)
   - Data transformations (log, Box-Cox)
   - Feature scaling (min-max, z-score)

### Key Takeaways

- Functions are fundamental mappings with unique outputs
- Domain restrictions affect existence of inverses
- Composition order matters: (f∘g) ≠ (g∘f) generally
- Only bijective functions have inverses
- Transformations allow systematic modification of function behavior
- Functions are everywhere in data science and ML

**Next Week**: Exponential and Logarithmic Functions

---

**Excellent work completing Week 5! 🎯**""")

# Save
output_path = "/Users/godsonkurishinkal/Projects/iitm-bs-learning/01-Foundation-Level/01-Mathematics-I/notebooks/week-05-functions.ipynb"
with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"✓ Notebook created: {output_path}")
print(f"✓ Total cells: {len(notebook['cells'])}")
print(f"  - Markdown: {sum(1 for c in notebook['cells'] if c['cell_type'] == 'markdown')}")
print(f"  - Code: {sum(1 for c in notebook['cells'] if c['cell_type'] == 'code')}")
