#!/usr/bin/env python3
"""
Generate Week 8 Jupyter Notebook: Sequences and Series
Creates comprehensive interactive notebook with visualizations and applications.
"""

import json

def create_markdown_cell(content):
    """Create a markdown cell."""
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": content.split('\n')
    }

def create_code_cell(code, outputs=None):
    """Create a code cell with optional outputs."""
    cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": outputs or [],
        "source": code.split('\n')
    }
    return cell

# Build notebook structure
notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.9.6"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Cell 1: Title
notebook["cells"].append(create_markdown_cell("""# Week 8: Sequences and Series - Practice Notebook

---
**Date**: 2025-11-21
**Course**: BSMA1001 - Mathematics for Data Science I
**Level**: Foundation
**Week**: 8 of 12
**Topic Area**: Sequences and Series
---

## Learning Objectives

This notebook provides hands-on practice with:
- Computing and visualizing sequences
- Working with arithmetic and geometric sequences
- Evaluating finite and infinite series
- Testing series convergence
- Applying Taylor series for approximations
- Understanding algorithm complexity through series analysis

## Prerequisites

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import factorial
```"""))

# Cell 2: Setup
notebook["cells"].append(create_code_cell("""import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.special import factorial
import warnings
warnings.filterwarnings('ignore')

# Set random seed
np.random.seed(42)

# Configure plotting
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("✓ Libraries imported successfully")
print(f"NumPy version: {np.__version__}")"""))

# Cell 3: Sequences intro
notebook["cells"].append(create_markdown_cell("""## 1. Sequences

A **sequence** is an ordered list of numbers: $\\{a_1, a_2, a_3, \\ldots\\}$

General term: $a_n = f(n)$

Let's explore different types of sequences and their convergence behavior."""))

# Cell 4: Define and visualize sequences
notebook["cells"].append(create_code_cell("""def generate_sequence(formula, n_terms):
    \"\"\"Generate first n terms of a sequence.\"\"\"
    return [formula(n) for n in range(1, n_terms + 1)]

# Define several sequences
sequences = {
    'Constant': lambda n: 5,
    'Linear': lambda n: 2*n + 1,
    'Quadratic': lambda n: n**2,
    'Reciprocal': lambda n: 1/n,
    'Alternating': lambda n: (-1)**n / n,
    'Exponential Decay': lambda n: (0.8)**n,
    'Bounded Oscillating': lambda n: np.sin(n) / n
}

n_terms = 20
n_range = range(1, n_terms + 1)

fig, axes = plt.subplots(3, 3, figsize=(15, 12))
axes = axes.flatten()

for idx, (name, formula) in enumerate(sequences.items()):
    if idx < len(axes):
        terms = generate_sequence(formula, n_terms)

        ax = axes[idx]
        ax.plot(n_range, terms, 'o-', markersize=6, linewidth=1.5)
        ax.set_xlabel('n', fontsize=10)
        ax.set_ylabel('$a_n$', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Add horizontal line at limit if it converges
        if name in ['Reciprocal', 'Alternating', 'Exponential Decay', 'Bounded Oscillating']:
            ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, linewidth=1)

# Hide extra subplots
for idx in range(len(sequences), len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()

print("Sequence Types:")
print("  • Constant: Does not change")
print("  • Linear/Quadratic: Grows without bound (diverges)")
print("  • Reciprocal/Alternating: Approaches 0 (converges)")
print("  • Exponential Decay: Approaches 0 exponentially")
print("  • Bounded Oscillating: Oscillates around 0, approaches 0")"""))

# Cell 5: Limits of sequences
notebook["cells"].append(create_markdown_cell("""## 2. Limits of Sequences

A sequence $\\{a_n\\}$ **converges** to limit $L$ if:
$$\\lim_{n \\to \\infty} a_n = L$$

Let's compute limits numerically and verify convergence:"""))

# Cell 6: Computing limits
notebook["cells"].append(create_code_cell("""def compute_limit(formula, max_n=1000):
    \"\"\"Compute approximate limit by evaluating at large n.\"\"\"
    return formula(max_n)

# Test sequences
limit_sequences = {
    'n/(n+1)': lambda n: n / (n + 1),
    '(2n² + 3n)/(n² - 1)': lambda n: (2*n**2 + 3*n) / (n**2 - 1),
    '(1 + 1/n)^n': lambda n: (1 + 1/n)**n,
    '1/n²': lambda n: 1 / n**2,
    'sin(n)/n': lambda n: np.sin(n) / n
}

print("Computing Limits:")
print("=" * 70)

for name, formula in limit_sequences.items():
    # Compute for increasing values of n
    n_values = [10, 100, 1000, 10000]
    values = [formula(n) for n in n_values]

    print(f"\\nSequence: a_n = {name}")
    for n, val in zip(n_values, values):
        print(f"  n = {n:>5}: a_n = {val:.10f}")

    # Estimate limit
    limit_est = values[-1]
    print(f"  Estimated limit: {limit_est:.6f}")

# Special case: (1 + 1/n)^n converges to e
print(f"\\nNote: (1 + 1/n)^n converges to e ≈ {np.e:.6f}")"""))

# Cell 7: Arithmetic sequences
notebook["cells"].append(create_markdown_cell("""## 3. Arithmetic Sequences

**Definition:** Constant difference between consecutive terms

General term: $a_n = a_1 + (n-1)d$

Sum formula: $S_n = \\frac{n}{2}(a_1 + a_n)$"""))

# Cell 8: Arithmetic sequence operations
notebook["cells"].append(create_code_cell("""class ArithmeticSequence:
    \"\"\"Class for arithmetic sequence operations.\"\"\"

    def __init__(self, first_term, common_diff):
        self.a1 = first_term
        self.d = common_diff

    def nth_term(self, n):
        \"\"\"Calculate the nth term.\"\"\"
        return self.a1 + (n - 1) * self.d

    def sum_n_terms(self, n):
        \"\"\"Calculate sum of first n terms.\"\"\"
        an = self.nth_term(n)
        return n * (self.a1 + an) / 2

    def generate_terms(self, n):
        \"\"\"Generate first n terms.\"\"\"
        return [self.nth_term(i) for i in range(1, n + 1)]

# Example 1: 3, 7, 11, 15, ...
seq1 = ArithmeticSequence(first_term=3, common_diff=4)

print("Arithmetic Sequence: 3, 7, 11, 15, ...")
print(f"  First term a₁ = {seq1.a1}")
print(f"  Common difference d = {seq1.d}")
print(f"  10th term: a₁₀ = {seq1.nth_term(10)}")
print(f"  Sum of first 10 terms: S₁₀ = {seq1.sum_n_terms(10)}")

# Example 2: 100, 95, 90, 85, ...
seq2 = ArithmeticSequence(first_term=100, common_diff=-5)

print("\\nArithmetic Sequence: 100, 95, 90, 85, ...")
print(f"  First term a₁ = {seq2.a1}")
print(f"  Common difference d = {seq2.d}")

# Find when term equals 0
n_zero = (0 - seq2.a1) / seq2.d + 1
print(f"  Term equals 0 at n = {int(n_zero)}: a₂₁ = {seq2.nth_term(21)}")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot terms
n_vals = range(1, 16)
terms1 = [seq1.nth_term(n) for n in n_vals]
terms2 = [seq2.nth_term(n) for n in n_vals]

ax1.plot(n_vals, terms1, 'o-', label='Seq 1 (d=4)', markersize=8)
ax1.plot(n_vals, terms2, 's-', label='Seq 2 (d=-5)', markersize=8)
ax1.set_xlabel('n', fontsize=12)
ax1.set_ylabel('$a_n$', fontsize=12)
ax1.set_title('Arithmetic Sequences', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot cumulative sums
sums1 = [seq1.sum_n_terms(n) for n in n_vals]
sums2 = [seq2.sum_n_terms(n) for n in n_vals]

ax2.plot(n_vals, sums1, 'o-', label='Sum Seq 1', markersize=8)
ax2.plot(n_vals, sums2, 's-', label='Sum Seq 2', markersize=8)
ax2.set_xlabel('n', fontsize=12)
ax2.set_ylabel('$S_n$', fontsize=12)
ax2.set_title('Cumulative Sums', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

# Cell 9: Geometric sequences
notebook["cells"].append(create_markdown_cell("""## 4. Geometric Sequences

**Definition:** Constant ratio between consecutive terms

General term: $a_n = a_1 \\cdot r^{n-1}$

Finite sum: $S_n = a_1\\frac{1-r^n}{1-r}$ (if $r \\neq 1$)

Infinite sum: $S_\\infty = \\frac{a_1}{1-r}$ (if $|r| < 1$)"""))

# Cell 10: Geometric sequence operations
notebook["cells"].append(create_code_cell("""class GeometricSequence:
    \"\"\"Class for geometric sequence operations.\"\"\"

    def __init__(self, first_term, common_ratio):
        self.a1 = first_term
        self.r = common_ratio

    def nth_term(self, n):
        \"\"\"Calculate the nth term.\"\"\"
        return self.a1 * (self.r ** (n - 1))

    def sum_n_terms(self, n):
        \"\"\"Calculate sum of first n terms.\"\"\"
        if self.r == 1:
            return n * self.a1
        return self.a1 * (1 - self.r**n) / (1 - self.r)

    def infinite_sum(self):
        \"\"\"Calculate infinite sum (if |r| < 1).\"\"\"
        if abs(self.r) >= 1:
            return float('inf')
        return self.a1 / (1 - self.r)

    def converges(self):
        \"\"\"Check if sequence converges.\"\"\"
        return abs(self.r) < 1

# Example 1: 2, 6, 18, 54, ... (growth)
geo1 = GeometricSequence(first_term=2, common_ratio=3)

print("Geometric Sequence 1: 2, 6, 18, 54, ...")
print(f"  First term a₁ = {geo1.a1}")
print(f"  Common ratio r = {geo1.r}")
print(f"  8th term: a₈ = {geo1.nth_term(8):.0f}")
print(f"  Sum of first 6 terms: S₆ = {geo1.sum_n_terms(6):.0f}")
print(f"  Converges: {geo1.converges()}")

# Example 2: 1, 1/2, 1/4, 1/8, ... (decay)
geo2 = GeometricSequence(first_term=1, common_ratio=0.5)

print("\\nGeometric Sequence 2: 1, 1/2, 1/4, 1/8, ...")
print(f"  First term a₁ = {geo2.a1}")
print(f"  Common ratio r = {geo2.r}")
print(f"  10th term: a₁₀ = {geo2.nth_term(10):.6f}")
print(f"  Sum of first 10 terms: S₁₀ = {geo2.sum_n_terms(10):.6f}")
print(f"  Infinite sum: S∞ = {geo2.infinite_sum()}")
print(f"  Converges: {geo2.converges()}")

# Visualize convergence to infinite sum
n_range = range(1, 21)
partial_sums = [geo2.sum_n_terms(n) for n in n_range]
infinite_sum = geo2.infinite_sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Plot terms approaching 0
terms = [geo2.nth_term(n) for n in n_range]
ax1.plot(n_range, terms, 'o-', markersize=8, linewidth=2)
ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Limit = 0')
ax1.set_xlabel('n', fontsize=12)
ax1.set_ylabel('$a_n$', fontsize=12)
ax1.set_title('Terms Approaching 0', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-0.1, 1.1)

# Plot partial sums approaching infinite sum
ax2.plot(n_range, partial_sums, 'o-', markersize=8, linewidth=2, label='Partial Sums $S_n$')
ax2.axhline(y=infinite_sum, color='r', linestyle='--', linewidth=2, label=f'Infinite Sum = {infinite_sum}')
ax2.fill_between(n_range, partial_sums, infinite_sum, alpha=0.2)
ax2.set_xlabel('n', fontsize=12)
ax2.set_ylabel('$S_n$', fontsize=12)
ax2.set_title('Convergence to Infinite Sum', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()"""))

# Cell 11: Summation notation
notebook["cells"].append(create_markdown_cell("""## 5. Summation Notation and Formulas

Sigma notation: $\\sum_{i=m}^{n} a_i = a_m + a_{m+1} + \\cdots + a_n$

**Common formulas:**
- $\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2}$
- $\\sum_{i=1}^{n} i^2 = \\frac{n(n+1)(2n+1)}{6}$
- $\\sum_{i=1}^{n} i^3 = \\left[\\frac{n(n+1)}{2}\\right]^2$"""))

# Cell 12: Summation computations
notebook["cells"].append(create_code_cell("""def sum_integers(n):
    \"\"\"Sum of first n integers.\"\"\"
    return n * (n + 1) // 2

def sum_squares(n):
    \"\"\"Sum of first n squares.\"\"\"
    return n * (n + 1) * (2*n + 1) // 6

def sum_cubes(n):
    \"\"\"Sum of first n cubes.\"\"\"
    return (n * (n + 1) // 2) ** 2

# Verify formulas
n = 100

# Method 1: Direct computation
direct_sum = sum(range(1, n + 1))
direct_squares = sum(i**2 for i in range(1, n + 1))
direct_cubes = sum(i**3 for i in range(1, n + 1))

# Method 2: Using formulas
formula_sum = sum_integers(n)
formula_squares = sum_squares(n)
formula_cubes = sum_cubes(n)

print(f"For n = {n}:")
print("\\nSum of integers (1 + 2 + ... + n):")
print(f"  Direct: {direct_sum}")
print(f"  Formula: {formula_sum}")
print(f"  Match: {direct_sum == formula_sum}")

print("\\nSum of squares (1² + 2² + ... + n²):")
print(f"  Direct: {direct_squares}")
print(f"  Formula: {formula_squares}")
print(f"  Match: {direct_squares == formula_squares}")

print("\\nSum of cubes (1³ + 2³ + ... + n³):")
print(f"  Direct: {direct_cubes}")
print(f"  Formula: {formula_cubes}")
print(f"  Match: {direct_cubes == formula_cubes}")

# Visualize growth rates
n_values = range(1, 51)
sums_1 = [sum_integers(n) for n in n_values]
sums_2 = [sum_squares(n) for n in n_values]
sums_3 = [sum_cubes(n) for n in n_values]

plt.figure(figsize=(12, 6))
plt.plot(n_values, sums_1, 'o-', label='$\\sum i = n(n+1)/2$', markersize=4)
plt.plot(n_values, sums_2, 's-', label='$\\sum i^2 = n(n+1)(2n+1)/6$', markersize=4)
plt.plot(n_values, sums_3, '^-', label='$\\sum i^3 = [n(n+1)/2]^2$', markersize=4)
plt.xlabel('n', fontsize=12)
plt.ylabel('Sum', fontsize=12)
plt.title('Growth Rates of Summation Formulas', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()

print("\\nObservation: Higher powers grow much faster!")"""))

# Cell 13: Series convergence
notebook["cells"].append(create_markdown_cell("""## 6. Infinite Series and Convergence Tests

An infinite series $\\sum_{n=1}^{\\infty} a_n$ converges if the sequence of partial sums converges.

**Key Tests:**
1. **Divergence Test**: If $\\lim_{n\\to\\infty} a_n \\neq 0$, series diverges
2. **Geometric Series**: Converges if $|r| < 1$
3. **p-Series**: $\\sum \\frac{1}{n^p}$ converges if $p > 1$
4. **Ratio Test**: If $L = \\lim \\frac{a_{n+1}}{a_n} < 1$, converges"""))

# Cell 14: Test convergence
notebook["cells"].append(create_code_cell("""def divergence_test(term_func, n=1000):
    \"\"\"Check if limit of nth term is non-zero.\"\"\"
    limit = term_func(n)
    return abs(limit) > 1e-6

def ratio_test(term_func, n=100):
    \"\"\"Apply ratio test to determine convergence.\"\"\"
    ratio = term_func(n + 1) / term_func(n)
    return ratio

def visualize_convergence(term_func, name, n_terms=50):
    \"\"\"Visualize partial sums to check convergence.\"\"\"
    terms = [term_func(n) for n in range(1, n_terms + 1)]
    partial_sums = np.cumsum(terms)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot terms
    ax1.plot(range(1, n_terms + 1), terms, 'o-', markersize=6)
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_xlabel('n', fontsize=12)
    ax1.set_ylabel('$a_n$', fontsize=12)
    ax1.set_title(f'Terms: {name}', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot partial sums
    ax2.plot(range(1, n_terms + 1), partial_sums, 'o-', markersize=6, color='green')
    ax2.set_xlabel('n', fontsize=12)
    ax2.set_ylabel('$S_n$', fontsize=12)
    ax2.set_title(f'Partial Sums: {name}', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Check if converging
    last_10 = partial_sums[-10:]
    variation = np.std(last_10)

    return {
        'final_sum': partial_sums[-1],
        'variation': variation,
        'converging': variation < 0.1
    }

# Test 1: Harmonic series (diverges)
print("Test 1: Harmonic Series ∑(1/n)")
harmonic = lambda n: 1/n
result = visualize_convergence(harmonic, '$\\sum 1/n$', n_terms=100)
print(f"  Partial sum S₁₀₀ = {result['final_sum']:.4f}")
print(f"  Variation in last 10: {result['variation']:.4f}")
print(f"  Appears to converge: {result['converging']}")
print(f"  Divergence test: lim(1/n) = 0 (inconclusive)")
print(f"  p-Series test: p=1, so DIVERGES\\n")

# Test 2: p-series with p=2 (converges)
print("Test 2: p-Series ∑(1/n²)")
p_series = lambda n: 1/(n**2)
result = visualize_convergence(p_series, '$\\sum 1/n^2$', n_terms=100)
print(f"  Partial sum S₁₀₀ = {result['final_sum']:.4f}")
print(f"  Exact sum = π²/6 ≈ {np.pi**2/6:.4f}")
print(f"  Error: {abs(result['final_sum'] - np.pi**2/6):.6f}")
print(f"  p-Series test: p=2 > 1, so CONVERGES\\n")"""))

# Cell 15: Taylor series
notebook["cells"].append(create_markdown_cell("""## 7. Taylor Series Approximations

Taylor series of $f(x)$ at $x=a$:
$$f(x) = \\sum_{n=0}^{\\infty} \\frac{f^{(n)}(a)}{n!}(x-a)^n$$

**Common series:**
- $e^x = \\sum_{n=0}^{\\infty} \\frac{x^n}{n!}$
- $\\sin(x) = \\sum_{n=0}^{\\infty} \\frac{(-1)^n x^{2n+1}}{(2n+1)!}$
- $\\cos(x) = \\sum_{n=0}^{\\infty} \\frac{(-1)^n x^{2n}}{(2n)!}$"""))

# Cell 16: Taylor series implementations
notebook["cells"].append(create_code_cell("""def taylor_exp(x, n_terms):
    \"\"\"Approximate e^x using first n terms of Taylor series.\"\"\"
    result = 0
    for n in range(n_terms):
        result += x**n / factorial(n)
    return result

def taylor_sin(x, n_terms):
    \"\"\"Approximate sin(x) using first n terms of Taylor series.\"\"\"
    result = 0
    for n in range(n_terms):
        result += ((-1)**n * x**(2*n + 1)) / factorial(2*n + 1)
    return result

def taylor_cos(x, n_terms):
    \"\"\"Approximate cos(x) using first n terms of Taylor series.\"\"\"
    result = 0
    for n in range(n_terms):
        result += ((-1)**n * x**(2*n)) / factorial(2*n)
    return result

# Test approximations
x_test = 1.0

print(f"Approximating functions at x = {x_test}:")
print("=" * 70)

# Exponential
print("\\ne^x Taylor Series:")
for n in [1, 2, 3, 5, 10]:
    approx = taylor_exp(x_test, n)
    exact = np.exp(x_test)
    error = abs(approx - exact)
    print(f"  n={n:>2} terms: {approx:.10f} (error: {error:.2e})")
print(f"  Exact: {exact:.10f}")

# Sine
x_test_trig = np.pi/4
print(f"\\nsin(π/4) Taylor Series:")
for n in [1, 2, 3, 5, 10]:
    approx = taylor_sin(x_test_trig, n)
    exact = np.sin(x_test_trig)
    error = abs(approx - exact)
    print(f"  n={n:>2} terms: {approx:.10f} (error: {error:.2e})")
print(f"  Exact: {exact:.10f}")

# Visualize convergence
x_range = np.linspace(-2*np.pi, 2*np.pi, 200)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Exponential approximations
ax1.plot(x_range, np.exp(x_range), 'k-', linewidth=2, label='$e^x$ (exact)')
for n in [1, 2, 3, 5]:
    y_approx = [taylor_exp(x, n) for x in x_range]
    ax1.plot(x_range, y_approx, '--', label=f'n={n} terms', alpha=0.7)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('y', fontsize=12)
ax1.set_title('Taylor Approximation of $e^x$', fontsize=13, fontweight='bold')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(-2, 10)

# Sine approximations
ax2.plot(x_range, np.sin(x_range), 'k-', linewidth=2, label='$\\sin(x)$ (exact)')
for n in [1, 2, 3, 5]:
    y_approx = [taylor_sin(x, n) for x in x_range]
    ax2.plot(x_range, y_approx, '--', label=f'n={n} terms', alpha=0.7)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('y', fontsize=12)
ax2.set_title('Taylor Approximation of $\\sin(x)$', fontsize=13, fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)
ax2.set_ylim(-2, 2)

plt.tight_layout()
plt.show()

print("\\nObservation: More terms = better approximation over wider range!")"""))

# Cell 17: Algorithm complexity
notebook["cells"].append(create_markdown_cell("""## 8. Application: Algorithm Complexity Analysis

Series analysis helps understand algorithm time complexity.

**Example 1:** Nested loops
```python
for i in range(n):
    for j in range(i):
        operation()  # O(1)
```
Operations: $\\sum_{i=1}^{n} i = \\frac{n(n+1)}{2} = O(n^2)$"""))

# Cell 18: Complexity analysis
notebook["cells"].append(create_code_cell("""import time

def nested_loops(n):
    \"\"\"Count operations in nested loops.\"\"\"
    count = 0
    for i in range(n):
        for j in range(i):
            count += 1
    return count

def triple_nested(n):
    \"\"\"Count operations in triple nested loops.\"\"\"
    count = 0
    for i in range(n):
        for j in range(i):
            for k in range(j):
                count += 1
    return count

# Test for various n values
n_values = [10, 20, 30, 40, 50]

print("Nested Loop Analysis:")
print("=" * 70)

for n in n_values:
    # Actual count
    actual = nested_loops(n)

    # Formula: n(n+1)/2
    formula = sum_integers(n)

    # O(n²) approximation
    big_o = n**2

    print(f"n = {n:>2}: Operations = {actual:>4} | Formula = {formula:>4} | O(n²) = {big_o:>4}")

# Visualize complexity
n_range = range(1, 51)
actual_ops = [nested_loops(n) for n in n_range]
formula_ops = [sum_integers(n) for n in n_range]
big_o_ops = [n**2 for n in n_range]

plt.figure(figsize=(12, 6))
plt.plot(n_range, actual_ops, 'o', label='Actual Count', markersize=6, alpha=0.7)
plt.plot(n_range, formula_ops, '-', label='$n(n+1)/2$', linewidth=2)
plt.plot(n_range, big_o_ops, '--', label='$O(n^2) = n^2$', linewidth=2, alpha=0.7)
plt.xlabel('n (input size)', fontsize=12)
plt.ylabel('Operations', fontsize=12)
plt.title('Algorithm Complexity: Nested Loops', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()

print("\\nTriple Nested Loop Analysis:")
print("=" * 70)

for n in [5, 10, 15, 20]:
    actual = triple_nested(n)
    # Formula involves sum of i(i+1)/2 = sum of triangular numbers
    formula = sum_cubes(n) // 6  # Approximation
    big_o = n**3

    print(f"n = {n:>2}: Operations = {actual:>5} | O(n³) ≈ {big_o:>6}")

print("\\nConclusion: Series help analyze algorithmic complexity!")"""))

# Cell 19: Financial application
notebook["cells"].append(create_markdown_cell("""## 9. Application: Financial Calculations

**Present Value** of future cash flows uses geometric series:

$$PV = \\sum_{t=1}^{n} \\frac{CF_t}{(1+r)^t}$$

For constant cashflow (annuity): This is a geometric series!"""))

# Cell 20: Financial calculations
notebook["cells"].append(create_code_cell("""def present_value_annuity(payment, rate, periods):
    \"\"\"Calculate present value of annuity using series.\"\"\"
    # Geometric series: payment * sum((1/(1+r))^t for t=1 to n)
    # = payment * (1/(1+r)) * (1 - (1/(1+r))^n) / (1 - 1/(1+r))
    r_factor = 1 / (1 + rate)
    return payment * r_factor * (1 - r_factor**periods) / (1 - r_factor)

def future_value_annuity(payment, rate, periods):
    \"\"\"Calculate future value of annuity.\"\"\"
    return payment * ((1 + rate)**periods - 1) / rate

# Example: Monthly savings plan
monthly_payment = 500  # dollars
annual_rate = 0.06     # 6% annual
years = 10
months = years * 12
monthly_rate = annual_rate / 12

print("Savings Plan Analysis:")
print(f"  Monthly payment: ${monthly_payment}")
print(f"  Annual interest: {annual_rate*100}%")
print(f"  Duration: {years} years ({months} months)")
print(f"  Monthly rate: {monthly_rate*100:.4f}%")

# Calculate future value
fv = future_value_annuity(monthly_payment, monthly_rate, months)
total_deposits = monthly_payment * months
interest_earned = fv - total_deposits

print(f"\\nResults:")
print(f"  Total deposits: ${total_deposits:,.2f}")
print(f"  Future value: ${fv:,.2f}")
print(f"  Interest earned: ${interest_earned:,.2f}")
print(f"  Return: {(interest_earned/total_deposits)*100:.2f}%")

# Visualize growth over time
months_range = range(1, months + 1)
balance = []
cumulative_deposits = []

for m in months_range:
    balance.append(future_value_annuity(monthly_payment, monthly_rate, m))
    cumulative_deposits.append(monthly_payment * m)

plt.figure(figsize=(12, 6))
plt.plot(months_range, balance, linewidth=2, label='Account Balance')
plt.plot(months_range, cumulative_deposits, '--', linewidth=2, label='Total Deposits')
plt.fill_between(months_range, cumulative_deposits, balance, alpha=0.3, label='Interest Earned')
plt.xlabel('Months', fontsize=12)
plt.ylabel('Amount ($)', fontsize=12)
plt.title('Savings Growth Over Time', fontsize=14, fontweight='bold')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()

# Perpetuity example
print("\\n" + "="*70)
print("Perpetuity Example (infinite series):")
annual_payment = 1000
discount_rate = 0.05

# PV = payment / rate (geometric series with r < 1)
pv_perpetuity = annual_payment / discount_rate

print(f"  Annual payment: ${annual_payment}")
print(f"  Discount rate: {discount_rate*100}%")
print(f"  Present value: ${pv_perpetuity:,.2f}")
print(f"  (Sum of infinite geometric series)")"""))

# Cell 21: Practice problems
notebook["cells"].append(create_markdown_cell("""## 10. Practice Problems

Solve these computationally and verify:

**Problem 1:** Find the 20th term of arithmetic sequence: 5, 9, 13, 17, ...

**Problem 2:** Sum the first 15 terms of geometric sequence: 3, 6, 12, 24, ...

**Problem 3:** Does $\\sum_{n=1}^{\\infty} \\frac{2^n}{3^n}$ converge? Find sum if yes.

**Problem 4:** Approximate $e^{0.2}$ using 5 terms of Taylor series.

**Problem 5:** Analyze complexity of:
```python
for i in range(n):
    for j in range(n):
        for k in range(n):
            operation()
```"""))

# Cell 22: Solutions
notebook["cells"].append(create_code_cell("""print("=" * 70)
print("PRACTICE PROBLEM SOLUTIONS")
print("=" * 70)

# Problem 1
print("\\nProblem 1: 20th term of 5, 9, 13, 17, ...")
seq = ArithmeticSequence(5, 4)
term_20 = seq.nth_term(20)
print(f"  a₂₀ = {term_20}")
print(f"  Verification: 5 + (20-1)×4 = {5 + 19*4}")

# Problem 2
print("\\nProblem 2: Sum first 15 terms of 3, 6, 12, 24, ...")
geo = GeometricSequence(3, 2)
sum_15 = geo.sum_n_terms(15)
print(f"  S₁₅ = {sum_15:.0f}")
print(f"  Formula: 3(1-2¹⁵)/(1-2) = 3(2¹⁵-1) = {3*(2**15-1)}")

# Problem 3
print("\\nProblem 3: Convergence of ∑(2ⁿ/3ⁿ) = ∑(2/3)ⁿ")
geo3 = GeometricSequence(1, 2/3)
print(f"  r = 2/3 = {2/3:.4f}")
print(f"  |r| < 1: {abs(2/3) < 1}")
print(f"  Converges: YES")
infinite_sum = geo3.infinite_sum()
print(f"  Sum = 1/(1-2/3) = {infinite_sum:.4f}")

# Problem 4
print("\\nProblem 4: Approximate e^0.2 using 5 terms")
x = 0.2
approx = taylor_exp(x, 5)
exact = np.exp(x)
error = abs(approx - exact)
print(f"  Approximation: {approx:.10f}")
print(f"  Exact value: {exact:.10f}")
print(f"  Error: {error:.2e}")

# Problem 5
print("\\nProblem 5: Triple nested loop complexity")
print("  for i in range(n):")
print("    for j in range(n):")
print("      for k in range(n):")
print("        operation()  # O(1)")
print(f"\\n  Operations = n × n × n = n³")
print(f"  Time Complexity: O(n³)")
for n in [10, 20, 30]:
    ops = n**3
    print(f"    n={n:>2}: {ops:>6} operations")"""))

# Cell 23: Self-assessment
notebook["cells"].append(create_markdown_cell("""## 11. Self-Assessment Checklist

Check your understanding:

**Sequences:**
- [ ] Can define sequences and find general terms
- [ ] Understand convergence vs divergence
- [ ] Can compute limits of sequences
- [ ] Know difference between bounded/unbounded sequences

**Arithmetic Sequences:**
- [ ] Can find nth term using $a_n = a_1 + (n-1)d$
- [ ] Can compute sum using $S_n = \\frac{n}{2}(a_1 + a_n)$
- [ ] Recognize arithmetic patterns in problems

**Geometric Sequences:**
- [ ] Can find nth term using $a_n = a_1 r^{n-1}$
- [ ] Can compute finite sum
- [ ] Understand infinite sum condition ($|r| < 1$)
- [ ] Can identify when geometric series applies

**Summation:**
- [ ] Understand sigma notation
- [ ] Know common formulas ($\\sum i$, $\\sum i^2$, $\\sum i^3$)
- [ ] Can manipulate summations using properties

**Convergence Tests:**
- [ ] Can apply divergence test
- [ ] Recognize geometric series
- [ ] Understand p-series test
- [ ] Can use ratio test for more complex series

**Taylor Series:**
- [ ] Know common Taylor expansions ($e^x$, $\\sin x$, $\\cos x$)
- [ ] Can approximate functions using series
- [ ] Understand error behavior with more terms

**Applications:**
- [ ] Can analyze algorithm complexity using series
- [ ] Understand financial calculations (annuities, perpetuities)
- [ ] Recognize series in real-world problems

---

## Next Steps

**Week 9 Preview: Limits and Continuity**
- Formal definition of limits
- One-sided limits and infinite limits
- Continuity and types of discontinuities
- Intermediate Value Theorem
- Applications to optimization and root finding

---

**Excellent progress! Sequences and series are fundamental to calculus, algorithm analysis, and many data science applications. Master these concepts and you'll have a strong foundation for advanced topics!**"""))

# Save notebook
output_path = "/Users/godsonkurishinkal/Projects/iitm-bs-learning/01-Foundation-Level/01-Mathematics-I/notebooks/week-08-sequences-series.ipynb"

with open(output_path, 'w') as f:
    json.dump(notebook, f, indent=2)

print(f"✓ Notebook created successfully")
print(f"✓ Output: {output_path}")
print(f"✓ Total cells: {len(notebook['cells'])}")
markdown_count = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'markdown')
code_count = sum(1 for cell in notebook['cells'] if cell['cell_type'] == 'code')
print(f"✓ Markdown cells: {markdown_count}")
print(f"✓ Code cells: {code_count}")
