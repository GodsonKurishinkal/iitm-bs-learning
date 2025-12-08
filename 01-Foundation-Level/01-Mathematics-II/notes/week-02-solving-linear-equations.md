# Week 02: Solving Linear Equations

**Course**: BSMA1003 - Mathematics II
**Level**: Foundation

---

## Learning Objectives
- Apply Cramer's Rule for solving systems with explicit formulas
- Understand Row Echelon Form (REF) and Reduced Row Echelon Form (RREF)
- Master Gaussian elimination with elementary row operations

---

## 1. Cramer's Rule

### 1.1 Theory

**Cramer's Rule** provides explicit formulas for the solution of a system with the same number of equations and unknowns, when the coefficient matrix is invertible.

### 1.2 Mathematical Definition

For system $A\mathbf{x} = \mathbf{b}$ where $\det(A) \neq 0$:

$$x_i = \frac{\det(A_i)}{\det(A)}$$

where $A_i$ is matrix $A$ with column $i$ replaced by $\mathbf{b}$.

### 1.3 Key Properties

| Property | Description |
|----------|-------------|
| **Applicability** | Square systems ($n \times n$) only |
| **Requirement** | $\det(A) \neq 0$ (invertible matrix) |
| **Complexity** | $O(n! \cdot n)$ - impractical for large systems |
| **Best Use** | Small systems (2×2, 3×3) or theoretical analysis |

### 1.4 Supply Chain Application

**Retail Context**: Cramer's Rule can solve small allocation problems explicitly - determining exact quantities for each warehouse when total demand equals supply.

---

## 2. Echelon Form and Row Reduction

### 2.1 Theory

**Row echelon form** simplifies systems for back-substitution. **Reduced row echelon form** gives direct solutions.

### 2.2 Row Echelon Form (REF)

A matrix is in REF if:
- All zero rows are at the bottom
- The leading entry (pivot) of each row is to the right of the row above
- All entries below a pivot are zero

### 2.3 Reduced Row Echelon Form (RREF)

A matrix is in RREF if:
- It is in REF
- Each pivot is 1
- Each pivot is the **only** nonzero entry in its column

### 2.4 Comparison

| Feature | REF | RREF |
|---------|-----|------|
| **Pivot value** | Any nonzero | Must be 1 |
| **Above pivots** | Can be nonzero | Must be zero |
| **Solution method** | Back-substitution | Direct reading |
| **Uniqueness** | Not unique | Unique |

### 2.5 Supply Chain Application

**Retail Context**: Echelon form reveals the structure of constraints:
- Which variables are **free** (excess capacity)
- Which are **determined** (binding constraints)
- Whether the system is **consistent** (feasible solution exists)

---

## 3. Gaussian Elimination

### 3.1 Theory

**Gaussian elimination** systematically transforms a system to echelon form, then solves by back-substitution. It's the workhorse algorithm for linear systems.

### 3.2 Elementary Row Operations

| Operation | Description | Effect on Solution |
|-----------|-------------|-------------------|
| **Swap** | Exchange two rows | No change |
| **Scale** | Multiply row by nonzero scalar | No change |
| **Add** | Add multiple of one row to another | No change |

### 3.3 Algorithm Steps

1. Form augmented matrix $[A | \mathbf{b}]$
2. Apply row operations to achieve REF
3. Solve by back-substitution (or continue to RREF)

### 3.4 Complexity

$$O(n^3) \text{ for } n \times n \text{ system}$$

### 3.5 Solution Types

| $\text{rank}(A)$ vs $\text{rank}([A|\mathbf{b}])$ | Solution |
|---------------------------------------------------|----------|
| $\text{rank}(A) = \text{rank}([A|\mathbf{b}]) = n$ | Unique solution |
| $\text{rank}(A) = \text{rank}([A|\mathbf{b}]) < n$ | Infinite solutions |
| $\text{rank}(A) < \text{rank}([A|\mathbf{b}])$ | No solution |

### 3.6 Supply Chain Application

**Retail Context**: Gaussian elimination solves:
- **Transportation problems**: Optimal shipments between locations
- **Production scheduling**: Meeting demand with resource constraints
- **Multi-period inventory planning**: Balancing stock over time

---

## Summary Table

| Method | Definition | Key Property | Supply Chain Application |
|--------|------------|--------------|--------------------------|
| **Cramer's Rule** | $x_i = \det(A_i)/\det(A)$ | Explicit formula | Small allocation problems |
| **REF** | Staircase form with pivots | Enables back-substitution | Constraint structure |
| **RREF** | Unique reduced form | Direct solution reading | Variable classification |
| **Gaussian Elimination** | Row operations to REF | $O(n^3)$ complexity | Transportation, scheduling |

---

## Key Takeaways

1. **Cramer's Rule** gives closed-form solutions but is impractical for large systems
2. **Echelon forms** reveal system structure - free variables, consistency, uniqueness
3. **Gaussian elimination** is the standard practical method for solving linear systems
4. **Elementary row operations** preserve solutions - the transformed system is equivalent

---

## Next Week Preview

Week 3 covers **Introduction to Vector Spaces** - the abstract structure underlying linear algebra.

---

*IIT Madras BS Degree in Data Science*
