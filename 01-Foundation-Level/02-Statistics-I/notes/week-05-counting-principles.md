# Week 05: Counting Principles

**Course**: BSST1001 - Statistics I
**Level**: Foundation

---

## Learning Objectives
- Master fundamental counting principles (multiplication and addition)
- Understand factorials and their applications
- Recognize combinatorial explosion in real-world problems

---

## 1. Fundamental Counting Principles

### 1.1 Theory

**Counting principles** form the foundation of probability. They help calculate the number of possible outcomes in a sample space.

### 1.2 Multiplication Principle (AND)

If task A can be done in $m$ ways AND task B can be done in $n$ ways, then both tasks together can be done in:

$$m \times n \text{ ways}$$

**Generalization**: For $k$ sequential tasks with $n_1, n_2, \ldots, n_k$ ways respectively:

$$\text{Total ways} = n_1 \times n_2 \times \cdots \times n_k$$

### 1.3 Addition Principle (OR)

If task A can be done in $m$ ways OR task B can be done in $n$ ways (mutually exclusive), then either task can be done in:

$$m + n \text{ ways}$$

### 1.4 Summary of Principles

| Keyword | Principle | Operation |
|---------|-----------|-----------|
| **AND** | Multiplication | × |
| **OR** (exclusive) | Addition | + |

### 1.5 Supply Chain Application

**Retail Context**:
- How many ways to assign **5 products** to **3 warehouses**? → $3^5 = 243$
- How many unique **SKU configurations** with 3 sizes × 4 colors? → $3 \times 4 = 12$
- How many ways to select **1 supplier from 5 OR 1 from 3 backup**? → $5 + 3 = 8$

---

## 2. Factorials

### 2.1 Theory

**Factorials** count the number of ways to arrange $n$ distinct objects in order. They grow extremely fast.

### 2.2 Mathematical Definition

$$n! = n \times (n-1) \times (n-2) \times \cdots \times 2 \times 1$$

### 2.3 Special Cases

| Expression | Value |
|------------|-------|
| $0!$ | $1$ |
| $1!$ | $1$ |
| $n!$ | $n \times (n-1)!$ |

### 2.4 Factorial Growth

| n | n! |
|---|---|
| 1 | 1 |
| 5 | 120 |
| 10 | 3,628,800 |
| 15 | 1,307,674,368,000 |
| 20 | 2.4 × 10¹⁸ |

**Key insight**: Factorial growth is faster than exponential growth, making exhaustive search impossible for large $n$.

### 2.5 Supply Chain Application

**Retail Context**:
- **Delivery scheduling**: 10 orders can be sequenced in $10! = 3.6$ million ways
- **Shelf arrangement**: 8 products can be arranged in $8! = 40,320$ ways
- **Vehicle routing**: 20 stops → $20!$ routes (impossible to enumerate)

This explains why **optimization algorithms** (not exhaustive search) are essential in operations.

---

## 3. Combinatorial Explosion

### 3.1 Theory

**Combinatorial explosion** refers to the rapid growth in the number of possibilities as problem size increases. This makes brute-force approaches infeasible.

### 3.2 Examples

| Problem | Size | Possibilities |
|---------|------|---------------|
| Binary choices | $n$ items | $2^n$ |
| Arrangements | $n$ items | $n!$ |
| Subsets | $n$ items | $2^n$ |
| Routes (TSP) | $n$ cities | $(n-1)!/2$ |

### 3.3 Implications for Supply Chain

- **Why optimization matters**: Cannot check all possibilities
- **Heuristics needed**: Greedy algorithms, genetic algorithms, simulated annealing
- **Trade-off**: Optimal vs. good-enough solutions

---

## Summary Table

| Concept | Definition | Formula | Supply Chain Application |
|---------|------------|---------|--------------------------|
| **Multiplication Principle** | A AND B | $m \times n$ | SKU configurations |
| **Addition Principle** | A OR B (exclusive) | $m + n$ | Supplier alternatives |
| **Factorial** | Arrangements of n items | $n!$ | Scheduling orders |
| **Combinatorial Explosion** | Rapid growth of possibilities | — | Need for optimization |

---

## Key Takeaways

1. **AND → Multiply**, **OR → Add** (fundamental counting)
2. **Factorials** count arrangements: $n! = n \times (n-1) \times \cdots \times 1$
3. **$0! = 1$** by definition
4. **Factorial growth** explains why exhaustive search fails for large problems

---

## Next Week Preview

Week 6 covers **Permutations and Combinations** - ordered vs unordered selections.

---

*IIT Madras BS Degree in Data Science*
