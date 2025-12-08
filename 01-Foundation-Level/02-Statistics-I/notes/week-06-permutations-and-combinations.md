# Week 06: Permutations and Combinations

**Course**: BSST1001 - Statistics I
**Level**: Foundation

---

## Learning Objectives
- Understand permutations (order matters)
- Master combinations (order doesn't matter)
- Apply decision framework to counting problems

---

## 1. Permutations

### 1.1 Theory

**Permutations** count arrangements where **ORDER MATTERS**. Selecting 3 items from 5 in different orders gives different permutations.

### 1.2 Mathematical Definition

$$P(n, r) = \frac{n!}{(n-r)!}$$

Where:
- $n$ = total number of items
- $r$ = number of items being arranged
- $P(n, r)$ = number of ordered arrangements

### 1.3 Special Cases

| Case | Formula | Example |
|------|---------|---------|
| Arrange all n items | $P(n, n) = n!$ | Arrange 5 books: $5! = 120$ |
| Arrange r from n | $P(n, r) = \frac{n!}{(n-r)!}$ | Top 3 from 10: $\frac{10!}{7!} = 720$ |

### 1.4 Python Implementation

```python
from math import perm
perm(n, r)  # Returns P(n, r)
```

### 1.5 Supply Chain Application

**Retail Context**:
- **Picking sequence** in warehouse - order affects travel time
- **Processing orders** - sequence impacts lead time
- **Delivery routes** - order of stops matters
- **Ranking suppliers** - 1st, 2nd, 3rd place distinctions

---

## 2. Combinations

### 2.1 Theory

**Combinations** count selections where **ORDER DOES NOT MATTER**. Selecting 3 items from 5 gives the same combination regardless of selection order.

### 2.2 Mathematical Definition

$$C(n, r) = \binom{n}{r} = \frac{n!}{r!(n-r)!}$$

Where:
- $\binom{n}{r}$ is read as "n choose r"
- Dividing by $r!$ removes the ordering

### 2.3 Relationship to Permutations

$$C(n, r) = \frac{P(n, r)}{r!}$$

**Intuition**: Combinations = Permutations ÷ (ways to arrange the selection)

### 2.4 Properties of Combinations

| Property | Formula |
|----------|---------|
| Symmetry | $\binom{n}{r} = \binom{n}{n-r}$ |
| Boundary | $\binom{n}{0} = \binom{n}{n} = 1$ |
| Sum | $\sum_{r=0}^{n} \binom{n}{r} = 2^n$ |

### 2.5 Python Implementation

```python
from math import comb
comb(n, r)  # Returns C(n, r)
```

### 2.6 Supply Chain Application

**Retail Context**:
- **Product selection** for promotion (which 5 of 20 products)
- **Warehouse selection** to fulfill order
- **Team formation** for projects
- **Sampling** products for quality inspection

---

## 3. Decision Framework

### 3.1 Key Question: Does Order Matter?

| Question | Answer | Use |
|----------|--------|-----|
| Does order matter? | **Yes** | Permutation |
| Does order matter? | **No** | Combination |

### 3.2 With or Without Replacement?

| Scenario | Order Matters | Order Doesn't Matter |
|----------|---------------|---------------------|
| **With replacement** | $n^r$ | $\binom{n+r-1}{r}$ |
| **Without replacement** | $P(n,r) = \frac{n!}{(n-r)!}$ | $C(n,r) = \binom{n}{r}$ |

### 3.3 Quick Comparison

| n=10, r=3 | Formula | Result |
|-----------|---------|--------|
| Permutation | $\frac{10!}{7!}$ | 720 |
| Combination | $\frac{10!}{3! \cdot 7!}$ | 120 |

**Note**: Permutations are always ≥ Combinations (by factor of $r!$)

---

## 4. Common Problem Types

### 4.1 Identification Guide

| Problem Type | Order? | Example |
|--------------|--------|---------|
| Rankings/Positions | Yes | Top 3 suppliers |
| Sequences | Yes | Password digits |
| Selections/Groups | No | Committee members |
| Subsets | No | Items in a bundle |

### 4.2 Supply Chain Examples

| Scenario | Type | Formula |
|----------|------|---------|
| Rank 3 warehouses by efficiency | Permutation | $P(10, 3)$ |
| Select 5 SKUs for audit | Combination | $C(100, 5)$ |
| Assign priorities to 4 orders | Permutation | $P(4, 4) = 4!$ |
| Choose 2 suppliers from 8 | Combination | $C(8, 2)$ |

---

## Summary Table

| Concept | Definition | Formula | Supply Chain Application |
|---------|------------|---------|--------------------------|
| **Permutation** | Ordered arrangements | $P(n,r) = \frac{n!}{(n-r)!}$ | Route sequencing |
| **Combination** | Unordered selections | $C(n,r) = \frac{n!}{r!(n-r)!}$ | Product selection |
| **Relationship** | Combinations remove order | $C = P / r!$ | — |

---

## Key Takeaways

1. **Permutations**: Order matters → $P(n,r) = \frac{n!}{(n-r)!}$
2. **Combinations**: Order doesn't matter → $C(n,r) = \frac{n!}{r!(n-r)!}$
3. **Key question**: "Does order matter?" determines which to use
4. These are **building blocks for probability** calculations

---

## Next Week Preview

Week 7 covers **Introduction to Probability** - sample spaces, events, and probability rules.

---

*IIT Madras BS Degree in Data Science*
