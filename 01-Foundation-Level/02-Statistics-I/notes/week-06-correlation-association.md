# Week 6: Permutations and Combinations - Counting Ordered and Unordered Selections

---
Date: 2025-11-22
Course: BSMA1002 - Statistics for Data Science I
Level: Foundation
Week: 6 of 12
Source: IIT Madras Statistics I Week 6
Topic Area: Combinatorics - Permutations & Combinations
Tags: #BSMA1002 #Permutations #Combinations #Combinatorics #Probability #Week6 #Foundation
---

## Overview (BLUF)

**One-sentence definition**: Permutations count **ordered** arrangements (order matters), while combinations count **unordered** selections (order doesn't matter)—both essential for calculating probabilities in scenarios involving selections from groups.

**Why it matters**: Many probability problems involve selecting or arranging objects. Does the order matter? (president vs vice president → permutation). Or just which objects are selected? (committee members → combination). Mastering these formulas is critical for probability distributions (binomial, hypergeometric), hypothesis testing, and sampling theory.

**When to use**:
- **Permutations**: Rankings, passwords, race positions, ordered sequences
- **Combinations**: Committees, lottery tickets, card hands, feature subsets

**Visual distinction**:
```
Objects: {A, B, C}
Select 2:

Permutations (order matters): AB, BA, AC, CA, BC, CB → 6 ways
Combinations (order doesn't matter): {A,B}, {A,C}, {B,C} → 3 ways
```

**Prerequisites**: Factorial notation, multiplication rule ([week-05](week-05-dispersion-variability.md)).

---

## Core Theory

### 1. Permutations - Ordered Arrangements

#### 1.1 Permutations of n Distinct Objects (All Taken)

**Question**: How many ways to arrange n distinct objects?

**Answer**: $n!$

**Reasoning**: Multiplication rule
- Position 1: n choices
- Position 2: (n-1) choices
- Position 3: (n-2) choices
- ...
- Position n: 1 choice

Total = $n \times (n-1) \times \ldots \times 1 = n!$

#### Example 1: Arranging Books

**Problem**: 5 distinct books on shelf. How many arrangements?

**Solution**:
$$P(5,5) = 5! = 120$$

```python
import math

n = 5
arrangements = math.factorial(n)
print(f"{n} books → {arrangements} arrangements")
```

#### 1.2 Permutations of r Objects from n (Order Matters)

**Question**: From n distinct objects, select r and arrange them. How many ways?

**Formula**:
$$P(n,r) = \frac{n!}{(n-r)!} = n \times (n-1) \times \ldots \times (n-r+1)$$

**Notation**: $P(n,r)$, $_nP_r$, or $P_r^n$

**Reasoning**:
- Position 1: n choices
- Position 2: (n-1) choices
- ...
- Position r: (n-r+1) choices

Total = $n \times (n-1) \times \ldots \times (n-r+1)$

**Why the formula works**:
$$P(n,r) = \frac{n!}{(n-r)!} = \frac{n!}{(n-r)!} = n \times (n-1) \times \ldots \times (n-r+1) \times \frac{(n-r)!}{(n-r)!}$$

The $(n-r)!$ terms cancel.

#### Example 2: Race Positions

**Problem**: 10 runners. How many ways to award gold, silver, bronze?

**Solution** (selecting and ordering 3 from 10):
$$P(10,3) = \frac{10!}{(10-3)!} = \frac{10!}{7!} = 10 \times 9 \times 8 = 720$$

```python
def permutation(n, r):
    """Calculate P(n,r) = n! / (n-r)!"""
    return math.factorial(n) // math.factorial(n - r)

n_runners = 10
r_positions = 3

result = permutation(n_runners, r_positions)
print(f"P({n_runners},{r_positions}) = {result}")

# Verify manually
manual = n_runners * (n_runners-1) * (n_runners-2)
print(f"Manual: {n_runners} × {n_runners-1} × {n_runners-2} = {manual}")
```

#### Example 3: Password with Distinct Characters

**Problem**: 4-character password from 26 lowercase letters (no repetition).

**Solution**:
$$P(26,4) = \frac{26!}{22!} = 26 \times 25 \times 24 \times 23 = 358,800$$

```python
n_letters = 26
password_length = 4

total_passwords = permutation(n_letters, password_length)
print(f"4-char passwords (no repetition): {total_passwords:,}")

# With repetition allowed (multiplication rule)
with_repetition = n_letters ** password_length
print(f"4-char passwords (with repetition): {with_repetition:,}")
```

---

### 2. Combinations - Unordered Selections

#### 2.1 Combinations of r Objects from n

**Question**: From n distinct objects, select r. Order doesn't matter. How many ways?

**Formula**:
$$C(n,r) = \binom{n}{r} = \frac{n!}{r!(n-r)!}$$

**Notation**: $C(n,r)$, $_nC_r$, $\binom{n}{r}$ (binomial coefficient)

**Derivation** (key insight):

1. If order mattered: $P(n,r) = \frac{n!}{(n-r)!}$ ways
2. But order doesn't matter, so we **overcount** by factor of $r!$ (number of ways to arrange r objects)
3. Therefore: $C(n,r) = \frac{P(n,r)}{r!} = \frac{n!}{r!(n-r)!}$

#### Example 4: Lottery Numbers

**Problem**: Lottery: choose 6 numbers from 1-49. Order doesn't matter.

**Solution**:
$$C(49,6) = \frac{49!}{6! \cdot 43!} = 13,983,816$$

```python
def combination(n, r):
    """Calculate C(n,r) = n! / (r! × (n-r)!)"""
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

n_numbers = 49
r_picks = 6

total_combos = combination(n_numbers, r_picks)
print(f"C({n_numbers},{r_picks}) = {total_combos:,}")
print(f"\nOdds of winning: 1 in {total_combos:,}")

# Using scipy
from scipy.special import comb
result_scipy = int(comb(n_numbers, r_picks))
print(f"Using scipy.special.comb: {result_scipy:,}")
```

#### Example 5: Committee Selection

**Problem**: Select 3-person committee from 8 candidates.

**Solution**:
$$C(8,3) = \frac{8!}{3! \cdot 5!} = \frac{8 \times 7 \times 6}{3 \times 2 \times 1} = \frac{336}{6} = 56$$

**Key insight**: {Alice, Bob, Carol} = {Carol, Alice, Bob} → same committee!

```python
n_candidates = 8
committee_size = 3

total_committees = combination(n_candidates, committee_size)
print(f"C({n_candidates},{committee_size}) = {total_committees}")

# Enumerate (small example)
from itertools import combinations

candidates_small = ['A', 'B', 'C', 'D']
committees_small = list(combinations(candidates_small, 2))

print(f"\n4 candidates, select 2:")
print(f"C(4,2) = {combination(4, 2)}")
print(f"All combinations: {committees_small}")
```

#### 2.2 Properties of Binomial Coefficients

**Symmetry**:
$$\binom{n}{r} = \binom{n}{n-r}$$

**Reasoning**: Choosing r objects to **include** is same as choosing (n-r) objects to **exclude**.

**Pascal's Identity**:
$$\binom{n}{r} = \binom{n-1}{r-1} + \binom{n-1}{r}$$

**Interpretation**: To select r from n objects:
- Either include specific object (choose r-1 from remaining n-1)
- Or exclude it (choose r from remaining n-1)

**Sum**:
$$\sum_{r=0}^{n} \binom{n}{r} = 2^n$$

**Interpretation**: Total number of subsets of n-element set is $2^n$.

#### Example 6: Pascal's Triangle

```python
def pascal_triangle(n_rows):
    """Generate Pascal's triangle (binomial coefficients)"""
    triangle = []
    for n in range(n_rows):
        row = [combination(n, r) for r in range(n+1)]
        triangle.append(row)
    return triangle

# Generate and display
triangle = pascal_triangle(7)
print("Pascal's Triangle (C(n,r)):")
for i, row in enumerate(triangle):
    print(f"n={i}: {row}")

# Visualize
import numpy as np
import matplotlib.pyplot as plt

max_val = max(max(row) for row in triangle)

fig, ax = plt.subplots(figsize=(12, 10))

for n, row in enumerate(triangle):
    for r, value in enumerate(row):
        x = r - n/2
        y = -n
        ax.text(x, y, str(value), ha='center', va='center', fontsize=14,
                bbox=dict(boxstyle='circle', facecolor='lightblue', edgecolor='black'))

ax.set_xlim(-4, 4)
ax.set_ylim(-7, 1)
ax.axis('off')
ax.set_title("Pascal's Triangle - Binomial Coefficients C(n,r)", fontsize=16, fontweight='bold')
plt.tight_layout()
plt.show()

# Verify properties
n, r = 10, 4
print(f"\nSymmetry: C({n},{r}) = {combination(n,r)}, C({n},{n-r}) = {combination(n,n-r)}")
print(f"Pascal's Identity: C({n},{r}) = {combination(n,r)}")
print(f"  C({n-1},{r-1}) + C({n-1},{r}) = {combination(n-1,r-1)} + {combination(n-1,r)} = {combination(n-1,r-1) + combination(n-1,r)}")
```

---

### 3. Permutations vs Combinations - Decision Framework

| Question Asks About... | Order Matters? | Formula | Example |
|------------------------|----------------|---------|---------|
| Arranging all n objects | Yes | $n!$ | Bookshelf arrangement |
| Selecting r, arranging them | Yes | $P(n,r) = \frac{n!}{(n-r)!}$ | Race podium positions |
| Selecting r, order irrelevant | No | $C(n,r) = \frac{n!}{r!(n-r)!}$ | Lottery numbers |
| Password (with repetition) | Yes | $n^r$ | 4-digit PIN |
| Subsets | No | $2^n$ | Power set |

**Key questions to ask**:
1. **Does order matter?**
   - "Arrangement", "sequence", "ordering" → YES → Permutation
   - "Selection", "choose", "committee" → NO → Combination

2. **Repetition allowed?**
   - Not allowed → Use P(n,r) or C(n,r)
   - Allowed → Use multiplication rule $n^r$

#### Example 7: Decision Tree Practice

```python
# Problem 1: 5 people, choose president and VP
# Order matters (president ≠ VP)
problem1 = permutation(5, 2)
print(f"Problem 1: P(5,2) = {problem1}")

# Problem 2: 5 people, choose 2-person committee
# Order doesn't matter
problem2 = combination(5, 2)
print(f"Problem 2: C(5,2) = {problem2}")

# Problem 3: 3-digit code (0-9, repetition allowed)
# Multiplication rule
problem3 = 10 ** 3
print(f"Problem 3: 10^3 = {problem3}")

# Problem 4: Arrange 4 distinct books
# All arrangements
problem4 = math.factorial(4)
print(f"Problem 4: 4! = {problem4}")

# Problem 5: Choose 3 toppings from 10 (order doesn't matter)
problem5 = combination(10, 3)
print(f"Problem 5: C(10,3) = {problem5}")
```

---

## Data Science Applications

### 1. Feature Selection - Comparing All Subsets

**Problem**: 12 features, want to test all possible 4-feature subsets.

**Solution**:
$$C(12,4) = \frac{12!}{4! \cdot 8!} = 495 \text{ subsets}$$

```python
n_features = 12
subset_size = 4

total_subsets = combination(n_features, subset_size)
print(f"Total 4-feature subsets from {n_features} features: {total_subsets}")

# If each model takes 2 seconds to train
train_time_seconds = total_subsets * 2
train_time_minutes = train_time_seconds / 60
print(f"Training time: {train_time_minutes:.1f} minutes")

# Enumerate (small example)
features = ['Age', 'Income', 'Education', 'Location', 'Gender']
subsets_2 = list(combinations(features, 2))
print(f"\n5 features, select 2: C(5,2) = {len(subsets_2)}")
for i, subset in enumerate(subsets_2, 1):
    print(f"  {i}. {subset}")
```

### 2. Hyperparameter Grid Search

**Problem**:
- 3 values for learning rate
- 4 values for max_depth
- 2 values for min_samples_split

**Solution** (multiplication rule):
$$3 \times 4 \times 2 = 24 \text{ configurations}$$

But if we want top-k configurations, use combinations:

```python
# Total configurations
configs = 3 * 4 * 2
print(f"Total hyperparameter configurations: {configs}")

# If testing all pairwise combinations of hyperparameter values
# (for interaction effects)
param_pairs = combination(3, 2)  # 3 parameters, choose 2
print(f"Pairwise combinations of parameters: {param_pairs}")
```

### 3. A/B Testing - Treatment Assignment

**Problem**: 100 users, assign 50 to treatment A, 50 to control B.

**Question**: How many ways to choose 50 for treatment?

**Solution**:
$$C(100,50) = \frac{100!}{50! \cdot 50!} \approx 1.01 \times 10^{29}$$

```python
n_users = 100
treatment_size = 50

total_assignments = combination(n_users, treatment_size)
print(f"C({n_users},{treatment_size}) = {total_assignments:.2e}")
print("This is astronomically large!")
```

### 4. Probability Calculations

**Problem**: Poker hand (5 cards from 52-card deck). What's probability of specific hand?

**Solution**:
$$P(\text{specific hand}) = \frac{1}{C(52,5)} = \frac{1}{2,598,960}$$

```python
# Total 5-card hands
total_hands = combination(52, 5)
print(f"Total 5-card poker hands: {total_hands:,}")

# Probability of royal flush (10, J, Q, K, A of same suit)
# Only 4 royal flushes (one per suit)
royal_flushes = 4
prob_royal_flush = royal_flushes / total_hands
print(f"\nRoyal flush hands: {royal_flushes}")
print(f"P(royal flush) = {prob_royal_flush:.10f}")
print(f"Odds: 1 in {int(1/prob_royal_flush):,}")

# Probability of any pair
# Choose value for pair (13 choices)
# Choose 2 cards of that value: C(4,2)
# Choose 3 other values: C(12,3)
# Choose 1 card from each: 4^3
pair_hands = 13 * combination(4,2) * combination(12,3) * (4**3)
prob_pair = pair_hands / total_hands
print(f"\nPair hands: {pair_hands:,}")
print(f"P(pair) = {prob_pair:.4f}")
```

---

## Common Pitfalls and Misconceptions

### Pitfall 1: Confusing P(n,r) and C(n,r)

❌ **Wrong**: "Select 3 from 10 → P(10,3) = 720 ways"

✅ **Right**: "If order doesn't matter → C(10,3) = 120 ways"

**Remember**: $C(n,r) = \frac{P(n,r)}{r!}$ (combinations are permutations divided by arrangements of selected items)

### Pitfall 2: Using Combination When Repetition Allowed

❌ **Wrong**: "4-digit PIN (0-9 with repetition) → C(10,4)"

✅ **Right**: "Multiplication rule → $10^4 = 10,000$"

### Pitfall 3: Forgetting to Divide by r!

❌ **Wrong**: "C(n,r) = $\frac{n!}{(n-r)!}$"

✅ **Right**: "C(n,r) = $\frac{n!}{r!(n-r)!}$" (don't forget $r!$ in denominator!)

### Pitfall 4: Computing Large Factorials Inefficiently

❌ **Wrong**: Compute 100! then 95! separately

✅ **Right**: $P(100,5) = \frac{100!}{95!} = 100 \times 99 \times 98 \times 97 \times 96$ (cancel common factors)

```python
# Inefficient (overflow risk)
# result = math.factorial(100) // math.factorial(95)

# Efficient
result = 100 * 99 * 98 * 97 * 96
print(f"P(100,5) = {result:,}")
```

### Pitfall 5: Misapplying "Choose" Interpretation

❌ **Wrong**: "$\binom{5}{3}$ means 'from 5, choose 3 times'"

✅ **Right**: "$\binom{5}{3}$ means 'from 5 objects, select 3'" (one-time selection)

---

## Self-Assessment and Active Recall

### Concept Check Questions

1. **Distinguish**: When do you use P(n,r) vs C(n,r)?

2. **Compute**: Calculate C(7,3) and P(7,3). Why are they different?

3. **Apply**: 10 finalists for 3 prizes (gold, silver, bronze). Permutation or combination?

4. **Explain**: Why is $\binom{n}{0} = \binom{n}{n} = 1$?

5. **Prove**: Show that $\binom{n}{r} = \binom{n}{n-r}$ using the formula.

### Practice Problems

#### Basic Level

1. How many 3-letter "words" (arrangements) from {A,B,C,D,E} without repetition?

2. Choose 4 students from 20 for committee. How many ways?

3. Compute: P(6,2), C(6,2), and explain why one is larger.

#### Intermediate Level

4. Deck of 52 cards. How many:
   a) 5-card hands
   b) 5-card sequences (order matters)

5. Password: 8 characters from 26 letters + 10 digits. How many passwords if:
   a) Repetition allowed
   b) No repetition

6. From 10 men and 8 women, select 5-person committee with at least 2 women. How many ways?

#### Advanced Level

7. Prove: $\binom{n}{r} = \binom{n-1}{r-1} + \binom{n-1}{r}$ (Pascal's Identity)

8. Show that $\sum_{r=0}^{n} \binom{n}{r} = 2^n$ (hint: binomial theorem with x=y=1)

9. How many ways to distribute 10 identical balls into 3 distinct boxes? (hint: stars and bars)

---

## Quick Reference Summary

### Key Formulas

**Permutation** (order matters):
$$P(n,r) = \frac{n!}{(n-r)!}$$

**Combination** (order doesn't matter):
$$C(n,r) = \binom{n}{r} = \frac{n!}{r!(n-r)!}$$

**Relationship**:
$$C(n,r) = \frac{P(n,r)}{r!}$$

**With repetition**:
$$n^r \text{ (ordered)}, \quad \binom{n+r-1}{r} \text{ (unordered)}$$

### Decision Flowchart

```
Problem involves selection/arrangement?
│
├─ Repetition allowed?
│  ├─ Yes → nʳ (multiplication rule)
│  └─ No → Continue
│
└─ Does ORDER matter?
   ├─ Yes → P(n,r) = n!/(n-r)!
   └─ No → C(n,r) = n!/[r!(n-r)!]
```

### Python Templates

```python
import math
from scipy.special import comb, perm
from itertools import permutations, combinations

# Manual calculation
def P(n, r):
    return math.factorial(n) // math.factorial(n - r)

def C(n, r):
    return math.factorial(n) // (math.factorial(r) * math.factorial(n - r))

# Using scipy (handles large numbers better)
comb(n, r, exact=True)  # Combination
perm(n, r, exact=True)  # Permutation

# Generate all permutations/combinations
list(permutations([1,2,3], 2))
list(combinations([1,2,3], 2))
```

### Top 3 Things to Remember

1. **Order matters → Permutation**, **Order doesn't matter → Combination**
2. **C(n,r) < P(n,r)** because combinations don't count different orderings
3. **Divide by r!** when going from permutations to combinations

---

## Further Resources

### Documentation
- Python `math.factorial()`, `math.comb()`, `math.perm()`
- SciPy `scipy.special.comb()`, `scipy.special.perm()`
- Python `itertools.permutations()`, `itertools.combinations()`

### Books
- Kenneth Rosen, "Discrete Mathematics and Its Applications" - Chapter 6
- Sheldon Ross, "A First Course in Probability" - Chapter 1

### Practice
- Project Euler problems (many use combinatorics)
- Brilliant.org (combinatorics course)
- AoPS (Art of Problem Solving) - Counting & Probability

### Review Schedule
- **After 1 day**: Compute 10 permutation/combination problems by hand
- **After 3 days**: Implement functions, solve probability problems
- **After 1 week**: Apply to poker hand probabilities
- **After 2 weeks**: Use for binomial distribution (Week 9)

---

**Related Notes**:
- Previous: [week-05-dispersion-variability.md](week-05-dispersion-variability.md)
- Next: [week-07-probability-basics.md](week-07-probability-basics.md)
- Connection: Combinatorics → probability calculations (Week 7+)

**Last Updated**: 2025-11-22
**Status**: Complete and comprehensive ✅
