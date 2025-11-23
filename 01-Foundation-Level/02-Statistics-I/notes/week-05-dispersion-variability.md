# Week 5: Counting Principles and Factorials - Foundation for Probability

---
Date: 2025-11-22
Course: BSMA1002 - Statistics for Data Science I
Level: Foundation
Week: 5 of 12
Source: IIT Madras Statistics I Week 5
Topic Area: Combinatorics - Counting Methods
Tags: #BSMA1002 #Counting #Factorial #Probability #Combinatorics #Week5 #Foundation
---

## Overview (BLUF)

**One-sentence definition**: Counting principles provide systematic methods to determine the number of outcomes in experiments—essential for calculating probabilities where $P(A) = \frac{\text{favorable outcomes}}{\text{total outcomes}}$.

**Why it matters**: Before computing probabilities, you must count outcomes correctly. Multiplication rule counts ordered sequences (passwords, race results), addition rule handles mutually exclusive choices (menu options, routes), and factorials count arrangements (permutations). These are building blocks for probability theory, combinations, permutations, and ultimately distributions like binomial and hypergeometric.

**When to use**:
- **Addition rule**: "How many ways to choose A OR B?" (mutually exclusive)
- **Multiplication rule**: "How many ways to do Step 1 AND Step 2 AND ...?" (sequential decisions)
- **Factorials**: "How many ways to arrange n distinct objects?"

**Prerequisites**: Basic arithmetic, set theory (mutually exclusive events from [week-01](week-01-data-types-scales.md)).

---

## Core Theory

### 1. The Addition Rule of Counting

**Principle**: If task can be done in $m$ ways OR $n$ ways (but not both), total ways = $m + n$.

**Formal statement**: If sets A and B are **disjoint** (mutually exclusive), then:
$$|A \cup B| = |A| + |B|$$

**Generalization** for k disjoint sets:
$$|A_1 \cup A_2 \cup \ldots \cup A_k| = |A_1| + |A_2| + \ldots + |A_k|$$

**Key requirement**: Options must be **mutually exclusive** (can't overlap).

#### Example 1: Restaurant Menu

**Scenario**: Restaurant offers:
- 5 vegetarian entrees
- 7 non-vegetarian entrees
- 3 vegan entrees

All categories are disjoint (no dish appears in multiple categories).

**Question**: How many entree options total?

**Solution**:
$$\text{Total options} = 5 + 7 + 3 = 15$$

```python
veg = 5
non_veg = 7
vegan = 3

total_options = veg + non_veg + vegan
print(f"Total entree options: {total_options}")

# Verify with simulation
menu = ['veg']*veg + ['non-veg']*non_veg + ['vegan']*vegan
print(f"Menu has {len(menu)} distinct dishes")
```

#### Example 2: Travel Routes

**Scenario**: From city A to B:
- 3 bus routes
- 2 train routes
- 1 flight

**Question**: How many ways to travel from A to B?

**Solution**:
$$\text{Total ways} = 3 + 2 + 1 = 6$$

**Visualization**:
```python
import matplotlib.pyplot as plt
import numpy as np

methods = ['Bus', 'Train', 'Flight']
counts = [3, 2, 1]
colors = ['skyblue', 'lightgreen', 'salmon']

plt.figure(figsize=(10, 6))
plt.bar(methods, counts, color=colors, edgecolor='black', linewidth=1.5)
plt.xlabel('Travel Method', fontsize=12)
plt.ylabel('Number of Options', fontsize=12)
plt.title('Travel Options from A to B (Addition Rule)', fontsize=14, fontweight='bold')
plt.ylim(0, 4)

for i, (method, count) in enumerate(zip(methods, counts)):
    plt.text(i, count + 0.1, str(count), ha='center', fontsize=14, fontweight='bold')

plt.axhline(y=sum(counts), color='red', linestyle='--', linewidth=2, alpha=0.7,
            label=f'Total = {sum(counts)}')
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()
```

---

### 2. The Multiplication Rule of Counting

**Principle**: If task involves **sequence** of steps:
- Step 1: $m$ ways
- Step 2: $n$ ways (for each choice in Step 1)
- Step 3: $p$ ways (for each combination of Steps 1-2)

Total ways = $m \times n \times p$

**Formal statement**: If sets A and B are independent choices:
$$|A \times B| = |A| \times |B|$$

**Key requirement**: Steps are **independent** (choice in one step doesn't limit choices in other steps).

#### Example 3: Password Creation

**Scenario**: Create password with:
- First character: uppercase letter (26 options)
- Second character: digit 0-9 (10 options)
- Third character: lowercase letter (26 options)

**Question**: How many possible passwords?

**Solution**:
$$\text{Total passwords} = 26 \times 10 \times 26 = 6,760$$

**Step-by-step**:
- Step 1 (uppercase): 26 choices
- Step 2 (digit): 10 choices **for each** uppercase choice → $26 \times 10 = 260$ combinations so far
- Step 3 (lowercase): 26 choices **for each** previous combination → $260 \times 26 = 6,760$

```python
uppercase = 26
digits = 10
lowercase = 26

total_passwords = uppercase * digits * lowercase
print(f"Total 3-character passwords: {total_passwords:,}")

# Verify with smaller example (2 letters, 2 digits)
small_upper = 2  # A, B
small_digits = 3  # 0, 1, 2

passwords = []
for letter in ['A', 'B']:
    for digit in ['0', '1', '2']:
        passwords.append(letter + digit)

print(f"\nSmall example (2×3): {len(passwords)} passwords")
print(f"Passwords: {passwords}")
print(f"Formula: 2 × 3 = {small_upper * small_digits} ✓")
```

#### Example 4: Race Rankings

**Scenario**: 8 runners compete in race.

**Question**: How many possible orderings for top 3 finishers (gold, silver, bronze)?

**Solution** (using multiplication rule):
- Gold medal: 8 choices
- Silver medal: 7 remaining choices (can't repeat gold winner)
- Bronze medal: 6 remaining choices

$$\text{Total orderings} = 8 \times 7 \times 6 = 336$$

```python
n_runners = 8

# Top 3 positions
gold_choices = n_runners
silver_choices = n_runners - 1
bronze_choices = n_runners - 2

total_orderings = gold_choices * silver_choices * bronze_choices
print(f"Possible top-3 orderings: {total_orderings}")

# Simulation verification (smaller example)
from itertools import permutations

runners_small = ['A', 'B', 'C', 'D']  # 4 runners
top_2_orderings = list(permutations(runners_small, 2))

print(f"\n4 runners, top 2 positions:")
print(f"Formula: 4 × 3 = {4*3}")
print(f"Enumeration: {len(top_2_orderings)} orderings")
print(f"First 10: {top_2_orderings[:10]}")
```

#### Example 5: Combining Addition and Multiplication

**Scenario**: Outfit selection
- Shirts: 3 (red, blue, green)
- Pants: 2 (jeans, khakis)
- Shoes: 2 (sneakers, boots)

**Question 1**: How many complete outfits (shirt + pants + shoes)?

**Solution** (multiplication rule):
$$\text{Outfits} = 3 \times 2 \times 2 = 12$$

**Question 2**: How many ways to choose shirt OR pants (not both)?

**Solution** (addition rule):
$$\text{Ways} = 3 + 2 = 5$$

```python
shirts = ['Red', 'Blue', 'Green']
pants = ['Jeans', 'Khakis']
shoes = ['Sneakers', 'Boots']

# Q1: Complete outfits (multiplication)
outfits = []
for shirt in shirts:
    for pant in pants:
        for shoe in shoes:
            outfits.append(f"{shirt} shirt + {pant} + {shoe}")

print(f"Q1: Complete outfits = {len(outfits)}")
print(f"Formula: {len(shirts)} × {len(pants)} × {len(shoes)} = {len(shirts)*len(pants)*len(shoes)}")
print("\nFirst 5 outfits:")
for i, outfit in enumerate(outfits[:5], 1):
    print(f"  {i}. {outfit}")

# Q2: Shirt OR pants (addition)
shirt_or_pants = len(shirts) + len(pants)
print(f"\nQ2: Shirt OR Pants (not both) = {shirt_or_pants}")
```

---

### 3. Factorials - Counting Arrangements

**Definition**: Factorial of non-negative integer n:
$$n! = n \times (n-1) \times (n-2) \times \ldots \times 2 \times 1$$

**Special cases**:
- $0! = 1$ (by convention, important for combinations formula)
- $1! = 1$

**Interpretation**: $n!$ counts number of ways to **arrange** n distinct objects in a row.

**Why?**:
- Position 1: n choices
- Position 2: (n-1) choices (can't reuse object in position 1)
- Position 3: (n-2) choices
- ...
- Position n: 1 choice

Total = $n \times (n-1) \times \ldots \times 1 = n!$

#### Example 6: Arranging Books

**Scenario**: 5 distinct books on shelf.

**Question**: How many arrangements?

**Solution**:
$$5! = 5 \times 4 \times 3 \times 2 \times 1 = 120$$

```python
import math

n_books = 5
arrangements = math.factorial(n_books)
print(f"{n_books}! = {arrangements}")

# Manual calculation
manual = 1
for i in range(1, n_books + 1):
    manual *= i
print(f"Manual calculation: {manual}")

# Verify with permutations (small example)
books_small = ['A', 'B', 'C']
perms = list(permutations(books_small))
print(f"\n3 books: 3! = {math.factorial(3)}")
print(f"Enumeration: {len(perms)} arrangements")
print(f"All arrangements: {perms}")
```

#### Example 7: Factorial Growth

**Question**: How fast do factorials grow?

**Analysis**:
| n | n! | Scientific Notation |
|---|-----|---------------------|
| 1 | 1 | 1 |
| 2 | 2 | 2 |
| 3 | 6 | 6 |
| 4 | 24 | 2.4×10¹ |
| 5 | 120 | 1.2×10² |
| 10 | 3,628,800 | 3.6×10⁶ |
| 20 | 2.43×10¹⁸ | 2.43×10¹⁸ |
| 50 | 3.04×10⁶⁴ | 3.04×10⁶⁴ |

**Insight**: Factorials grow **extremely fast** (faster than exponentials). This means number of arrangements explodes quickly.

```python
# Compute and visualize factorial growth
import pandas as pd

n_values = list(range(1, 16))
factorials = [math.factorial(n) for n in n_values]

df = pd.DataFrame({
    'n': n_values,
    'n!': factorials,
    'log10(n!)': [math.log10(f) for f in factorials]
})

print(df)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Linear scale (becomes vertical quickly)
axes[0].plot(n_values, factorials, marker='o', linewidth=2, markersize=8, color='blue')
axes[0].set_xlabel('n', fontsize=12)
axes[0].set_ylabel('n!', fontsize=12)
axes[0].set_title('Factorial Growth (Linear Scale)', fontsize=14, fontweight='bold')
axes[0].grid(alpha=0.3)

# Log scale (linear on log scale)
axes[1].semilogy(n_values, factorials, marker='o', linewidth=2, markersize=8, color='red')
axes[1].set_xlabel('n', fontsize=12)
axes[1].set_ylabel('n! (log scale)', fontsize=12)
axes[1].set_title('Factorial Growth (Log Scale)', fontsize=14, fontweight='bold')
axes[1].grid(alpha=0.3, which='both')

plt.tight_layout()
plt.show()
```

#### Example 8: Arranging with Repetition

**Scenario**: Arrange letters in word "BOOK" (4 letters, but two O's are identical).

**Question**: How many distinct arrangements?

**Naive answer**: $4! = 24$ (if all letters were distinct)

**Correct answer**: Since two O's are indistinguishable, we **overcount by factor of 2!**:
$$\frac{4!}{2!} = \frac{24}{2} = 12$$

**General formula** for n objects with repetitions:
$$\frac{n!}{n_1! \cdot n_2! \cdot \ldots \cdot n_k!}$$

where $n_1, n_2, \ldots, n_k$ are frequencies of repeated objects.

**Verification**:
```python
word = "BOOK"
unique_chars = list(set(word))  # ['B', 'O', 'K']

# Generate all permutations of positions
from itertools import permutations as perm
all_arrangements = list(perm(word))
unique_arrangements = list(set(all_arrangements))

print(f"Word: {word}")
print(f"All permutations (treating O's as distinct): {len(all_arrangements)}")
print(f"Unique permutations (O's indistinguishable): {len(unique_arrangements)}")
print(f"\nFormula: 4! / 2! = {math.factorial(4) // math.factorial(2)}")

print("\nUnique arrangements:")
for i, arr in enumerate(sorted(unique_arrangements), 1):
    print(f"  {i}. {''.join(arr)}")
```

---

## Data Science Applications

### 1. Password Strength Analysis

**Problem**: Estimate time to crack password using brute force.

```python
def password_strength(length, char_set_size, attempts_per_second=1e9):
    """
    Calculate time to brute force password

    Parameters:
    - length: password length
    - char_set_size: number of possible characters (26 for lowercase, 62 for alphanumeric, etc.)
    - attempts_per_second: hacker's computational power

    Returns:
    - Total combinations and time to crack (worst case)
    """
    total_combinations = char_set_size ** length  # Multiplication rule
    seconds_to_crack = total_combinations / attempts_per_second

    # Convert to human-readable units
    minutes = seconds_to_crack / 60
    hours = minutes / 60
    days = hours / 24
    years = days / 365.25

    return total_combinations, years

# Compare different password strategies
strategies = [
    ('4-digit PIN', 4, 10),
    ('6 lowercase letters', 6, 26),
    ('8 alphanumeric', 8, 62),
    ('12 alphanumeric', 12, 62),
    ('8 char (lowercase + uppercase + digits + symbols)', 8, 95)
]

print("Password Strength Analysis (1 billion attempts/sec):")
print("-" * 70)
for name, length, charset in strategies:
    combos, years_to_crack = password_strength(length, charset)
    print(f"{name:50s}: {combos:15,.0f} combinations")
    print(f"{'':50s}  {years_to_crack:15.2e} years to crack\n")
```

### 2. A/B Testing - Test Combinations

**Problem**: Testing website with multiple variables:
- 3 headline versions
- 2 button colors
- 2 image styles

**Question**: How many unique test combinations?

**Solution**: $3 \times 2 \times 2 = 12$ combinations

```python
headlines = ['Headline A', 'Headline B', 'Headline C']
buttons = ['Blue', 'Green']
images = ['Photo', 'Illustration']

# Generate all combinations
test_combinations = []
for h in headlines:
    for b in buttons:
        for i in images:
            test_combinations.append({'headline': h, 'button': b, 'image': i})

print(f"Total A/B test combinations: {len(test_combinations)}")
print("\nAll test variants:")
for idx, combo in enumerate(test_combinations, 1):
    print(f"  Variant {idx}: {combo}")

# If testing at 1000 visitors per variant
visitors_per_variant = 1000
total_visitors_needed = len(test_combinations) * visitors_per_variant
print(f"\nTotal visitors needed for testing: {total_visitors_needed:,}")
```

### 3. Feature Selection - Subset Counting

**Problem**: Dataset with 10 features. Want to test all possible feature subsets (for ML model).

**Question**: How many subsets?

**Answer**: Each feature can be IN or OUT → $2^{10} = 1,024$ subsets

**Counting logic** (multiplication rule):
- Feature 1: 2 choices (include or exclude)
- Feature 2: 2 choices
- ...
- Feature 10: 2 choices

Total = $2^{10}$

```python
n_features = 10
total_subsets = 2 ** n_features

print(f"Number of features: {n_features}")
print(f"Total possible subsets: {total_subsets:,}")
print(f"If testing each subset takes 1 minute: {total_subsets} minutes = {total_subsets/60:.1f} hours")

# Enumerate subsets for small example (4 features)
from itertools import combinations

features_small = ['Age', 'Income', 'Education', 'Location']
all_subsets = []

for r in range(len(features_small) + 1):  # r = subset size
    for subset in combinations(features_small, r):
        all_subsets.append(subset)

print(f"\n4 features → 2^4 = {2**4} subsets:")
for i, subset in enumerate(all_subsets, 1):
    print(f"  {i}. {subset if subset else '(empty set)'}")
```

### 4. Experimental Design - Treatment Orderings

**Problem**: Clinical trial with 5 treatments. Want to test on each patient in randomized order to control for time effects.

**Question**: How many possible orderings per patient?

**Solution**: $5! = 120$ orderings

```python
treatments = ['Treatment A', 'Treatment B', 'Treatment C', 'Treatment D', 'Treatment E']
n_treatments = len(treatments)

total_orderings = math.factorial(n_treatments)
print(f"Number of treatments: {n_treatments}")
print(f"Possible orderings per patient: {total_orderings}")

# If have 10 patients, could assign each a different ordering
n_patients = 10
if n_patients <= total_orderings:
    print(f"\n{n_patients} patients → can assign each a unique ordering")
    print(f"(Using {n_patients} out of {total_orderings} possible orderings)")
else:
    print(f"\n{n_patients} patients > {total_orderings} orderings")
    print(f"(Some orderings will repeat)")
```

---

## Common Pitfalls and Misconceptions

### Pitfall 1: Confusing Addition and Multiplication

❌ **Wrong**: "3 shirts and 2 pants → 3 + 2 = 5 outfits"

✅ **Right**: "3 shirts **AND** 2 pants → 3 × 2 = 6 outfits"

**Rule**:
- **OR** → Addition (mutually exclusive choices)
- **AND** → Multiplication (sequential decisions)

### Pitfall 2: Ignoring Order When It Matters

❌ **Wrong**: "3 people, choose 2 for president and VP → C(3,2) = 3 ways"

✅ **Right**: "Order matters (president ≠ VP) → 3 × 2 = 6 ways"

**Clue**: If positions/roles are different, order matters → use multiplication rule or permutations.

### Pitfall 3: Applying Multiplication When Choices Aren't Independent

❌ **Wrong**: "Pick 2 distinct cards from deck. Total = 52 × 52 = 2,704 ways"

✅ **Right**: "First card: 52 choices. Second card: 51 choices (can't repeat) → 52 × 51 = 2,652 ways"

### Pitfall 4: Forgetting Repetitions Reduce Count

❌ **Wrong**: "Arrange letters in 'MISSISSIPPI' → 11! ways"

✅ **Right**: "11!/(4!×4!×2!) ways" (accounting for repeated I's, S's, P's)

### Pitfall 5: Misusing Factorials for Selections

❌ **Wrong**: "Choose 3 people from 10 for committee → 10! ways"

✅ **Right**: "If order doesn't matter → C(10,3) = 10!/(3!×7!) = 120 ways"

**Note**: Factorials count **arrangements**, not selections.

---

## Self-Assessment and Active Recall

### Concept Check Questions

1. **Explain**: When do you use addition rule vs multiplication rule?

2. **Calculate**: 7! / 5! without computing full factorials.

3. **Identify**: Problem says "select outfit". Does order matter?

4. **True/False**: "n! is always even for n ≥ 2." Explain.

5. **Apply**: Password has 8 positions, each can be 0-9. How many passwords? Which rule?

### Practice Problems

#### Basic Level

1. Restaurant has 4 appetizers, 6 entrees, 3 desserts. How many 3-course meals?

2. Compute: 6!, 8!/6!, 10!/9!

3. License plate: 3 letters followed by 3 digits. How many possible plates?

#### Intermediate Level

4. Arrange 7 people in row. How many arrangements if:
   a) No restrictions
   b) Two specific people must sit together

5. Bit string of length 5 (each position 0 or 1). How many:
   a) Total strings
   b) Strings starting with 1
   c) Strings with exactly three 1's

6. Arrange letters in "SUCCESS". How many distinct arrangements?

#### Advanced Level

7. Prove: n! > 2^n for n ≥ 4

8. Committee of 3 from 10 people. How many ways if:
   a) No restrictions (order doesn't matter)
   b) Must include specific person
   c) Can't include two specific people together

9. How many ways to arrange n distinct objects in circle? (Hint: rotations are equivalent)

---

## Quick Reference Summary

### Key Formulas

**Addition Rule** (OR):
$$|A \cup B| = |A| + |B| \quad \text{(if disjoint)}$$

**Multiplication Rule** (AND):
$$|A \times B| = |A| \times |B|$$

**Factorial**:
$$n! = n \times (n-1) \times \ldots \times 2 \times 1$$
$$0! = 1$$

**Arrangements with Repetition**:
$$\frac{n!}{n_1! \cdot n_2! \cdot \ldots \cdot n_k!}$$

### Decision Tree

```
Counting problem
│
├─ Mutually exclusive options? (OR)
│  └─ Use ADDITION: m + n
│
├─ Sequential steps? (AND)
│  └─ Use MULTIPLICATION: m × n
│
└─ Arranging n objects?
   ├─ All distinct → n!
   └─ With repetitions → n! / (n₁! × n₂! × ...)
```

### Python Templates

```python
import math
from itertools import permutations, combinations

# Factorial
math.factorial(n)

# Generate all arrangements
list(permutations([1, 2, 3]))  # All orderings

# Password combinations (multiplication rule)
total = (char_set_size) ** (password_length)

# Arrangements with repetition
from collections import Counter
def arrangements_with_rep(word):
    n = len(word)
    freqs = Counter(word).values()
    denominator = math.prod(math.factorial(f) for f in freqs)
    return math.factorial(n) // denominator
```

### Top 3 Things to Remember

1. **Addition = OR (mutually exclusive)**, **Multiplication = AND (sequential)**
2. **n! counts arrangements** (order matters), grows extremely fast
3. **Always check**: Does order matter? Any repetitions? Choices independent?

---

## Further Resources

### Documentation
- Python `math.factorial()`
- Python `itertools.permutations()`, `itertools.combinations()`

### Books
- Sheldon Ross, "A First Course in Probability" - Chapter 1
- Feller, "An Introduction to Probability Theory" - Chapter 2

### Practice
- Project Euler (combinatorics problems)
- LeetCode (permutation/combination questions)

### Review Schedule
- **After 1 day**: Work through examples by hand
- **After 3 days**: Implement counting functions in Python
- **After 1 week**: Solve 10 practice problems (mix of rules)
- **After 2 weeks**: Apply to probability calculations (Week 6)

---

**Related Notes**:
- Previous: [week-04-central-tendency-measures.md](week-04-central-tendency-measures.md)
- Next: [week-06-probability-basics.md](week-06-probability-basics.md)
- Connection: Counting is prerequisite for probability (P = favorable/total)

**Last Updated**: 2025-11-22
**Status**: Complete and comprehensive ✅
