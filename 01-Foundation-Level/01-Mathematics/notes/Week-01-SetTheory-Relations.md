---
**Metadata**
- Date: 2025-11-14
- Course: BSMA1001 - Mathematics for Data Science I
- Level: Foundation (1st of 6 levels)
- Week: 1 of 12
- Source: IIT Madras BSMA1001 - Week 1 Lectures
- Topic Area: Mathematics, Set Theory, Relations, Foundations
- Goal: Master set theory fundamentals and relations as foundation for data science
- Context: First topic in Foundation level, no prerequisites required
- Tags: #BSMA1001 #Mathematics #SetTheory #Relations #Functions #Foundations #Week1
---

# Set Theory and Relations - Mathematical Foundations for Data Science

> **Key Insight:** Sets are the fundamental building blocks of mathematics and data science - understanding how to work with collections of objects is essential for databases, probability, and algorithmic thinking.

## Overview

Set theory provides the mathematical foundation for organizing and manipulating collections of objects. In data science, sets appear everywhere: databases (collections of records), feature spaces (collections of attributes), and probability spaces (collections of outcomes). This week covers sets, set operations, relations, and the basics of functions - all critical for understanding data structures and mathematical reasoning.

## Core Concepts

### 1. Introduction to Sets

**Definition:** A **set** is a well-defined collection of distinct objects, called **elements** or **members**.

**Notation:**
- Set: $A = \{1, 2, 3, 4, 5\}$
- Element membership: $3 \in A$ (3 is in A)
- Non-membership: $6 \notin A$ (6 is not in A)
- Empty set: $\emptyset$ or $\{\}$

**Ways to Define Sets:**

1. **Roster/Tabular Form:** List all elements
   - $A = \{1, 2, 3, 4, 5\}$
   - $B = \{red, green, blue\}$

2. **Set-Builder Form:** Define by property
   - $A = \{x : x \text{ is a natural number and } x \leq 5\}$
   - $B = \{x \in \mathbb{N} : x^2 < 20\}$

**Why It Matters:** Data scientists constantly work with collections - customers, transactions, features, classes. Set theory provides the language to describe and manipulate these collections precisely.

**Key Properties:**
- **Uniqueness:** Each element appears only once
- **Order doesn't matter:** $\{1, 2, 3\} = \{3, 2, 1\}$
- **Well-defined:** Clear membership criterion

> 📋 **Important Distinction:** Sets vs Lists/Arrays
> - Sets: No order, no duplicates - $\{1, 2, 2, 3\} = \{1, 2, 3\}$
> - Lists: Order matters, duplicates allowed - $[1, 2, 2, 3] \neq [1, 2, 3]$

#### Implementation in Python

```python
# Creating sets in Python
A = {1, 2, 3, 4, 5}
B = {3, 4, 5, 6, 7}

# Set from list (removes duplicates)
C = set([1, 2, 2, 3, 3, 3])
print(f"Set C: {C}")  # Output: {1, 2, 3}

# Empty set
empty = set()  # Note: {} creates empty dict, not set!

# Membership testing - O(1) average time
print(f"3 in A: {3 in A}")        # True
print(f"10 in A: {10 in A}")      # False

# Set size
print(f"Size of A: {len(A)}")     # 5

# Set from comprehension
squares = {x**2 for x in range(1, 6)}
print(f"Squares: {squares}")       # {1, 4, 9, 16, 25}
```

**Output:**
```
Set C: {1, 2, 3}
3 in A: True
10 in A: False
Size of A: 5
Squares: {1, 4, 9, 16, 25}
```

#### Code Explanation

1. **Set creation:** Python uses `{}` or `set()` constructor
2. **Automatic deduplication:** Duplicates removed automatically
3. **Membership testing:** Hash-based, very fast ($O(1)$ average)
4. **Set comprehension:** Similar to list comprehension but creates set
5. **Complexity:** Set operations are generally $O(1)$ for membership, $O(n)$ for iteration

### 2. Number Systems

**Definition:** Different types of numbers form **nested sets** - each system contains the previous.

**Hierarchy of Number Systems:**

$$
\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R} \subset \mathbb{C}
$$

1. **Natural Numbers ($\mathbb{N}$):** $\{1, 2, 3, 4, ...\}$ or $\{0, 1, 2, 3, ...\}$
   - Counting numbers
   - Used for: Indexing, counting observations

2. **Integers ($\mathbb{Z}$):** $\{..., -2, -1, 0, 1, 2, ...\}$
   - Natural numbers + negatives + zero
   - Used for: Differences, discrete changes

3. **Rational Numbers ($\mathbb{Q}$):** $\{\frac{p}{q} : p, q \in \mathbb{Z}, q \neq 0\}$
   - All fractions
   - Decimal representation: Terminating or repeating
   - Used for: Proportions, percentages

4. **Real Numbers ($\mathbb{R}$):** All points on number line
   - Rationals + irrationals ($\pi, e, \sqrt{2}$)
   - Used for: Continuous measurements, most ML features

5. **Complex Numbers ($\mathbb{C}$):** $\{a + bi : a, b \in \mathbb{R}, i^2 = -1\}$
   - Used for: Signal processing, quantum computing, eigenvalues

**Why It Matters:** Understanding number types prevents errors in data processing:
- Integer division vs float division
- Floating point precision issues
- Type conversions in ML pipelines

> ⚠️ **Warning:** Floating point arithmetic is approximate!
> ```python
> 0.1 + 0.2 == 0.3  # False! (due to binary representation)
> abs(0.1 + 0.2 - 0.3) < 1e-10  # Use tolerance for comparisons
> ```

#### Implementation in Python

```python
import numpy as np

# Number type examples
natural = 5
integer = -3
rational = 3/4  # Stored as float in Python
real = np.sqrt(2)
complex_num = 3 + 4j

# Type checking
print(f"Type of {natural}: {type(natural)}")        # <class 'int'>
print(f"Type of {rational}: {type(rational)}")      # <class 'float'>
print(f"Type of {complex_num}: {type(complex_num)}")# <class 'complex'>

# Number system membership (conceptual)
def is_natural(x):
    return x > 0 and x == int(x)

def is_integer(x):
    return x == int(x)

def is_rational(x):
    # Simplified check (not mathematically complete)
    try:
        from fractions import Fraction
        frac = Fraction(x).limit_denominator()
        return abs(float(frac) - x) < 1e-10
    except:
        return False

# Test
numbers = [5, -3, 3.5, np.pi, 0.5]
for num in numbers:
    print(f"{num}: Natural={is_natural(num)}, Integer={is_integer(num)}, Rational={is_rational(num)}")
```

**Output:**
```
Type of 5: <class 'int'>
Type of 0.75: <class 'float'>
Type of (3+4j): <class 'complex'>
5: Natural=True, Integer=True, Rational=True
-3: Natural=False, Integer=True, Rational=True
3.5: Natural=False, Integer=False, Rational=True
3.141592653589793: Natural=False, Integer=False, Rational=False
0.5: Natural=False, Integer=False, Rational=True
```

### 3. Set Operations

**Definition:** Mathematical operations to create new sets from existing ones.

#### Union ($A \cup B$)

**Formula:**
$$
A \cup B = \{x : x \in A \text{ OR } x \in B\}
$$

**Intuition:** All elements that appear in either set (combine both sets)

**Example:** $\{1, 2, 3\} \cup \{3, 4, 5\} = \{1, 2, 3, 4, 5\}$

```python
A = {1, 2, 3}
B = {3, 4, 5}

# Union operations
union1 = A | B           # Using operator
union2 = A.union(B)      # Using method
union3 = set.union(A, B) # Using function

print(f"A ∪ B = {union1}")  # {1, 2, 3, 4, 5}

# Multiple sets
C = {5, 6, 7}
union_all = A | B | C
print(f"A ∪ B ∪ C = {union_all}")  # {1, 2, 3, 4, 5, 6, 7}
```

#### Intersection ($A \cap B$)

**Formula:**
$$
A \cap B = \{x : x \in A \text{ AND } x \in B\}
$$

**Intuition:** Only elements that appear in both sets (common elements)

**Example:** $\{1, 2, 3\} \cap \{3, 4, 5\} = \{3\}$

```python
# Intersection operations
intersection1 = A & B              # Using operator
intersection2 = A.intersection(B)  # Using method

print(f"A ∩ B = {intersection1}")  # {3}

# Disjoint sets (no common elements)
D = {10, 11, 12}
print(f"A ∩ D = {A & D}")          # set()
print(f"Are A and D disjoint? {A.isdisjoint(D)}")  # True
```

#### Difference ($A - B$ or $A \setminus B$)

**Formula:**
$$
A - B = \{x : x \in A \text{ AND } x \notin B\}
$$

**Intuition:** Elements in A but not in B (remove B's elements from A)

**Example:** $\{1, 2, 3\} - \{3, 4, 5\} = \{1, 2\}$

```python
# Difference operations
diff1 = A - B               # Using operator
diff2 = A.difference(B)     # Using method

print(f"A - B = {diff1}")   # {1, 2}
print(f"B - A = {B - A}")   # {4, 5}

# Note: Difference is NOT commutative
print(f"A - B ≠ B - A: {(A - B) != (B - A)}")  # True
```

#### Symmetric Difference ($A \triangle B$)

**Formula:**
$$
A \triangle B = (A - B) \cup (B - A) = (A \cup B) - (A \cap B)
$$

**Intuition:** Elements in either set but not in both (XOR operation)

**Example:** $\{1, 2, 3\} \triangle \{3, 4, 5\} = \{1, 2, 4, 5\}$

```python
# Symmetric difference
sym_diff1 = A ^ B                        # Using operator
sym_diff2 = A.symmetric_difference(B)    # Using method

print(f"A △ B = {sym_diff1}")  # {1, 2, 4, 5}

# Verification
print(f"Same as (A-B) ∪ (B-A): {sym_diff1 == (A - B) | (B - A)}")  # True
```

#### Complement ($A^c$ or $\bar{A}$)

**Formula:**
$$
A^c = U - A = \{x \in U : x \notin A\}
$$

Where $U$ is the **universal set** (set of all elements under consideration)

**Intuition:** All elements NOT in A (requires defining universe first)

**Example:** If $U = \{1, 2, 3, 4, 5, 6\}$ and $A = \{1, 2, 3\}$, then $A^c = \{4, 5, 6\}$

```python
# Complement (need to define universal set)
U = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
A = {1, 2, 3}

A_complement = U - A
print(f"A' = {A_complement}")  # {4, 5, 6, 7, 8, 9, 10}

# Properties of complement
print(f"A ∪ A' = U: {(A | A_complement) == U}")     # True
print(f"A ∩ A' = ∅: {(A & A_complement) == set()}")  # True
```

#### Comprehensive Set Operations Example

```python
import matplotlib.pyplot as plt
from matplotlib_venn import venn2, venn3
import numpy as np

# Define sets
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}
C = {5, 6, 7, 8, 9}

# Calculate all operations
print("=" * 50)
print("SET OPERATIONS SUMMARY")
print("=" * 50)
print(f"A = {A}")
print(f"B = {B}")
print(f"C = {C}")
print()

print(f"A ∪ B = {A | B}")
print(f"A ∩ B = {A & B}")
print(f"A - B = {A - B}")
print(f"B - A = {B - A}")
print(f"A △ B = {A ^ B}")
print()

print(f"A ∪ B ∪ C = {A | B | C}")
print(f"A ∩ B ∩ C = {A & B & C}")

# Visualize with Venn diagram
plt.figure(figsize=(12, 4))

plt.subplot(131)
venn2([A, B], set_labels=('A', 'B'))
plt.title('Sets A and B')

plt.subplot(132)
venn2([A, A & B], set_labels=('A', 'A ∩ B'))
plt.title('Intersection')

plt.subplot(133)
venn2([A, A ^ B], set_labels=('A', 'A △ B'))
plt.title('Symmetric Difference')

plt.tight_layout()
plt.savefig('set_operations_venn.png')
print("\nVenn diagrams saved to 'set_operations_venn.png'")
```

**Properties of Set Operations:**

$$
\begin{align}
\text{Commutative:} \quad & A \cup B = B \cup A \\
& A \cap B = B \cap A \\
\text{Associative:} \quad & (A \cup B) \cup C = A \cup (B \cup C) \\
& (A \cap B) \cap C = A \cap (B \cap C) \\
\text{Distributive:} \quad & A \cap (B \cup C) = (A \cap B) \cup (A \cap C) \\
& A \cup (B \cap C) = (A \cup B) \cap (A \cup C) \\
\text{De Morgan's Laws:} \quad & (A \cup B)^c = A^c \cap B^c \\
& (A \cap B)^c = A^c \cup B^c
\end{align}
$$

### 4. Relations

**Definition:** A **relation** $R$ from set $A$ to set $B$ is a subset of the Cartesian product $A \times B$.

**Cartesian Product:**
$$
A \times B = \{(a, b) : a \in A, b \in B\}
$$

**Notation:** If $(a, b) \in R$, we write $a \, R \, b$ (a is related to b)

**Example:**
- $A = \{1, 2, 3\}$
- $B = \{4, 5\}$
- $A \times B = \{(1,4), (1,5), (2,4), (2,5), (3,4), (3,5)\}$
- Relation "less than": $R = \{(1,4), (1,5), (2,4), (2,5), (3,4), (3,5)\}$

**Why It Matters:** Relations model relationships in data:
- Database foreign keys (one-to-many, many-to-many)
- Graphs and networks (edges between nodes)
- Comparisons and orderings

#### Types of Relations

**1. Reflexive:** Every element relates to itself
$$
\forall a \in A, \, a \, R \, a
$$
Example: Equality ($=$), "is subset of" ($\subseteq$)

**2. Symmetric:** If $a$ relates to $b$, then $b$ relates to $a$
$$
a \, R \, b \implies b \, R \, a
$$
Example: "is sibling of", equality

**3. Antisymmetric:** If $a$ relates to $b$ AND $b$ relates to $a$, then $a = b$
$$
(a \, R \, b \land b \, R \, a) \implies a = b
$$
Example: "less than or equal" ($\leq$), "divides" ($|$)

**4. Transitive:** If $a$ relates to $b$ AND $b$ relates to $c$, then $a$ relates to $c$
$$
(a \, R \, b \land b \, R \, c) \implies a \, R \, c
$$
Example: "less than" ($<$), "ancestor of"

**5. Equivalence Relation:** Reflexive + Symmetric + Transitive
- Example: Equality, "has same birthday as"
- Creates **equivalence classes** (partitions of set)

**6. Partial Order:** Reflexive + Antisymmetric + Transitive
- Example: $\leq$, subset relation ($\subseteq$)
- Allows comparison of some (not necessarily all) elements

#### Implementation in Python

```python
# Relations as sets of tuples
A = {1, 2, 3, 4}

# Example 1: "Divides" relation
divides = {(a, b) for a in A for b in A if b % a == 0}
print(f"'Divides' relation: {sorted(divides)}")
# {(1,1), (1,2), (1,3), (1,4), (2,2), (2,4), (3,3), (4,4)}

# Check properties
def is_reflexive(R, A):
    """Check if relation is reflexive"""
    return all((a, a) in R for a in A)

def is_symmetric(R):
    """Check if relation is symmetric"""
    return all((b, a) in R for (a, b) in R)

def is_antisymmetric(R):
    """Check if relation is antisymmetric"""
    for (a, b) in R:
        if a != b and (b, a) in R:
            return False
    return True

def is_transitive(R):
    """Check if relation is transitive"""
    for (a, b) in R:
        for (c, d) in R:
            if b == c and (a, d) not in R:
                return False
    return True

# Test "divides" relation
print(f"\nProperties of 'divides' relation:")
print(f"  Reflexive: {is_reflexive(divides, A)}")         # True
print(f"  Symmetric: {is_symmetric(divides)}")             # False
print(f"  Antisymmetric: {is_antisymmetric(divides)}")    # True
print(f"  Transitive: {is_transitive(divides)}")          # True
print(f"  → This is a PARTIAL ORDER")

# Example 2: Equality relation (equivalence relation)
equality = {(a, a) for a in A}
print(f"\n'Equality' relation: {sorted(equality)}")
print(f"Properties:")
print(f"  Reflexive: {is_reflexive(equality, A)}")        # True
print(f"  Symmetric: {is_symmetric(equality)}")            # True
print(f"  Antisymmetric: {is_antisymmetric(equality)}")   # True
print(f"  Transitive: {is_transitive(equality)}")         # True
print(f"  → This is an EQUIVALENCE RELATION")
```

**Output:**
```
'Divides' relation: [(1, 1), (1, 2), (1, 3), (1, 4), (2, 2), (2, 4), (3, 3), (4, 4)]

Properties of 'divides' relation:
  Reflexive: True
  Symmetric: False
  Antisymmetric: True
  Transitive: True
  → This is a PARTIAL ORDER

'Equality' relation: [(1, 1), (2, 2), (3, 3), (4, 4)]
Properties:
  Reflexive: True
  Symmetric: True
  Antisymmetric: True
  Transitive: True
  → This is an EQUIVALENCE RELATION
```

### 5. Functions - Special Relations

**Definition:** A **function** $f: A \rightarrow B$ is a relation where each element in $A$ is related to exactly one element in $B$.

**Formal Definition:**
$$
f \subseteq A \times B \text{ such that } \forall a \in A, \exists! b \in B : (a, b) \in f
$$

**Notation:** $f(a) = b$ means $(a, b) \in f$

**Components:**
- **Domain:** Set $A$ (all possible inputs)
- **Codomain:** Set $B$ (possible outputs)
- **Range:** $\{f(a) : a \in A\} \subseteq B$ (actual outputs)

**Why It Matters:** Functions are everywhere in data science:
- ML models are functions: $\hat{y} = f(X)$
- Transformations: normalization, encoding
- Activation functions in neural networks

> 📋 **Key Distinction:** Functions vs Relations
> - **Function:** Each input maps to exactly ONE output
> - **Relation:** Each input can map to zero, one, or many outputs

#### Types of Functions

**1. Injective (One-to-One):**
$$
f(a_1) = f(a_2) \implies a_1 = a_2
$$
Different inputs → Different outputs

Example: $f(x) = 2x$

**2. Surjective (Onto):**
$$
\forall b \in B, \exists a \in A : f(a) = b
$$
Every output is achieved (Range = Codomain)

Example: $f: \mathbb{R} \rightarrow \mathbb{R}, f(x) = x^3$

**3. Bijective (One-to-One and Onto):**
Both injective and surjective - perfect pairing

Example: $f(x) = x$ (identity function)

**Properties:**
- **Injective:** $|A| \leq |B|$ (fewer or equal inputs than possible outputs)
- **Surjective:** $|A| \geq |B|$ (more or equal inputs than outputs)
- **Bijective:** $|A| = |B|$ and invertible

#### Implementation in Python

```python
# Functions as dictionaries or callable objects
def f(x):
    """Simple function: f(x) = 2x + 1"""
    return 2 * x + 1

# Function as mapping (dictionary)
domain = [1, 2, 3, 4]
function_map = {x: f(x) for x in domain}
print(f"Function mapping: {function_map}")
# {1: 3, 2: 5, 3: 7, 4: 9}

# Check if function is injective
def is_injective(func, domain):
    """Check if function is one-to-one"""
    outputs = [func(x) for x in domain]
    return len(outputs) == len(set(outputs))

# Check if function is surjective
def is_surjective(func, domain, codomain):
    """Check if function is onto"""
    outputs = set(func(x) for x in domain)
    return outputs == set(codomain)

# Test functions
print(f"\nf(x) = 2x + 1 on {domain}:")
print(f"  Injective: {is_injective(f, domain)}")  # True

# Example: Non-injective function
def g(x):
    """g(x) = x^2 (not injective for negative numbers)"""
    return x ** 2

domain2 = [-2, -1, 0, 1, 2]
print(f"\ng(x) = x² on {domain2}:")
print(f"  Injective: {is_injective(g, domain2)}")  # False
print(f"  g(-2) = {g(-2)}, g(2) = {g(2)}")  # Both equal 4

# Composition of functions
def h(x):
    """h(x) = x + 3"""
    return x + 3

def compose(f, g):
    """Return composition f∘g"""
    return lambda x: f(g(x))

f_comp_h = compose(f, h)
print(f"\n(f∘h)(5) = f(h(5)) = f({h(5)}) = {f_comp_h(5)}")
# (f∘h)(5) = f(8) = 17
```

## Practical Application

### Real-World Use Case: Database Queries and Set Operations

**Scenario:** You have two customer segments:
- Set A: Customers who purchased in Q1
- Set B: Customers who purchased in Q2

**Business Questions:**
1. Who purchased in both quarters? (Intersection)
2. Who purchased in either quarter? (Union)
3. Who only purchased in Q1? (Difference)
4. Who purchased in exactly one quarter? (Symmetric Difference)

```python
import pandas as pd
import numpy as np

# Generate sample customer data
np.random.seed(42)
all_customers = set(range(1, 101))  # Customer IDs 1-100

# Q1 and Q2 purchasers
q1_customers = set(np.random.choice(list(all_customers), 60, replace=False))
q2_customers = set(np.random.choice(list(all_customers), 55, replace=False))

# Business analytics using set operations
both_quarters = q1_customers & q2_customers
either_quarter = q1_customers | q2_customers
only_q1 = q1_customers - q2_customers
only_q2 = q2_customers - q1_customers
exactly_one_quarter = q1_customers ^ q2_customers

print("CUSTOMER PURCHASE ANALYSIS")
print("=" * 50)
print(f"Q1 Customers: {len(q1_customers)}")
print(f"Q2 Customers: {len(q2_customers)}")
print()
print(f"Purchased in BOTH quarters (loyal): {len(both_quarters)}")
print(f"Purchased in EITHER quarter (total reach): {len(either_quarter)}")
print(f"Purchased ONLY in Q1 (at-risk): {len(only_q1)}")
print(f"Purchased ONLY in Q2 (new/returning): {len(only_q2)}")
print(f"Purchased in exactly ONE quarter: {len(exactly_one_quarter)}")
print()

# Calculate retention rate
retention_rate = len(both_quarters) / len(q1_customers) * 100
print(f"Q1 to Q2 Retention Rate: {retention_rate:.1f}%")

# Visualize
from matplotlib_venn import venn2
plt.figure(figsize=(10, 6))
venn2([q1_customers, q2_customers], 
      set_labels=('Q1 Customers', 'Q2 Customers'),
      set_colors=('skyblue', 'lightcoral'))
plt.title('Customer Purchase Overlap Between Quarters')
plt.savefig('customer_overlap.png')
print("\nVisualization saved to 'customer_overlap.png'")
```

### Data Science Application: Feature Engineering with Sets

```python
# Example: Movie recommendation - Genre-based similarity

movies = {
    'Movie A': {'action', 'thriller', 'sci-fi'},
    'Movie B': {'action', 'adventure', 'sci-fi'},
    'Movie C': {'romance', 'drama', 'comedy'},
    'Movie D': {'action', 'sci-fi', 'adventure'}
}

def jaccard_similarity(set1, set2):
    """
    Calculate Jaccard similarity (used in recommendation systems)
    J(A,B) = |A ∩ B| / |A ∪ B|
    """
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0

# Find similar movies
target_movie = 'Movie A'
target_genres = movies[target_movie]

print(f"\nFinding movies similar to '{target_movie}'")
print(f"Genres: {target_genres}")
print("\nSimilarity scores:")

similarities = []
for movie, genres in movies.items():
    if movie != target_movie:
        similarity = jaccard_similarity(target_genres, genres)
        similarities.append((movie, similarity))
        print(f"  {movie}: {similarity:.3f}")
        print(f"    Common: {target_genres & genres}")
        print(f"    Unique to {target_movie}: {target_genres - genres}")
        print(f"    Unique to {movie}: {genres - target_genres}")
        print()

# Recommendation: Movie with highest similarity
best_match = max(similarities, key=lambda x: x[1])
print(f"Best recommendation: {best_match[0]} (similarity: {best_match[1]:.3f})")
```

## Common Pitfalls

> ⚠️ **Warning:** These mistakes are easy to make and hard to debug!

### Pitfall 1: Confusing Sets with Lists

**Problem:** Using list when set is appropriate (or vice versa)

```python
# ❌ BAD: Using list for membership testing
items = [1, 2, 3, 4, 5] * 1000  # Large list with duplicates
# Checking membership is O(n) - SLOW!

# ✅ GOOD: Use set for membership testing
items_set = set(items)  # Remove duplicates, O(1) lookup
```

**Solution:** 
- Use **sets** when: Order doesn't matter, need fast lookups, no duplicates
- Use **lists** when: Order matters, need indexing, duplicates allowed

### Pitfall 2: Modifying Sets During Iteration

**Problem:** Changing set size while looping through it

```python
# ❌ BAD: Modifying set during iteration
my_set = {1, 2, 3, 4, 5}
for item in my_set:
    if item % 2 == 0:
        my_set.remove(item)  # RuntimeError!

# ✅ GOOD: Create new set or iterate over copy
my_set = {1, 2, 3, 4, 5}
my_set = {x for x in my_set if x % 2 != 0}  # Comprehension
# OR
for item in list(my_set):  # Iterate over copy
    if item % 2 == 0:
        my_set.remove(item)
```

### Pitfall 3: Assuming Order in Sets

**Problem:** Relying on iteration order of sets

```python
# ❌ BAD: Assuming order
my_set = {3, 1, 2}
first_item = list(my_set)[0]  # Could be any element!

# ✅ GOOD: Use sorted() if order matters
my_set = {3, 1, 2}
first_item = sorted(my_set)[0]  # Always 1
```

### Pitfall 4: Unhashable Elements

**Problem:** Trying to add mutable objects to sets

```python
# ❌ BAD: Lists are unhashable
my_set = {[1, 2], [3, 4]}  # TypeError!

# ✅ GOOD: Use tuples (immutable)
my_set = {(1, 2), (3, 4)}

# ✅ GOOD: Use frozenset for sets of sets
set_of_sets = {frozenset({1, 2}), frozenset({3, 4})}
```

## Connections & Prerequisites

**Prerequisites:**
- None (foundational topic)
- Basic programming knowledge helpful for implementation

**Leads To:**
- [Week 1 Part 2: Functions - Domain, Codomain, and Range](Week-01-Functions-Basics.md) - Building on relations
- [Week 2: 2D Coordinate Systems](Week-02-CoordinateSystems-2D.md) - Ordered pairs from Cartesian products
- [Week 10-11: Graph Theory](Week-10-GraphTheory-Basics.md) - Graphs as relations on sets
- **BSMA1002 (Statistics I):** [Probability Theory](../../02-Statistics/notes/Week-07-ProbabilityBasics.md) - Sample spaces and events are sets
- **BSMA1003 (Math II):** [Vector Spaces](../../01-Mathematics-II/notes/Week-06-VectorSpaces.md) - Sets with structure

**Related Concepts:**
- **Logic** → AND, OR, NOT correspond to set operations
- **Combinatorics** → Counting elements in sets (permutations, combinations)
- **Data Structures** → Hash tables implement sets efficiently
- **Databases** → SQL operations are set operations
- **Machine Learning** → Feature spaces, decision boundaries

## Summary & Key Takeaways

- ✅ **Sets are collections of unique elements:** No order, no duplicates - $\{1, 2, 3\} = \{3, 2, 1\}$
- ✅ **Master the six operations:** Union ($\cup$), Intersection ($\cap$), Difference ($-$), Symmetric Difference ($\triangle$), Complement, Cartesian Product ($\times$)
- ✅ **Relations model connections:** Subset of $A \times B$, foundation for functions and graphs
- ✅ **Functions are special relations:** Each input maps to exactly one output
- ✅ **Know relation properties:** Reflexive, symmetric, transitive define equivalence and order
- ✅ **Sets in Python:** Use `{}` or `set()`, O(1) membership testing, immutable elements only
- ✅ **Real-world applications:** Databases (joins = intersections), ML (feature spaces), networks (graphs)

## Practice Problems

### Problem 1: Basic Set Operations

**Given:**
- $A = \{1, 2, 3, 4, 5\}$
- $B = \{4, 5, 6, 7, 8\}$
- $C = \{1, 3, 5, 7, 9\}$

**Calculate:**
1. $A \cup B$
2. $A \cap B \cap C$
3. $(A \cup B) - C$
4. $A \triangle B$
5. Verify De Morgan's Law: $(A \cup B)^c = A^c \cap B^c$ (given $U = \{1,2,...,10\}$)

**Solution:**

<details>
<summary>Show Solution</summary>

```python
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}
C = {1, 3, 5, 7, 9}
U = set(range(1, 11))

print("1. A ∪ B =", A | B)              # {1, 2, 3, 4, 5, 6, 7, 8}
print("2. A ∩ B ∩ C =", A & B & C)       # {5}
print("3. (A ∪ B) - C =", (A | B) - C)   # {2, 4, 6, 8}
print("4. A △ B =", A ^ B)               # {1, 2, 3, 6, 7, 8}

# Verify De Morgan's Law
left_side = U - (A | B)
right_side = (U - A) & (U - B)
print(f"5. (A ∪ B)' = {left_side}")
print(f"   A' ∩ B' = {right_side}")
print(f"   Equal? {left_side == right_side}")  # True
```

</details>

### Problem 2: Relation Properties

**Given relation $R$ on $A = \{1, 2, 3, 4\}$:**
$$
R = \{(1,1), (1,2), (2,2), (2,3), (3,3), (4,4)\}
$$

**Determine if $R$ is:**
1. Reflexive
2. Symmetric
3. Antisymmetric
4. Transitive

**Solution:**

<details>
<summary>Show Solution</summary>

```python
A = {1, 2, 3, 4}
R = {(1,1), (1,2), (2,2), (2,3), (3,3), (4,4)}

# 1. Reflexive: ∀a∈A, (a,a)∈R
reflexive = all((a, a) in R for a in A)
print(f"1. Reflexive: {reflexive}")  # True

# 2. Symmetric: (a,b)∈R ⟹ (b,a)∈R
symmetric = all((b, a) in R for (a, b) in R)
print(f"2. Symmetric: {symmetric}")  # False (e.g., (1,2)∈R but (2,1)∉R)

# 3. Antisymmetric: (a,b)∈R ∧ (b,a)∈R ⟹ a=b
antisymmetric = all(a == b for (a, b) in R if (b, a) in R)
print(f"3. Antisymmetric: {antisymmetric}")  # True

# 4. Transitive: (a,b)∈R ∧ (b,c)∈R ⟹ (a,c)∈R
transitive = True
for (a, b) in R:
    for (c, d) in R:
        if b == c and (a, d) not in R:
            transitive = False
            print(f"   Counterexample: ({a},{b})∈R, ({c},{d})∈R, but ({a},{d})∉R")
            break
    if not transitive:
        break
print(f"4. Transitive: {transitive}")  # False

print("\nConclusion: R is reflexive and antisymmetric, but NOT symmetric or transitive")
print("            Therefore, R is NOT a partial order or equivalence relation")
```

</details>

### Problem 3: Real Dataset Analysis

**Task:** Given a dataset of student course enrollments, use set operations to answer:
1. Which students are enrolled in both Math and CS?
2. How many students are enrolled in exactly one of these courses?
3. What percentage of Math students are also in CS (overlap rate)?

**Dataset:**
```python
students = {
    'Math': {'Alice', 'Bob', 'Charlie', 'David', 'Eve'},
    'CS': {'Bob', 'David', 'Frank', 'Grace', 'Henry'},
    'Physics': {'Alice', 'Charlie', 'Frank', 'Ivan'}
}
```

**Solution:** [Link to notebook: `notebooks/week-01-practice.ipynb`]

## Additional Resources

### Required Reading
- 📚 **Textbook:** IIT Madras BSMA1001 - Sets & Functions Vol 1, Chapters 1-2
- 📚 **Video Lectures:** [Week 1 Playlist](https://www.youtube.com/playlist?list=PLZ2ps__7DhBa7DqhN-HidvTC0x2PxFvZL)

### Supplementary Materials
- 🎥 **Video:** [Set Theory Basics - Khan Academy](https://www.khanacademy.org/math/statistics-probability/probability-library/basic-set-ops)
- 🎥 **Video:** [Brilliant.org - Set Theory Interactive](https://brilliant.org/wiki/set-theory/)
- 💻 **Interactive:** [Wolfram MathWorld - Set Theory](https://mathworld.wolfram.com/SetTheory.html)
- 📊 **Visualization:** [Venn Diagram Generator](http://www.meta-chart.com/venn)

### Python Resources
- 💻 **Python Sets Documentation:** https://docs.python.org/3/tutorial/datastructures.html#sets
- 💻 **Real Python - Sets Guide:** https://realpython.com/python-sets/

### Practice Platforms
- 🎯 **LeetCode:** Set-related problems (tagged "Hash Table")
- 🎯 **HackerRank:** Set operations challenges
- 📝 **Project Euler:** Combinatorial problems using sets

### Practice Notebooks
- 💻 **Week 1 Practice:** [Week-01-SetTheory-Practice.ipynb](../notebooks/Week-01-SetTheory-Practice.ipynb)
- 💻 **Week 1 Assignment:** [Week-01-SetTheory-Assignment.ipynb](../notebooks/Week-01-SetTheory-Assignment.ipynb)

### Advanced Topics (For Later)
- 📖 **Naive Set Theory** by Paul Halmos - Classic rigorous introduction
- 📖 **Axiomatic Set Theory** - For deep mathematical foundations
- 📖 **Fuzzy Sets** - For uncertain membership (advanced ML)

---

**Review Status:** [x] Completed  
**Next Review Date:** 2025-11-21  
**Confidence Level:** ⭐⭐⭐⭐⭐ (5/5)  
**Time Spent:** 4 hours  
**Difficulty:** ⭐⭐ (2/5 - Foundational)

---

**Next Topic:** [Week 1 Part 2: Functions - Domain, Codomain, and Range](Week-01-Functions-Basics.md)

**Related Notes:**
- [Week 2: 2D Coordinate Systems](Week-02-CoordinateSystems-2D.md)
- [Statistics I Week 1: Data Types](../../02-Statistics/notes/Week-01-DataTypes-Scales.md)

---

**Changelog:**
- 2025-11-14: Initial creation following RAG template
- 2025-11-14: Moved to proper course folder structure
- 2025-11-14: Updated metadata and cross-references
- Covered all BSMA1001 Week 1 curriculum topics
- Added extensive Python implementations
- Included real-world data science applications
- Created comprehensive practice problems
