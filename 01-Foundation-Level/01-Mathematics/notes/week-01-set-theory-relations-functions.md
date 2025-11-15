# Week 1: Set Theory, Relations, and Functions

**Date**: 2025-11-15  
**Course**: Mathematics for Data Science I (BSMA1001)

## Topics Covered

1. Number Systems
2. Sets and Set Operations
3. Relations and Their Types
4. Functions and Their Types

---

## Key Concepts

### 1. Number Systems

Understanding different types of numbers is fundamental to mathematics:

#### Natural Numbers (ℕ)
- Counting numbers: {1, 2, 3, 4, ...}
- Used for counting discrete objects
- Closed under addition and multiplication

#### Whole Numbers (W)
- Natural numbers plus zero: {0, 1, 2, 3, ...}
- Extension of natural numbers

#### Integers (ℤ)
- Whole numbers and their negatives: {..., -2, -1, 0, 1, 2, ...}
- Closed under addition, subtraction, and multiplication
- **Why important for DS**: Indexing, counting, discrete data

#### Rational Numbers (ℚ)
- Numbers that can be expressed as p/q where p, q ∈ ℤ and q ≠ 0
- Includes terminating and repeating decimals
- Examples: 1/2, 0.75, -3/4, 0.333...

#### Real Numbers (ℝ)
- All rational and irrational numbers
- Includes non-terminating, non-repeating decimals
- Examples: π, √2, e
- **Why important for DS**: Continuous data, measurements, probabilities

#### Complex Numbers (ℂ)
- Numbers of the form a + bi where i² = -1
- Examples: 3 + 4i, -2i
- **Why important for DS**: Signal processing, eigenvalues

### 2. Sets

A **set** is a well-defined collection of distinct objects.

#### Set Notation
- **Roster/Tabular form**: A = {1, 2, 3, 4, 5}
- **Set-builder form**: A = {x | x is a natural number, x ≤ 5}
- **Interval notation**: [a, b], (a, b), [a, b), (a, b]

#### Special Sets
- **Empty set (∅)**: Set with no elements
- **Universal set (U)**: Set containing all elements under consideration
- **Power set P(A)**: Set of all subsets of A
  - If |A| = n, then |P(A)| = 2ⁿ

#### Set Relations
- **Subset (⊆)**: A ⊆ B if every element of A is in B
- **Proper subset (⊂)**: A ⊂ B if A ⊆ B and A ≠ B
- **Superset (⊇)**: B ⊇ A if A ⊆ B
- **Equality**: A = B if A ⊆ B and B ⊆ A

### 3. Set Operations

#### Union (∪)
- A ∪ B = {x | x ∈ A or x ∈ B}
- Combines all elements from both sets
- **Example**: {1,2,3} ∪ {3,4,5} = {1,2,3,4,5}

#### Intersection (∩)
- A ∩ B = {x | x ∈ A and x ∈ B}
- Common elements only
- **Example**: {1,2,3} ∩ {3,4,5} = {3}

#### Difference (−)
- A − B = {x | x ∈ A and x ∉ B}
- Elements in A but not in B
- **Example**: {1,2,3} − {3,4,5} = {1,2}

#### Complement (Aᶜ or A')
- Aᶜ = U − A
- All elements not in A
- **Example**: If U = {1,2,3,4,5} and A = {1,2}, then Aᶜ = {3,4,5}

#### Symmetric Difference (△)
- A △ B = (A − B) ∪ (B − A)
- Elements in either set but not both
- **Example**: {1,2,3} △ {3,4,5} = {1,2,4,5}

### 4. Relations

A **relation** R from set A to set B is a subset of A × B (Cartesian product).

#### Cartesian Product
- A × B = {(a, b) | a ∈ A, b ∈ B}
- **Example**: If A = {1,2} and B = {x,y}, then A × B = {(1,x), (1,y), (2,x), (2,y)}

#### Types of Relations

**1. Reflexive Relation**
- R is reflexive on A if (a, a) ∈ R for all a ∈ A
- Every element is related to itself
- **Example**: "is equal to" (=)

**2. Symmetric Relation**
- R is symmetric if (a, b) ∈ R ⟹ (b, a) ∈ R
- **Example**: "is sibling of"

**3. Antisymmetric Relation**
- R is antisymmetric if (a, b) ∈ R and (b, a) ∈ R ⟹ a = b
- **Example**: "is less than or equal to" (≤)

**4. Transitive Relation**
- R is transitive if (a, b) ∈ R and (b, c) ∈ R ⟹ (a, c) ∈ R
- **Example**: "is ancestor of"

**5. Equivalence Relation**
- Reflexive + Symmetric + Transitive
- **Example**: "has the same remainder when divided by 5"
- Creates equivalence classes that partition the set

**6. Partial Order**
- Reflexive + Antisymmetric + Transitive
- **Example**: "divides" relation on integers

### 5. Functions

A **function** f from A to B is a relation where each element in A is related to exactly one element in B.

#### Function Notation
- f: A → B
- f(x) = y means x maps to y
- **Domain**: Set A (all input values)
- **Codomain**: Set B (all possible output values)
- **Range**: {f(x) | x ∈ A} ⊆ B (actual output values)

#### Types of Functions

**1. One-to-One (Injective)**
- Different inputs give different outputs
- f(x₁) = f(x₂) ⟹ x₁ = x₂
- **Example**: f(x) = 2x
- **Test**: Horizontal line test - no horizontal line intersects graph twice

**2. Onto (Surjective)**
- Every element in codomain is mapped to
- Range = Codomain
- **Example**: f: ℝ → ℝ, f(x) = x³

**3. Bijective (One-to-One and Onto)**
- Both injective and surjective
- Perfect pairing between domain and codomain
- **Example**: f(x) = 2x + 3
- Has an inverse function

**4. Identity Function**
- f(x) = x for all x
- Maps every element to itself

**5. Constant Function**
- f(x) = c for all x (c is constant)
- All inputs give same output

---

## Definitions

- **Set**: A well-defined collection of distinct objects called elements or members
- **Element (∈)**: An object belonging to a set. x ∈ A means "x is an element of A"
- **Cardinality (|A|)**: The number of elements in a set A
- **Subset**: A ⊆ B means every element of A is also in B
- **Proper Subset**: A ⊂ B means A ⊆ B but A ≠ B
- **Disjoint Sets**: Sets with no common elements (A ∩ B = ∅)
- **Partition**: A collection of non-empty, pairwise disjoint subsets whose union is the entire set
- **Relation**: A subset of the Cartesian product A × B
- **Function**: A relation where each input has exactly one output
- **Domain**: The set of all possible inputs for a function
- **Codomain**: The set that contains all possible outputs
- **Range**: The set of actual outputs {f(x) | x ∈ Domain}
- **Image**: The output f(a) for a given input a
- **Preimage**: The set of all inputs that map to a given output

---

## Important Formulas

### Set Cardinality
- **Inclusion-Exclusion Principle**: 
  ```
  |A ∪ B| = |A| + |B| − |A ∩ B|
  ```
  
- **For three sets**:
  ```
  |A ∪ B ∪ C| = |A| + |B| + |C| − |A ∩ B| − |A ∩ C| − |B ∩ C| + |A ∩ B ∩ C|
  ```

- **Power set size**: If |A| = n, then |P(A)| = 2ⁿ

- **Cartesian product size**: |A × B| = |A| · |B|

### De Morgan's Laws
```
(A ∪ B)ᶜ = Aᶜ ∩ Bᶜ
(A ∩ B)ᶜ = Aᶜ ∪ Bᶜ
```

### Set Identities
- **Idempotent**: A ∪ A = A, A ∩ A = A
- **Identity**: A ∪ ∅ = A, A ∩ U = A
- **Domination**: A ∪ U = U, A ∩ ∅ = ∅
- **Commutative**: A ∪ B = B ∪ A, A ∩ B = B ∩ A
- **Associative**: (A ∪ B) ∪ C = A ∪ (B ∪ C)
- **Distributive**: A ∪ (B ∩ C) = (A ∪ B) ∩ (A ∪ C)
- **Complement**: A ∪ Aᶜ = U, A ∩ Aᶜ = ∅

---

## Theorems & Proofs

### Theorem 1: Size of Power Set
**Statement**: If a set A has n elements, then P(A) has 2ⁿ elements.

**Proof**: 
For each element in A, we have 2 choices when forming a subset:
1. Include the element in the subset
2. Don't include the element

Since we make this choice independently for each of the n elements, the total number of subsets is 2 × 2 × ... × 2 (n times) = 2ⁿ.

**Example**: If A = {a, b, c}, then |A| = 3 and |P(A)| = 2³ = 8
P(A) = {∅, {a}, {b}, {c}, {a,b}, {a,c}, {b,c}, {a,b,c}}

### Theorem 2: Properties of Equivalence Relations
**Statement**: An equivalence relation on a set A partitions A into disjoint equivalence classes.

**Key Points**:
- Each element belongs to exactly one equivalence class
- Two equivalence classes are either identical or disjoint
- The union of all equivalence classes equals A

---

## Examples (Worked Problems)

### Example 1: Set Operations
**Problem**: Let U = {1,2,3,4,5,6,7,8,9,10}, A = {1,2,3,4,5}, B = {4,5,6,7}
Find: (a) A ∪ B, (b) A ∩ B, (c) A − B, (d) Aᶜ, (e) A △ B

**Solution**: 
- (a) A ∪ B = {1,2,3,4,5,6,7}
- (b) A ∩ B = {4,5}
- (c) A − B = {1,2,3}
- (d) Aᶜ = {6,7,8,9,10}
- (e) A △ B = (A − B) ∪ (B − A) = {1,2,3} ∪ {6,7} = {1,2,3,6,7}

### Example 2: Cardinality with Inclusion-Exclusion
**Problem**: In a class of 50 students, 30 study Python, 25 study SQL, and 15 study both. How many students study at least one of these languages?

**Solution**: 
Let P = set of students studying Python, S = set of students studying SQL
- |P| = 30
- |S| = 25
- |P ∩ S| = 15
- |P ∪ S| = |P| + |S| − |P ∩ S| = 30 + 25 − 15 = 40

**Answer**: 40 students study at least one language.

### Example 3: Power Set
**Problem**: Find P(A) where A = {x, y}

**Solution**: 
P(A) = {∅, {x}, {y}, {x,y}}

Number of elements: |P(A)| = 2² = 4 ✓

### Example 4: Checking Relation Properties
**Problem**: Let A = {1,2,3} and R = {(1,1), (2,2), (3,3), (1,2), (2,1)}
Check if R is reflexive, symmetric, transitive, and equivalence.

**Solution**:
- **Reflexive?** Yes! All pairs (1,1), (2,2), (3,3) are present
- **Symmetric?** Yes! If (a,b) ∈ R then (b,a) ∈ R
  - (1,2) ∈ R and (2,1) ∈ R ✓
- **Transitive?** Yes! 
  - (1,2) ∈ R and (2,1) ∈ R ⟹ (1,1) ∈ R ✓
  - All other transitive requirements are satisfied
- **Equivalence?** Yes! (reflexive + symmetric + transitive)

### Example 5: Function Types
**Problem**: Determine if f: ℝ → ℝ, f(x) = x² is injective, surjective, or bijective.

**Solution**:
- **Injective?** NO! f(2) = 4 and f(−2) = 4
  Different inputs (2 and −2) give the same output
  
- **Surjective?** NO! Range = [0, ∞) ≠ Codomain (ℝ)
  Negative numbers are never outputs
  
- **Bijective?** NO! (not injective AND not surjective)

**Note**: If we change the function to f: ℝ → [0, ∞), f(x) = x², it becomes surjective but still not injective.
If we further restrict to f: [0, ∞) → [0, ∞), f(x) = x², it becomes bijective!

### Example 6: Equivalence Classes
**Problem**: Define relation R on ℤ by: a R b if a − b is divisible by 3.
Show R is an equivalence relation and find equivalence classes.

**Solution**:
**Reflexive**: a − a = 0, divisible by 3 ✓

**Symmetric**: If a − b divisible by 3, then b − a = −(a − b) is also divisible by 3 ✓

**Transitive**: If (a − b) and (b − c) are divisible by 3, then:
(a − c) = (a − b) + (b − c) is also divisible by 3 ✓

**Equivalence classes** (partition ℤ into 3 classes):
- [0] = {..., −6, −3, 0, 3, 6, 9, ...} (numbers ≡ 0 mod 3)
- [1] = {..., −5, −2, 1, 4, 7, 10, ...} (numbers ≡ 1 mod 3)
- [2] = {..., −4, −1, 2, 5, 8, 11, ...} (numbers ≡ 2 mod 3)

--- 

---

## Data Science Applications

### Why Set Theory Matters in Data Science

1. **Data Filtering & Selection**
   - Union: Combining datasets from different sources
   - Intersection: Finding common records between datasets
   - Difference: Identifying unique records
   
   ```python
   # Example: User analysis
   active_users = {user_id for user_id in users if last_active < 30_days}
   premium_users = {user_id for user_id in users if has_subscription}
   
   # Users who are active AND premium
   engaged_premium = active_users ∩ premium_users
   
   # Users who are premium but NOT active (churn risk!)
   inactive_premium = premium_users − active_users
   ```

2. **Database Operations**
   - SQL JOIN operations use set theory
   - UNION, INTERSECT, EXCEPT are set operations
   - Venn diagrams help visualize JOIN types

3. **Feature Engineering**
   - Creating categorical variables using set membership
   - One-hot encoding represents sets as binary vectors
   - Checking if values belong to valid categories

4. **Relations in Data**
   - Database relationships: one-to-one, one-to-many, many-to-many
   - Foreign keys represent relations between tables
   - Graph databases model complex relations

5. **Functions in Machine Learning**
   - ML models are functions: f: Features → Predictions
   - Loss functions: f: (Predictions, ActualValues) → Error
   - Activation functions in neural networks
   - Checking if function is injective (no information loss)

### Real-World Example: Customer Segmentation

```
Dataset: E-commerce customers
A = {customers who purchased in last month}
B = {customers who clicked on email}
C = {customers with loyalty points > 1000}

Segments:
- High-value engaged: A ∩ B ∩ C
- Potential churn: C − A (have points but no recent purchase)
- Email-responsive: B − A (clicked but didn't buy)
- Inactive: Aᶜ ∩ Bᶜ ∩ Cᶜ
```

---

## Practice Problems

### Basic Level
1. **Set Operations**: Let A = {1,3,5,7}, B = {2,3,5,8}. Find:
   - (a) A ∪ B
   - (b) A ∩ B
   - (c) A − B
   - (d) B − A
   - (e) A △ B (symmetric difference)

2. **Inclusion-Exclusion**: If |A| = 20, |B| = 15, |A ∪ B| = 28, find |A ∩ B|

3. **Power Sets**: List all subsets of {a, b, c}. Verify |P(A)| = 2³

4. **Cardinality**: Find |P({1,2,3,4})| without listing all elements

5. **Complement**: If U = {1,2,3,4,5,6,7,8,9,10} and A = {2,4,6,8}, find Aᶜ

### Intermediate Level

6. **Distributive Law**: Prove that A ∩ (B ∪ C) = (A ∩ B) ∪ (A ∩ C) using:
   - (a) Venn diagrams
   - (b) Element argument (show x ∈ LHS ⟺ x ∈ RHS)

7. **Relation Properties**: Let R = {(1,1), (2,2), (3,3), (1,2), (2,3), (1,3)} on A = {1,2,3}
   - Check if R is reflexive
   - Check if R is symmetric
   - Check if R is transitive
   - Is R an equivalence relation? Why or why not?

8. **Function Analysis**: Determine if f(x) = 3x − 5 is injective, surjective, bijective for f: ℝ → ℝ
   - Prove your answer for injectivity
   - Prove your answer for surjectivity
   - Find the inverse function if it exists

9. **Survey Problem**: In a survey of 100 people:
   - 60 like tea
   - 50 like coffee
   - 30 like both
   - How many like neither?
   - How many like only tea?
   - How many like only coffee?

10. **Cartesian Product**: If A = {1,2}, B = {x,y,z}, find:
    - A × B
    - B × A
    - |A × B|
    - Is A × B = B × A?

### Advanced Level

11. **De Morgan's Laws**: Prove (A ∪ B)ᶜ = Aᶜ ∩ Bᶜ using element argument

12. **Equivalence Relation on ℤ×ℤ**: Define relation R on ℤ×ℤ by: 
    (a,b) R (c,d) if a+d = b+c
    - Prove R is reflexive
    - Prove R is symmetric
    - Prove R is transitive
    - Describe the equivalence classes

13. **Composition of Functions**: For f: A → B and g: B → C, prove:
    If f and g are injective, then g∘f is injective

14. **Function Construction**: Give an example of a function that is:
    - (a) Injective but not surjective
    - (b) Surjective but not injective
    - (c) Neither injective nor surjective
    - (d) Bijective

15. **Modular Arithmetic**: Define R on ℤ by: a R b if (a − b) is divisible by 5
    - Prove R is an equivalence relation
    - Find all equivalence classes
    - How many equivalence classes exist?

### Challenge Problems

16. **Three Sets**: In a college of 200 students:
    - 120 study Python
    - 90 study SQL
    - 80 study R
    - 50 study both Python and SQL
    - 40 study both Python and R
    - 35 study both SQL and R
    - 20 study all three
    How many students study:
    - (a) At least one language?
    - (b) Exactly one language?
    - (c) None of the languages?

17. **Function Properties**: Prove that if f: A → B is bijective, then f has an inverse function f⁻¹: B → A that is also bijective

18. **Partition**: Let A = {1,2,3,4,5,6,7,8,9,10}. Create a partition of A into equivalence classes using the relation "has the same remainder when divided by 3"

---

## Questions/Doubts

- [ ] What's the difference between codomain and range?
- [ ] Why do we need equivalence relations in mathematics?
- [ ] How do I prove a function is bijective?
- [ ] What's the intuition behind power sets?
- [ ] How are relations used in databases?

---

## Action Items

- [x] Review lecture slides on set theory
- [x] Complete practice problems 1-4
- [ ] Work through notebook examples (Week 1 Practice)
- [ ] Watch 3Blue1Brown video on functions
- [ ] Solve textbook exercises: Chapter 1, problems 1-20
- [ ] Create flashcards for key definitions
- [ ] Draw Venn diagrams for 3-set problems

---

## Key Takeaways

1. **Sets are fundamental**: Almost everything in mathematics can be described using sets
2. **Relations connect elements**: Understanding relations helps model real-world connections
3. **Functions are special relations**: Each input maps to exactly one output
4. **Equivalence relations partition sets**: They group elements into distinct classes
5. **Set operations mirror logical operations**: ∪ is OR, ∩ is AND, complement is NOT
6. **Cardinality counting is crucial**: Inclusion-exclusion principle solves many problems

---

## References

- **Textbook**: 
  - Rosen, K.H. - *Discrete Mathematics*, Chapter 2 (Sets) & Chapter 9 (Relations)
- **Video Lectures**: 
  - IIT Madras Week 1 lectures
  - [3Blue1Brown - Essence of Linear Algebra](https://www.youtube.com/watch?v=fNk_zzaMoSs)
- **Practice**: 
  - Week 1 Practice Notebook
  - Khan Academy: Set Theory basics

---

## Connection to Next Week

Week 2 will build on functions by exploring:
- Coordinate systems (ordered pairs from Cartesian product!)
- Linear functions (special type of function)
- Graphical representation of relations

The foundation you're building with sets and functions is crucial for understanding coordinate geometry!

---

**Last Updated**: 2025-11-15  
**Next Class**: Week 2 - Coordinate Systems and Straight Lines
