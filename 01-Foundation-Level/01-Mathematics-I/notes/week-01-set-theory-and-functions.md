# Week 01: Set Theory and Functions

**Course**: BSMA1001 - Mathematics I  
**Level**: Foundation

---

## 1. Number Systems

### 1.1 Theory

Number systems form the foundation of mathematics. We progress from natural numbers to complex numbers, each extension addressing limitations of the previous system.

### 1.2 Mathematical Definition

- **Natural Numbers** ($\mathbb{N}$): $\{1, 2, 3, ...\}$
- **Integers** ($\mathbb{Z}$): $\{..., -2, -1, 0, 1, 2, ...\}$
- **Rationals** ($\mathbb{Q}$): $\{\frac{p}{q} : p, q \in \mathbb{Z}, q \neq 0\}$
- **Reals** ($\mathbb{R}$): All points on the number line
- **Complex** ($\mathbb{C}$): $\{a + bi : a, b \in \mathbb{R}, i^2 = -1\}$

### 1.3 Key Properties

- **Number System Hierarchy**: $\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R} \subset \mathbb{C}$
- Each extension solves a limitation:
  - $\mathbb{N} \to \mathbb{Z}$: Allows subtraction (e.g., $3 - 5 = -2$)
  - $\mathbb{Z} \to \mathbb{Q}$: Allows division (e.g., $3 \div 5 = 0.6$)
  - $\mathbb{Q} \to \mathbb{R}$: Fills gaps (e.g., $\sqrt{2}, \pi$)
  - $\mathbb{R} \to \mathbb{C}$: Allows $\sqrt{-1} = i$

### 1.4 Supply Chain Application

**Retail Context**: Number systems are fundamental in inventory management:
- Natural numbers for item counts
- Integers for stock adjustments (returns as negative)
- Rationals for unit prices
- Reals for statistical calculations like average demand

---

## 2. Sets and Their Operations

### 2.1 Theory

A set is a well-defined collection of distinct objects. Set operations allow us to combine, compare, and manipulate collections in meaningful ways.

### 2.2 Mathematical Definition

- **Union**: $A \cup B = \{x : x \in A \text{ or } x \in B\}$
- **Intersection**: $A \cap B = \{x : x \in A \text{ and } x \in B\}$
- **Complement**: $A^c = \{x : x \notin A\}$
- **Difference**: $A - B = \{x : x \in A \text{ and } x \notin B\}$
- **Symmetric Difference**: $A \triangle B = (A - B) \cup (B - A)$

### 2.3 Important Properties

**De Morgan's Laws**:
- $(A \cup B)' = A' \cap B'$
- $(A \cap B)' = A' \cup B'$

**Inclusion-Exclusion Principle**:
$$|A \cup B| = |A| + |B| - |A \cap B|$$

For three sets:
$$|A \cup B \cup C| = |A| + |B| + |C| - |A \cap B| - |B \cap C| - |A \cap C| + |A \cap B \cap C|$$

### 2.4 Supply Chain Application

**Retail Context**: Sets are essential for product categorization:
- Finding products common to multiple categories (intersection)
- All products across warehouses (union)
- Products not in stock at a location (difference)

---

## 3. Relations and Their Types

### 3.1 Theory

A relation from set A to set B is a subset of the Cartesian product $A \times B$. Relations can have special properties that make them useful for modeling real-world connections.

### 3.2 Mathematical Definition

A relation $R$ on set $A$ is:

- **Reflexive**: $\forall a \in A: (a, a) \in R$
- **Symmetric**: $(a, b) \in R \Rightarrow (b, a) \in R$
- **Antisymmetric**: $(a, b) \in R \land (b, a) \in R \Rightarrow a = b$
- **Transitive**: $(a, b) \in R \land (b, c) \in R \Rightarrow (a, c) \in R$

### 3.3 Special Relation Types

| Relation Type | Reflexive | Symmetric | Antisymmetric | Transitive |
|---------------|-----------|-----------|---------------|------------|
| Equivalence   | ✓         | ✓         | ✗             | ✓          |
| Partial Order | ✓         | ✗         | ✓             | ✓          |
| Strict Order  | ✗         | ✗         | ✓             | ✓          |

### 3.4 Cartesian Product

For sets $X$ and $Y$:
$$X \times Y = \{(x, y) : x \in X, y \in Y\}$$
$$|X \times Y| = |X| \times |Y|$$

### 3.5 Supply Chain Application

**Retail Context**: Relations model connections in supply networks:
- "Supplies to" relation between suppliers and warehouses
- "Is substitute for" relation between products
- "Located in" relation between stores and regions

---

## 4. Functions and Their Types

### 4.1 Theory

A function is a special relation where each element in the domain maps to exactly one element in the codomain. Functions are classified by how they map elements.

### 4.2 Mathematical Definition

For function $f: A \rightarrow B$:

- **Injective (One-to-one)**: $f(a_1) = f(a_2) \Rightarrow a_1 = a_2$
  - No two different inputs map to the same output
  - Requires $|A| \leq |B|$

- **Surjective (Onto)**: $\forall b \in B, \exists a \in A: f(a) = b$
  - Every element in the codomain is mapped to
  - Requires $|A| \geq |B|$

- **Bijective**: Both injective and surjective
  - One-to-one correspondence between domain and codomain
  - Requires $|A| = |B|$
  - Has an inverse function $f^{-1}: B \rightarrow A$

### 4.3 Function Classification Summary

| Property        | Injective    | Surjective   | Bijective    |
|-----------------|--------------|--------------|--------------|
| Definition      | No repeated outputs | All codomain covered | Both |
| \|Domain\| vs \|Codomain\| | \|A\| ≤ \|B\| | \|A\| ≥ \|B\| | \|A\| = \|B\| |
| Inverse exists? | Left inverse | Right inverse | Full inverse |

### 4.4 Function Composition

For functions $f: A \to B$ and $g: B \to C$:
$$(g \circ f)(x) = g(f(x))$$

### 4.5 Supply Chain Application

**Retail Context**: Functions appear everywhere:
- Price-to-tier mapping (assigning products to price categories)
- SKU-to-warehouse assignment
- Demand transformation functions for forecasting

---

## Summary

1. **Number systems** build hierarchically: $\mathbb{N} \subset \mathbb{Z} \subset \mathbb{Q} \subset \mathbb{R} \subset \mathbb{C}$

2. **Set operations** (union, intersection, complement, difference) enable powerful data manipulation

3. **Relations** with special properties (reflexive, symmetric, transitive) model real-world connections

4. **Functions** are classified as injective, surjective, or bijective based on their mapping behavior

---

## Key Formulas

| Concept | Formula |
|---------|---------|
| Inclusion-Exclusion (2 sets) | $\|A \cup B\| = \|A\| + \|B\| - \|A \cap B\|$ |
| Cartesian Product Size | $\|A \times B\| = \|A\| \times \|B\|$ |
| De Morgan's Law 1 | $(A \cup B)' = A' \cap B'$ |
| De Morgan's Law 2 | $(A \cap B)' = A' \cup B'$ |

---

## Next Week Preview

Week 2 covers **Coordinate Systems and Straight Lines** - we'll explore the rectangular coordinate system, slopes, and line equations with applications in trend analysis.

---

*IIT Madras BS Degree in Data Science*
