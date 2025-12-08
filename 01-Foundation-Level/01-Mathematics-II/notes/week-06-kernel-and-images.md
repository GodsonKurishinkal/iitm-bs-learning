# Week 06: Kernel and Images

**Course**: BSMA1003 - Mathematics II
**Level**: Foundation

---

## Learning Objectives
- Understand the kernel of a linear transformation
- Learn about the image (range) of a linear transformation
- Master the concepts of injective, surjective, and bijective transformations

---

## 1. Kernel of Linear Transformations

### 1.1 Theory

The **kernel** is the set of all inputs that map to the zero vector. It measures what information is "lost" by the transformation.

### 1.2 Mathematical Definition

$$\ker(T) = \{\mathbf{v} \in V : T(\mathbf{v}) = \mathbf{0}\}$$

For matrix $A$: $\ker(A) = \text{null}(A)$

### 1.3 Key Properties

| Property | Description |
|----------|-------------|
| **Subspace** | $\ker(T)$ is a subspace of the domain $V$ |
| **Contains zero** | $\mathbf{0} \in \ker(T)$ always |
| **Injectivity test** | $T$ is injective $\iff \ker(T) = \{\mathbf{0}\}$ |
| **Dimension** | $\dim(\ker(T))$ = nullity |

### 1.4 Supply Chain Application

**Retail Context**: In **dimensionality reduction** (like PCA), the kernel represents information lost. Understanding the kernel helps assess what features become indistinguishable after transformation.

---

## 2. Image (Range) of Linear Transformations

### 2.1 Theory

The **image** (or range) is the set of all possible outputs of the transformation. It's a subspace of the codomain.

### 2.2 Mathematical Definition

$$\text{im}(T) = \{T(\mathbf{v}) : \mathbf{v} \in V\}$$

For matrix $A$: $\text{im}(A) = \text{col}(A)$ (column space)

### 2.3 Key Properties

| Property | Description |
|----------|-------------|
| **Subspace** | $\text{im}(T)$ is a subspace of the codomain $W$ |
| **Surjectivity test** | $T$ is surjective $\iff \text{im}(T) = W$ |
| **Dimension** | $\dim(\text{im}(T))$ = rank |
| **Rank-Nullity** | $\dim(\ker(T)) + \dim(\text{im}(T)) = \dim(V)$ |

### 2.4 Supply Chain Application

**Retail Context**: The image represents all **reachable states**. In production planning, it shows what outputs are achievable given the constraints and resources.

---

## 3. Injective, Surjective, Bijective

### 3.1 Theory

These properties describe whether a transformation preserves information and covers the target space.

### 3.2 Definitions

| Property | Definition | Kernel/Image Condition |
|----------|------------|------------------------|
| **Injective** (1-to-1) | $T(\mathbf{u}) = T(\mathbf{v}) \Rightarrow \mathbf{u} = \mathbf{v}$ | $\ker(T) = \{\mathbf{0}\}$ |
| **Surjective** (onto) | Every $\mathbf{w} \in W$ is achieved | $\text{im}(T) = W$ |
| **Bijective** | Both injective and surjective | Invertible |

### 3.3 Matrix Conditions

For $A \in \mathbb{R}^{m \times n}$:

| Property | Matrix Condition |
|----------|-----------------|
| **Injective** | $\text{rank}(A) = n$ (full column rank) |
| **Surjective** | $\text{rank}(A) = m$ (full row rank) |
| **Bijective** | $m = n$ and $\text{rank}(A) = n$ (invertible) |

### 3.4 Dimension Constraints

| Transformation $T: V \to W$ | Requirement |
|-----------------------------|-------------|
| Can be injective | $\dim(V) \leq \dim(W)$ |
| Can be surjective | $\dim(V) \geq \dim(W)$ |
| Can be bijective | $\dim(V) = \dim(W)$ |

---

## 4. First Isomorphism Theorem

### 4.1 Statement

For linear transformation $T: V \to W$:

$$V / \ker(T) \cong \text{im}(T)$$

The domain modulo the kernel is isomorphic to the image.

### 4.2 Practical Meaning

- The "effective" input space has dimension = rank
- Inputs that differ only by kernel elements produce the same output

---

## Summary Table

| Concept | Definition | Key Property | Supply Chain Application |
|---------|------------|--------------|--------------------------|
| **Kernel** | $\{\mathbf{v} : T(\mathbf{v}) = \mathbf{0}\}$ | Measures info loss | Dimensionality reduction |
| **Image** | $\{T(\mathbf{v}) : \mathbf{v} \in V\}$ | Measures coverage | Reachable states |
| **Injective** | No two inputs map to same output | Trivial kernel | Lossless transformation |
| **Surjective** | All outputs are achieved | Full image | Complete coverage |
| **Bijective** | Invertible | Both properties | Reversible mapping |

---

## Key Takeaways

1. **Kernel** = null space = what gets mapped to zero → measures information loss
2. **Image** = column space = what can be reached → measures output coverage
3. **Injective** means no information loss (only zero maps to zero)
4. **Bijective** transformations are invertible - they're the "perfect" transformations

---

## Next Week Preview

Week 7 covers **Similar Matrices and Inner Products** - matrix equivalence and geometric structure.

---

*IIT Madras BS Degree in Data Science*
