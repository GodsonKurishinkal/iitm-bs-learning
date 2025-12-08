# Week 04: Basis and Dimension

**Course**: BSMA1003 - Mathematics II
**Level**: Foundation

---

## Learning Objectives
- Understand the concept of a basis for a vector space
- Learn how to find a basis for column and row spaces
- Master dimension and rank computation

---

## 1. Basis of a Vector Space

### 1.1 Theory

A **basis** is a linearly independent set that spans the entire vector space. Every vector in the space can be **uniquely** expressed as a linear combination of basis vectors.

### 1.2 Mathematical Definition

A set $\mathcal{B} = \{\mathbf{v}_1, \mathbf{v}_2, ..., \mathbf{v}_n\}$ is a **basis** for vector space $V$ if:

1. $\mathcal{B}$ is **linearly independent**
2. $\text{span}(\mathcal{B}) = V$ (spans the entire space)

### 1.3 Key Properties

| Property | Description |
|----------|-------------|
| **Uniqueness of Representation** | Every $\mathbf{v} \in V$ has exactly one representation as $\sum c_i \mathbf{v}_i$ |
| **Minimality** | A basis is a minimal spanning set |
| **Maximality** | A basis is a maximal independent set |
| **Same Size** | All bases for $V$ have the same number of vectors |

### 1.4 Standard Bases

| Vector Space | Standard Basis |
|--------------|---------------|
| $\mathbb{R}^2$ | $\{(1,0), (0,1)\}$ |
| $\mathbb{R}^3$ | $\{(1,0,0), (0,1,0), (0,0,1)\}$ |
| $\mathbb{R}^n$ | $\{\mathbf{e}_1, \mathbf{e}_2, ..., \mathbf{e}_n\}$ |
| $P_2$ (polynomials) | $\{1, x, x^2\}$ |

### 1.5 Supply Chain Application

**Retail Context**: In **PCA for demand forecasting**, principal components form a basis for the transformed feature space, capturing maximum variance with minimum dimensions.

---

## 2. Dimension

### 2.1 Theory

**Dimension** is the number of vectors in any basis of a vector space. It measures the "degrees of freedom" in the space.

### 2.2 Mathematical Definition

$$\dim(V) = |\mathcal{B}|$$

where $\mathcal{B}$ is any basis for $V$.

### 2.3 Key Properties

| Property | Formula/Description |
|----------|---------------------|
| $\dim(\mathbb{R}^n)$ | $n$ |
| $\dim(M_{m \times n})$ | $m \times n$ |
| $\dim(P_n)$ | $n + 1$ |
| Subspace dimension | $\dim(W) \leq \dim(V)$ for $W \subseteq V$ |

---

## 3. Rank

### 3.1 Theory

**Matrix rank** is the dimension of its column space (equivalently, row space). It indicates how many linearly independent rows or columns the matrix has.

### 3.2 Mathematical Definition

$$\text{rank}(A) = \dim(\text{col}(A)) = \dim(\text{row}(A))$$

### 3.3 Computing Rank

| Method | Approach |
|--------|----------|
| **RREF** | Count number of pivot columns |
| **Column Space** | Find basis for column space, count vectors |
| **Row Space** | Find basis for row space, count vectors |

### 3.4 Key Properties

| Property | Description |
|----------|-------------|
| $\text{rank}(A) \leq \min(m, n)$ | For $A \in \mathbb{R}^{m \times n}$ |
| $\text{rank}(A) = \text{rank}(A^T)$ | Row rank equals column rank |
| $\text{rank}(AB) \leq \min(\text{rank}(A), \text{rank}(B))$ | Rank of product |
| Full rank | $\text{rank}(A) = \min(m, n)$ |

### 3.5 Supply Chain Application

**Retail Context**:
- Matrix rank indicates the **effective number of independent constraints** or features
- **Low rank** in a demand matrix might indicate similar products that could be grouped
- Rank reveals the **true dimensionality** of data for compression

---

## Summary Table

| Concept | Definition | Key Property | Supply Chain Application |
|---------|------------|--------------|--------------------------|
| **Basis** | Independent spanning set | Unique representation | PCA components |
| **Dimension** | Size of any basis | Degrees of freedom | Feature space size |
| **Rank** | Dimension of column/row space | From pivot count | Effective feature count |

---

## Key Takeaways

1. **Basis** = linearly independent + spanning - the minimal complete description of a space
2. **Dimension** is unique for each vector space - all bases have the same size
3. **Rank** is found by counting pivots in RREF - reveals true degrees of freedom
4. These concepts are **fundamental to dimensionality reduction** in data science

---

## Next Week Preview

Week 5 covers **Rank, Nullity, and Linear Transformations** - the relationship between these concepts and the Rank-Nullity Theorem.

---

*IIT Madras BS Degree in Data Science*
