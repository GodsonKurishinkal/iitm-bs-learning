---
Date: 2025-11-21
Course: BSMA1001 - Mathematics for Data Science I
Level: Foundation
Week: 5 of 12
Source: IIT Madras Mathematics for Data Science I Week 5
Topic Area: Mathematics
Tags: #BSMA1001 #Functions #DomainRange #Composition #Week5 #Foundation
---

# Week 5: Functions - Deep Dive

## Overview

Functions are one of the most fundamental concepts in mathematics and computer science. They form the basis for understanding relationships between variables, modeling real-world phenomena, and implementing algorithms. This week provides a comprehensive exploration of functions, their properties, operations, and applications in data science.

**Learning Objectives:**
- Master the formal definition of functions
- Understand domain, codomain, and range
- Work with function composition and inverses
- Classify functions by their properties
- Apply function concepts to data transformations
- Visualize and analyze function behavior

**Key Concepts:** Function definition, domain, range, one-to-one, onto, bijection, composition, inverse functions, transformations

---

## 1. Formal Definition of Functions

### 1.1 Mathematical Definition

**Definition:** A **function** $f$ from set $A$ to set $B$ is a rule that assigns to each element $x \in A$ exactly one element $y \in B$.

**Notation:** $f: A \to B$ or $y = f(x)$

**Components:**
- **Domain**: Set $A$ (all possible input values)
- **Codomain**: Set $B$ (set containing all possible outputs)
- **Range**: Subset of $B$ consisting of actual output values
- **Image of $x$**: The value $f(x)$ assigned to $x$

**Key Property:** Each input has **exactly one** output (uniqueness).

**Example 1.1: Function vs Non-Function**

Consider these mappings from $\{1, 2, 3\}$ to $\{a, b, c\}$:

**Mapping 1** (✓ Function):
- $1 \to a$
- $2 \to b$
- $3 \to a$

Each input has exactly one output. ✓

**Mapping 2** (✗ Not a Function):
- $1 \to a$
- $2 \to b, c$
- $3 \to a$

Element 2 maps to two outputs. ✗

### 1.2 Function Notation and Terminology

**Standard Notations:**
- $f(x)$: Function notation ("f of x")
- $x$: Independent variable (input)
- $y$ or $f(x)$: Dependent variable (output)
- $x \mapsto f(x)$: Mapping notation

**Example 1.2: Various Notations**

The function "square the input" can be written as:
1. $f(x) = x^2$
2. $x \mapsto x^2$
3. $f: \mathbb{R} \to \mathbb{R}$, $f(x) = x^2$
4. The function that maps each number to its square

All represent the same function.

---

## 2. Domain, Codomain, and Range

### 2.1 Domain

**Definition:** The **domain** of a function $f$ is the set of all possible input values for which $f$ is defined.

**Finding Domain:**
1. Identify restrictions on $x$
2. Exclude values that cause:
   - Division by zero
   - Square root of negative numbers (for real functions)
   - Logarithm of non-positive numbers
   - Other undefined operations

**Example 2.1: Finding Domains**

**Function 1:** $f(x) = \frac{1}{x-3}$

Division by zero when $x = 3$.

**Domain:** $\{x \in \mathbb{R} : x \neq 3\}$ or $(-\infty, 3) \cup (3, \infty)$

**Function 2:** $g(x) = \sqrt{x - 2}$

Square root requires non-negative argument: $x - 2 \geq 0$

**Domain:** $\{x \in \mathbb{R} : x \geq 2\}$ or $[2, \infty)$

**Function 3:** $h(x) = \frac{\sqrt{x}}{x-1}$

Requires: $x \geq 0$ (square root) AND $x \neq 1$ (division by zero)

**Domain:** $[0, 1) \cup (1, \infty)$

### 2.2 Range

**Definition:** The **range** of a function $f: A \to B$ is the set of all actual output values:

$$\text{Range}(f) = \{f(x) : x \in A\} = \{y \in B : y = f(x) \text{ for some } x \in A\}$$

**Finding Range:**
1. Analyze function behavior
2. Consider extreme values
3. Solve $y = f(x)$ for $x$ in terms of $y$
4. Determine which $y$ values are achievable

**Example 2.2: Finding Ranges**

**Function 1:** $f(x) = x^2$ with domain $\mathbb{R}$

Since $x^2 \geq 0$ for all real $x$, and every non-negative number is a square:

**Range:** $[0, \infty)$

**Function 2:** $g(x) = \frac{1}{x}$ with domain $\mathbb{R} \setminus \{0\}$

As $x \to 0^+$, $g(x) \to +\infty$; as $x \to 0^-$, $g(x) \to -\infty$

As $|x| \to \infty$, $g(x) \to 0$ (but never equals 0)

**Range:** $\mathbb{R} \setminus \{0\}$ or $(-\infty, 0) \cup (0, \infty)$

**Function 3:** $h(x) = 2\sin(x) + 1$

Since $-1 \leq \sin(x) \leq 1$:
$$-2 \leq 2\sin(x) \leq 2$$
$$-1 \leq 2\sin(x) + 1 \leq 3$$

**Range:** $[-1, 3]$

### 2.3 Codomain vs Range

**Important Distinction:**
- **Codomain**: The set we specify as possible outputs (part of definition)
- **Range**: The actual outputs achieved (derived from function)

**Relationship:** Range $\subseteq$ Codomain (always)

**Example 2.3: Codomain vs Range**

Consider $f: \mathbb{R} \to \mathbb{R}$ defined by $f(x) = x^2$

- **Domain**: $\mathbb{R}$ (all real numbers)
- **Codomain**: $\mathbb{R}$ (by definition)
- **Range**: $[0, \infty)$ (only non-negative reals)

Here, Range $\subsetneq$ Codomain (proper subset).

---

## 3. Types of Functions

### 3.1 One-to-One (Injective) Functions

**Definition:** A function $f: A \to B$ is **one-to-one** (or **injective**) if different inputs produce different outputs:

$$f(x_1) = f(x_2) \implies x_1 = x_2$$

Equivalently: $x_1 \neq x_2 \implies f(x_1) \neq f(x_2)$

**Graphical Test:** Horizontal Line Test
- If any horizontal line intersects the graph at most once, the function is one-to-one.

**Example 3.1: Testing One-to-One Property**

**Function 1:** $f(x) = 2x + 3$

Test: Assume $f(x_1) = f(x_2)$
$$2x_1 + 3 = 2x_2 + 3$$
$$2x_1 = 2x_2$$
$$x_1 = x_2$$

Therefore, $f$ is one-to-one. ✓

**Function 2:** $g(x) = x^2$

Counterexample: $g(2) = 4$ and $g(-2) = 4$

Since $2 \neq -2$ but $g(2) = g(-2)$, the function is NOT one-to-one. ✗

### 3.2 Onto (Surjective) Functions

**Definition:** A function $f: A \to B$ is **onto** (or **surjective**) if every element in $B$ is the image of at least one element in $A$:

$$\forall y \in B, \exists x \in A \text{ such that } f(x) = y$$

In other words: Range$(f) = B$ (range equals codomain)

**Example 3.2: Testing Onto Property**

**Function 1:** $f: \mathbb{R} \to \mathbb{R}$, $f(x) = 2x + 3$

For any $y \in \mathbb{R}$, we can solve for $x$:
$$y = 2x + 3$$
$$x = \frac{y - 3}{2}$$

Since we can find $x$ for every $y$, the function is onto. ✓

**Function 2:** $g: \mathbb{R} \to \mathbb{R}$, $g(x) = x^2$

Counterexample: There's no real $x$ such that $g(x) = -1$

The function is NOT onto $\mathbb{R}$. ✗

(However, $g: \mathbb{R} \to [0, \infty)$ defined by $g(x) = x^2$ IS onto.)

### 3.3 Bijective Functions

**Definition:** A function is **bijective** (or a **bijection**) if it is both one-to-one AND onto.

**Properties:**
- Each element in codomain has exactly one pre-image
- Perfect pairing between domain and codomain
- Invertible (has an inverse function)

**Example 3.3: Bijective Function**

$f: \mathbb{R} \to \mathbb{R}$ defined by $f(x) = 2x + 3$

**One-to-one?** Yes (shown in Example 3.1)
**Onto?** Yes (shown in Example 3.2)

Therefore, $f$ is bijective. ✓

### 3.4 Summary Table

| Type | Definition | Example | Non-Example |
|------|-----------|---------|-------------|
| **One-to-One** | Different inputs → different outputs | $f(x) = 2x$ | $f(x) = x^2$ |
| **Onto** | Every output is achieved | $f: \mathbb{R} \to \mathbb{R}$, $f(x) = x$ | $f: \mathbb{R} \to \mathbb{R}$, $f(x) = x^2$ |
| **Bijective** | One-to-one AND onto | $f(x) = 2x + 3$ | $f(x) = x^2$ |

---

## 4. Function Composition

### 4.1 Definition

**Definition:** The **composition** of functions $f$ and $g$, denoted $(f \circ g)$ or $f(g(x))$, is defined by:

$$(f \circ g)(x) = f(g(x))$$

**Requirements:**
- Range of $g$ must overlap with domain of $f$
- Apply $g$ first, then $f$

**Order Matters:** Generally, $f \circ g \neq g \circ f$

**Example 4.1: Computing Compositions**

Let $f(x) = x^2$ and $g(x) = x + 1$

**Compute $(f \circ g)(x)$:**
$$(f \circ g)(x) = f(g(x)) = f(x + 1) = (x + 1)^2 = x^2 + 2x + 1$$

**Compute $(g \circ f)(x)$:**
$$(g \circ f)(x) = g(f(x)) = g(x^2) = x^2 + 1$$

**Observation:** $(f \circ g)(x) = x^2 + 2x + 1 \neq x^2 + 1 = (g \circ f)(x)$

Composition is NOT commutative!

### 4.2 Properties of Composition

**Associativity:** $(f \circ g) \circ h = f \circ (g \circ h)$

**Identity Function:** $I(x) = x$ satisfies:
- $(f \circ I)(x) = f(x)$
- $(I \circ f)(x) = f(x)$

**Example 4.2: Three-Function Composition**

Let $f(x) = 2x$, $g(x) = x + 3$, $h(x) = x^2$

**Compute $(f \circ g \circ h)(x)$:**

Method 1: $(f \circ g) \circ h$
$$(f \circ g)(x) = f(g(x)) = f(x + 3) = 2(x + 3) = 2x + 6$$
$$((f \circ g) \circ h)(x) = (f \circ g)(h(x)) = 2x^2 + 6$$

Method 2: $f \circ (g \circ h)$
$$(g \circ h)(x) = g(h(x)) = g(x^2) = x^2 + 3$$
$$(f \circ (g \circ h))(x) = f(x^2 + 3) = 2(x^2 + 3) = 2x^2 + 6$$

Both methods give the same result! ✓

### 4.3 Domain of Composite Functions

**Finding Domain of $f \circ g$:**
1. Find domain of $g$
2. Ensure $g(x)$ is in domain of $f$
3. Intersect these constraints

**Example 4.3: Domain of Composition**

Let $f(x) = \sqrt{x}$ (domain: $[0, \infty)$) and $g(x) = x - 4$ (domain: $\mathbb{R}$)

**Find domain of $(f \circ g)(x) = f(g(x)) = \sqrt{x - 4}$:**

1. Domain of $g$: $\mathbb{R}$ ✓
2. Require $g(x) \geq 0$: $x - 4 \geq 0 \implies x \geq 4$

**Domain of $f \circ g$:** $[4, \infty)$

---

## 5. Inverse Functions

### 5.1 Definition

**Definition:** Let $f: A \to B$ be a bijection. The **inverse function** $f^{-1}: B \to A$ satisfies:

$$f^{-1}(f(x)) = x \text{ for all } x \in A$$
$$f(f^{-1}(y)) = y \text{ for all } y \in B$$

**Key Facts:**
- Only bijective functions have inverses
- $f^{-1}$ "undoes" what $f$ does
- Graph of $f^{-1}$ is reflection of $f$ across line $y = x$

### 5.2 Finding Inverse Functions

**Procedure:**
1. Verify $f$ is bijective (one-to-one and onto)
2. Write $y = f(x)$
3. Solve for $x$ in terms of $y$
4. Interchange $x$ and $y$ (or write result as $f^{-1}(x)$)

**Example 5.1: Finding an Inverse**

Find the inverse of $f(x) = 2x + 3$.

**Step 1:** Verify bijective
- One-to-one: ✓ (linear with non-zero slope)
- Onto $\mathbb{R}$: ✓

**Step 2:** Write $y = f(x)$
$$y = 2x + 3$$

**Step 3:** Solve for $x$
$$y - 3 = 2x$$
$$x = \frac{y - 3}{2}$$

**Step 4:** Interchange variables
$$f^{-1}(x) = \frac{x - 3}{2}$$

**Verification:**
$$f(f^{-1}(x)) = f\left(\frac{x-3}{2}\right) = 2\left(\frac{x-3}{2}\right) + 3 = x - 3 + 3 = x$$ ✓

$$f^{-1}(f(x)) = f^{-1}(2x+3) = \frac{(2x+3)-3}{2} = \frac{2x}{2} = x$$ ✓

**Example 5.2: Function Without Inverse**

Consider $f: \mathbb{R} \to \mathbb{R}$, $f(x) = x^2$

**Problem:** Not one-to-one (fails horizontal line test)
- $f(2) = 4$ and $f(-2) = 4$

**Solution:** Restrict domain to $[0, \infty)$

Then $f: [0, \infty) \to [0, \infty)$ defined by $f(x) = x^2$ is bijective.

**Finding inverse:**
$$y = x^2$$
$$x = \sqrt{y}$$ (taking positive root since $x \geq 0$)

$$f^{-1}(x) = \sqrt{x}$$

### 5.3 Properties of Inverse Functions

1. **Uniqueness:** If $f$ has an inverse, it is unique
2. **Symmetry:** If $f^{-1}$ is the inverse of $f$, then $f$ is the inverse of $f^{-1}$
3. **Composition:** $(f \circ f^{-1})(x) = (f^{-1} \circ f)(x) = x$
4. **Inverse of Composition:** $(f \circ g)^{-1} = g^{-1} \circ f^{-1}$ (order reverses!)

**Example 5.3: Inverse of Composition**

Let $f(x) = 2x$ and $g(x) = x + 3$

**Find $(f \circ g)^{-1}$:**

Method 1: Compose first, then invert
$$(f \circ g)(x) = 2(x + 3) = 2x + 6$$

To find inverse: $y = 2x + 6 \implies x = \frac{y-6}{2}$

$$(f \circ g)^{-1}(x) = \frac{x - 6}{2}$$

Method 2: Use property $(f \circ g)^{-1} = g^{-1} \circ f^{-1}$

$f^{-1}(x) = \frac{x}{2}$ and $g^{-1}(x) = x - 3$

$$(g^{-1} \circ f^{-1})(x) = g^{-1}\left(\frac{x}{2}\right) = \frac{x}{2} - 3 = \frac{x - 6}{2}$$

Both methods agree! ✓

---

## 6. Special Classes of Functions

### 6.1 Even and Odd Functions

**Even Function:** $f(-x) = f(x)$ for all $x$ in domain
- Symmetric about y-axis
- Examples: $f(x) = x^2$, $f(x) = \cos(x)$, $f(x) = |x|$

**Odd Function:** $f(-x) = -f(x)$ for all $x$ in domain
- Symmetric about origin
- Examples: $f(x) = x^3$, $f(x) = \sin(x)$, $f(x) = \frac{1}{x}$

**Example 6.1: Testing Even/Odd**

**Function 1:** $f(x) = x^4 - 2x^2 + 1$

Test: $f(-x) = (-x)^4 - 2(-x)^2 + 1 = x^4 - 2x^2 + 1 = f(x)$

**Conclusion:** Even function ✓

**Function 2:** $g(x) = x^3 - x$

Test: $g(-x) = (-x)^3 - (-x) = -x^3 + x = -(x^3 - x) = -g(x)$

**Conclusion:** Odd function ✓

**Function 3:** $h(x) = x^2 + x$

Test: $h(-x) = (-x)^2 + (-x) = x^2 - x$

$h(-x) \neq h(x)$ and $h(-x) \neq -h(x)$

**Conclusion:** Neither even nor odd

### 6.2 Monotonic Functions

**Increasing Function:** $f(x_1) < f(x_2)$ whenever $x_1 < x_2$
- **Strictly increasing:** Inequality is strict
- **Non-decreasing:** $f(x_1) \leq f(x_2)$ when $x_1 < x_2$

**Decreasing Function:** $f(x_1) > f(x_2)$ whenever $x_1 < x_2$
- **Strictly decreasing:** Inequality is strict
- **Non-increasing:** $f(x_1) \geq f(x_2)$ when $x_1 < x_2$

**Example 6.2: Monotonic Functions**

- $f(x) = 2x + 1$: Strictly increasing
- $g(x) = -3x + 5$: Strictly decreasing
- $h(x) = x^2$: Neither (increases on $[0, \infty)$, decreases on $(-\infty, 0]$)

### 6.3 Bounded Functions

**Bounded Above:** $\exists M$ such that $f(x) \leq M$ for all $x$ in domain

**Bounded Below:** $\exists m$ such that $f(x) \geq m$ for all $x$ in domain

**Bounded:** Bounded both above and below

**Example 6.3: Bounded Functions**

- $f(x) = \sin(x)$: Bounded ($-1 \leq \sin(x) \leq 1$)
- $g(x) = x^2$: Bounded below (by 0), not bounded above
- $h(x) = \frac{1}{1 + x^2}$: Bounded ($0 < h(x) \leq 1$)

---

## 7. Function Transformations

### 7.1 Vertical and Horizontal Shifts

**Vertical Shift:** $g(x) = f(x) + k$
- $k > 0$: Shift up by $k$ units
- $k < 0$: Shift down by $|k|$ units

**Horizontal Shift:** $g(x) = f(x - h)$
- $h > 0$: Shift right by $h$ units
- $h < 0$: Shift left by $|h|$ units

**Example 7.1: Shifts**

Starting with $f(x) = x^2$:

- $g(x) = x^2 + 3$: Shift up 3 units
- $h(x) = (x - 2)^2$: Shift right 2 units
- $k(x) = (x + 1)^2 - 4$: Shift left 1, down 4

### 7.2 Reflections and Stretches

**Reflection about x-axis:** $g(x) = -f(x)$

**Reflection about y-axis:** $g(x) = f(-x)$

**Vertical Stretch:** $g(x) = af(x)$, $|a| > 1$
- Stretches graph vertically by factor $|a|$

**Vertical Compression:** $g(x) = af(x)$, $0 < |a| < 1$
- Compresses graph vertically by factor $|a|$

**Horizontal Stretch/Compression:** $g(x) = f(bx)$
- $|b| > 1$: Horizontal compression by factor $\frac{1}{|b|}$
- $0 < |b| < 1$: Horizontal stretch by factor $\frac{1}{|b|}$

**Example 7.2: Combined Transformations**

Starting with $f(x) = \sqrt{x}$, obtain $g(x) = -2\sqrt{x - 1} + 3$:

1. Shift right 1: $\sqrt{x - 1}$
2. Vertical stretch by 2: $2\sqrt{x - 1}$
3. Reflect about x-axis: $-2\sqrt{x - 1}$
4. Shift up 3: $-2\sqrt{x - 1} + 3$

---

## 8. Data Science Applications

### 8.1 Activation Functions in Neural Networks

**Purpose:** Introduce non-linearity in neural networks

**Common Activation Functions:**

1. **Sigmoid:** $\sigma(x) = \frac{1}{1 + e^{-x}}$
   - Range: $(0, 1)$
   - Used in binary classification
   - Smooth, differentiable

2. **ReLU (Rectified Linear Unit):** $\text{ReLU}(x) = \max(0, x)$
   - Range: $[0, \infty)$
   - Most popular in deep learning
   - Computationally efficient

3. **Tanh:** $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$
   - Range: $(-1, 1)$
   - Zero-centered output

### 8.2 Loss Functions

**Purpose:** Measure prediction error

**Mean Squared Error (Regression):**
$$L(y, \hat{y}) = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**Cross-Entropy (Classification):**
$$L(y, \hat{y}) = -\sum_{i=1}^{n} y_i \log(\hat{y}_i)$$

### 8.3 Data Transformations

**Log Transform:** $y = \log(x)$
- Stabilizes variance
- Makes skewed data more normal
- Common in financial data

**Square Root Transform:** $y = \sqrt{x}$
- Reduces right skewness
- Used for count data

**Box-Cox Transform:** Generalized power transformation
$$y(\lambda) = \begin{cases}
\frac{x^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(x) & \text{if } \lambda = 0
\end{cases}$$

### 8.4 Feature Scaling

**Min-Max Normalization:** $f(x) = \frac{x - \min(x)}{\max(x) - \min(x)}$
- Maps to $[0, 1]$

**Z-Score Normalization:** $f(x) = \frac{x - \mu}{\sigma}$
- Mean 0, standard deviation 1

**Purpose:** Ensure features have similar scales for algorithms sensitive to magnitude (e.g., gradient descent, k-NN)

### 8.5 Hash Functions

**Purpose:** Map data to fixed-size values

**Properties:**
- Deterministic: Same input → same output
- Fast computation
- Uniform distribution
- Difficult to invert

**Applications:**
- Data structures (hash tables)
- Cryptography
- Feature hashing in ML

---

## 9. Common Pitfalls and Misconceptions

### 9.1 Domain Confusion

❌ **Wrong:** "The domain of $f(x) = \sqrt{x^2}$ is $[0, \infty)$"
✅ **Correct:** $\sqrt{x^2} = |x|$ has domain $\mathbb{R}$ (all real numbers)

**Reason:** $x^2 \geq 0$ for all real $x$

### 9.2 Composition Order

❌ **Wrong:** "$(f \circ g)(x) = (g \circ f)(x)$ always"
✅ **Correct:** Composition is generally NOT commutative

**Example:** $f(x) = x + 1$, $g(x) = x^2$
- $(f \circ g)(x) = x^2 + 1$
- $(g \circ f)(x) = (x + 1)^2 = x^2 + 2x + 1$

### 9.3 Inverse Notation

❌ **Wrong:** $f^{-1}(x) = \frac{1}{f(x)}$
✅ **Correct:** $f^{-1}$ is the inverse function, NOT reciprocal

**Clarification:**
- $f^{-1}(x)$: Inverse function
- $\frac{1}{f(x)}$ or $(f(x))^{-1}$: Reciprocal

### 9.4 Function vs Equation

❌ **Wrong:** Confusing $y = x^2$ (equation) with $f(x) = x^2$ (function)
✅ **Correct:**
- Equation: Describes relationship
- Function: Assigns unique output to each input

---

## 10. Worked Examples

### Example 10.1: Comprehensive Function Analysis

**Problem:** Analyze the function $f(x) = \frac{2x + 1}{x - 3}$

**Solution:**

**a) Find domain:**
Division by zero when $x = 3$

**Domain:** $\mathbb{R} \setminus \{3\}$ or $(-\infty, 3) \cup (3, \infty)$

**b) Find range:**
Set $y = \frac{2x + 1}{x - 3}$ and solve for $x$:
$$y(x - 3) = 2x + 1$$
$$yx - 3y = 2x + 1$$
$$yx - 2x = 3y + 1$$
$$x(y - 2) = 3y + 1$$
$$x = \frac{3y + 1}{y - 2}$$

This is undefined when $y = 2$, so:

**Range:** $\mathbb{R} \setminus \{2\}$ or $(-\infty, 2) \cup (2, \infty)$

**c) Is it one-to-one?**
Assume $f(x_1) = f(x_2)$:
$$\frac{2x_1 + 1}{x_1 - 3} = \frac{2x_2 + 1}{x_2 - 3}$$
$$(2x_1 + 1)(x_2 - 3) = (2x_2 + 1)(x_1 - 3)$$
$$2x_1 x_2 - 6x_1 + x_2 - 3 = 2x_1 x_2 - 6x_2 + x_1 - 3$$
$$-6x_1 + x_2 = -6x_2 + x_1$$
$$7x_2 = 7x_1$$
$$x_1 = x_2$$

**Yes, one-to-one** ✓

**d) Find inverse:**
From part (b), swapping $x$ and $y$:
$$f^{-1}(x) = \frac{3x + 1}{x - 2}$$

**e) Verify inverse:**
$$f(f^{-1}(x)) = f\left(\frac{3x+1}{x-2}\right) = \frac{2\left(\frac{3x+1}{x-2}\right) + 1}{\frac{3x+1}{x-2} - 3}$$

Simplify numerator: $\frac{2(3x+1) + (x-2)}{x-2} = \frac{7x}{x-2}$

Simplify denominator: $\frac{(3x+1) - 3(x-2)}{x-2} = \frac{7}{x-2}$

$$f(f^{-1}(x)) = \frac{\frac{7x}{x-2}}{\frac{7}{x-2}} = x$$ ✓

### Example 10.2: Composition with Domain Analysis

**Problem:** Let $f(x) = \sqrt{x}$ and $g(x) = x^2 - 4$. Find domains of $(f \circ g)(x)$ and $(g \circ f)(x)$.

**Solution:**

**For $(f \circ g)(x) = f(g(x)) = \sqrt{x^2 - 4}$:**

Require $x^2 - 4 \geq 0$:
$$x^2 \geq 4$$
$$|x| \geq 2$$
$$x \leq -2 \text{ or } x \geq 2$$

**Domain of $f \circ g$:** $(-\infty, -2] \cup [2, \infty)$

**For $(g \circ f)(x) = g(f(x)) = (\sqrt{x})^2 - 4 = x - 4$:**

Domain of $f$: $[0, \infty)$

Output $\sqrt{x}$ is always in domain of $g$ (which is $\mathbb{R}$)

**Domain of $g \circ f$:** $[0, \infty)$

### Example 10.3: Piecewise Function

**Problem:** Define and analyze:
$$f(x) = \begin{cases}
x^2 & \text{if } x < 0 \\
2x & \text{if } 0 \leq x < 3 \\
6 & \text{if } x \geq 3
\end{cases}$$

**Solution:**

**a) Evaluate $f(-2)$, $f(1)$, $f(5)$:**
- $f(-2) = (-2)^2 = 4$
- $f(1) = 2(1) = 2$
- $f(5) = 6$

**b) Find range:**
- For $x < 0$: $f(x) = x^2 > 0$ (takes all positive values as $x \to -\infty$)
- For $0 \leq x < 3$: $f(x) = 2x \in [0, 6)$
- For $x \geq 3$: $f(x) = 6$

**Range:** $[0, \infty)$

**c) Is it continuous?**
Check at boundaries:
- At $x = 0$: $\lim_{x \to 0^-} f(x) = 0$ and $f(0) = 0$ ✓
- At $x = 3$: $\lim_{x \to 3^-} f(x) = 6$ and $f(3) = 6$ ✓

**Yes, continuous everywhere**

---

## 11. Practice Problems

### Basic Level

**Problem 1:** Find the domain and range of $f(x) = \frac{1}{x + 2}$.

**Problem 2:** Determine if $f(x) = x^3 + 2x$ is even, odd, or neither.

**Problem 3:** If $f(x) = 3x - 1$ and $g(x) = x^2$, find $(f \circ g)(x)$ and $(g \circ f)(x)$.

**Problem 4:** Is the function $f(x) = |x|$ one-to-one? Justify your answer.

**Problem 5:** Find the inverse of $f(x) = \frac{x + 2}{3}$.

### Intermediate Level

**Problem 6:** Find the domain of $h(x) = \sqrt{\frac{x + 1}{x - 2}}$.

**Problem 7:** Prove that if $f$ and $g$ are both one-to-one, then $(f \circ g)$ is one-to-one.

**Problem 8:** Let $f(x) = 2x + 1$ and $g(x) = x^2 - 3$. Find a function $h(x)$ such that $(f \circ h)(x) = g(x)$.

**Problem 9:** Determine if the function $f: [0, \infty) \to [0, \infty)$ defined by $f(x) = x^2$ has an inverse. If yes, find it.

**Problem 10:** Describe the transformations needed to obtain $g(x) = -\frac{1}{2}(x + 3)^2 - 1$ from $f(x) = x^2$.

### Advanced Level

**Problem 11:** Prove that a function has an inverse if and only if it is bijective.

**Problem 12:** Let $f: \mathbb{R} \to \mathbb{R}$ be a function such that $f(f(x)) = x$ for all $x$. What can you conclude about $f$?

**Problem 13:** Find all functions $f: \mathbb{R} \to \mathbb{R}$ satisfying $f(x + y) = f(x) + f(y)$ for all $x, y \in \mathbb{R}$.

**Problem 14:** Show that the composition of two surjective functions is surjective.

**Problem 15:** Design a data transformation pipeline: Given sales data with exponential growth, propose a sequence of function transformations to make the data suitable for linear regression. Explain your choices.

---

## Summary and Key Takeaways

### Core Concepts Mastered

1. **Function Fundamentals**
   - Formal definition: unique output for each input
   - Domain, codomain, and range
   - Function notation and terminology

2. **Function Types**
   - **One-to-One (Injective)**: Different inputs → different outputs
   - **Onto (Surjective)**: All codomain elements are achieved
   - **Bijective**: Both one-to-one and onto (invertible)

3. **Operations on Functions**
   - **Composition**: $(f \circ g)(x) = f(g(x))$
   - **Inverse**: $f^{-1}(f(x)) = x$ (only for bijections)
   - **Transformations**: Shifts, reflections, stretches

4. **Special Function Classes**
   - Even/odd functions
   - Monotonic functions
   - Bounded functions

5. **Data Science Applications**
   - Activation functions (sigmoid, ReLU, tanh)
   - Loss functions (MSE, cross-entropy)
   - Data transformations (log, Box-Cox)
   - Feature scaling
   - Hash functions

### Essential Properties

| Property | Definition | Test |
|----------|-----------|------|
| **One-to-One** | $f(x_1) = f(x_2) \implies x_1 = x_2$ | Horizontal line test |
| **Onto** | Range = Codomain | Every $y$ has pre-image |
| **Even** | $f(-x) = f(x)$ | Y-axis symmetry |
| **Odd** | $f(-x) = -f(x)$ | Origin symmetry |
| **Invertible** | Bijective | Has inverse function |

### Key Formulas

| Concept | Formula |
|---------|---------|
| Composition | $(f \circ g)(x) = f(g(x))$ |
| Inverse property | $f(f^{-1}(x)) = f^{-1}(f(x)) = x$ |
| Inverse of composition | $(f \circ g)^{-1} = g^{-1} \circ f^{-1}$ |
| Vertical shift | $f(x) + k$ |
| Horizontal shift | $f(x - h)$ |
| Vertical stretch | $af(x)$, $|a| > 1$ |

### Connections to Other Topics

- **Week 4 (Polynomials)**: Polynomials as a class of functions
- **Week 6 (Exponential/Logarithmic)**: Special function types with unique properties
- **Week 7 (Calculus)**: Derivatives and integrals as operations on functions
- **Statistics**: Probability density functions, cumulative distribution functions
- **Machine Learning**: Loss functions, activation functions, feature engineering

### Study Checklist

- [ ] Understand formal definition of functions
- [ ] Can find domain and range of any function
- [ ] Distinguish between one-to-one, onto, and bijective functions
- [ ] Perform function composition correctly
- [ ] Find inverse functions when they exist
- [ ] Identify even/odd and monotonic functions
- [ ] Apply function transformations
- [ ] Connect function concepts to data science applications
- [ ] Avoid common pitfalls (composition order, inverse notation)

---

## Additional Resources

### Recommended Reading

1. **Textbook Sections**: IIT Madras BSMA1001 Week 5 materials
2. **Practice**: Khan Academy - Functions and their graphs
3. **Visualization**: Desmos function transformations
4. **Applications**: "Deep Learning" by Goodfellow - Chapter 6 (Activation Functions)

### Online Tools

- **Desmos**: Interactive function graphing
- **WolframAlpha**: Analyze function properties
- **GeoGebra**: Dynamic function transformations

### Next Week Preview

**Week 6: Exponential and Logarithmic Functions**
- Exponential function properties
- Logarithms and their laws
- Applications in data science (logistic regression, information theory)
- Growth and decay models

---

**End of Week 5 Notes**

*These notes are part of the IIT Madras BS in Data Science Foundation Level coursework.*
