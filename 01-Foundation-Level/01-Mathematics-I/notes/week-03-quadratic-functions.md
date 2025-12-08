# Week 03: Quadratic Functions

**Course**: BSMA1001 - Mathematics I
**Level**: Foundation

---

## 1. Quadratic Functions

### 1.1 Theory

A quadratic function creates a parabola when graphed. The coefficient of $x^2$ determines whether it opens upward (minimum) or downward (maximum).

### 1.2 Mathematical Definition

**Standard Form**:
$$f(x) = ax^2 + bx + c \quad \text{where } a \neq 0$$

**Vertex Form**:
$$f(x) = a(x - h)^2 + k$$

Where $(h, k)$ is the vertex of the parabola.

### 1.3 Key Properties

| Property | Condition | Result |
|----------|-----------|--------|
| Opens Upward | $a > 0$ | U-shaped, has minimum |
| Opens Downward | $a < 0$ | ∩-shaped, has maximum |
| Width | $|a|$ large | Narrow parabola |
| Width | $|a|$ small | Wide parabola |

### 1.4 Converting Between Forms

**Standard to Vertex Form**:
1. Calculate $h = -\frac{b}{2a}$
2. Calculate $k = f(h) = c - \frac{b^2}{4a}$
3. Write as $f(x) = a(x - h)^2 + k$

### 1.5 Supply Chain Application

**Retail Context**: Quadratic functions model:
- Cost curves where both too little and too much inventory are costly
- Revenue curves with optimal pricing points
- Total cost functions in inventory management

---

## 2. Vertex, Minima, and Maxima

### 2.1 Theory

The vertex is the turning point of a parabola. For optimization, this represents the best possible value (minimum cost or maximum profit).

### 2.2 Mathematical Definition

**Vertex Coordinates**:
$$\text{Vertex} = \left(-\frac{b}{2a}, f\left(-\frac{b}{2a}\right)\right)$$

Or equivalently:
$$\text{Vertex} = \left(-\frac{b}{2a}, c - \frac{b^2}{4a}\right)$$

### 2.3 Optimization Rules

| Coefficient | Parabola Shape | Vertex Type | Use Case |
|-------------|----------------|-------------|----------|
| $a > 0$ | Opens upward ∪ | **Minimum** | Minimize cost |
| $a < 0$ | Opens downward ∩ | **Maximum** | Maximize profit/revenue |

### 2.4 Finding Optimal Value

1. Identify the quadratic function $f(x) = ax^2 + bx + c$
2. Calculate optimal input: $x^* = -\frac{b}{2a}$
3. Calculate optimal output: $f(x^*) = f\left(-\frac{b}{2a}\right)$

### 2.5 Supply Chain Application

**Retail Context**: Finding the vertex gives:
- Optimal order quantity (EOQ - Economic Order Quantity)
- Optimal price point for maximum revenue
- Optimal inventory level that minimizes total costs

---

## 3. Quadratic Equations

### 3.1 Theory

Solving quadratic equations finds the x-intercepts (roots) of the parabola. These represent break-even points, equilibrium values, or critical thresholds.

### 3.2 The Quadratic Formula

For equation $ax^2 + bx + c = 0$:

$$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}$$

### 3.3 The Discriminant

The discriminant $\Delta = b^2 - 4ac$ determines the nature of roots:

| Discriminant | Value | Number of Real Roots | Graphical Meaning |
|--------------|-------|---------------------|-------------------|
| $\Delta > 0$ | Positive | Two distinct real roots | Parabola crosses x-axis twice |
| $\Delta = 0$ | Zero | One repeated root | Parabola touches x-axis once |
| $\Delta < 0$ | Negative | No real roots (complex) | Parabola doesn't cross x-axis |

### 3.4 Methods for Solving Quadratic Equations

1. **Factoring**: $ax^2 + bx + c = a(x - r_1)(x - r_2)$
2. **Quadratic Formula**: Always works
3. **Completing the Square**: Convert to vertex form
4. **Graphical**: Find x-intercepts visually

### 3.5 Vieta's Formulas

For roots $r_1$ and $r_2$ of $ax^2 + bx + c = 0$:
- Sum of roots: $r_1 + r_2 = -\frac{b}{a}$
- Product of roots: $r_1 \cdot r_2 = \frac{c}{a}$

### 3.6 Supply Chain Application

**Retail Context**:
- Roots of a profit equation indicate **break-even sales volumes**
- The discriminant tells us if break-even is achievable
- Two roots define the profitable operating range

---

## Summary

| Concept | Formula |
|---------|---------|
| Standard Form | $f(x) = ax^2 + bx + c$ |
| Vertex Form | $f(x) = a(x-h)^2 + k$ |
| Vertex x-coordinate | $x = -\frac{b}{2a}$ |
| Vertex y-coordinate | $y = c - \frac{b^2}{4a}$ |
| Quadratic Formula | $x = \frac{-b \pm \sqrt{b^2-4ac}}{2a}$ |
| Discriminant | $\Delta = b^2 - 4ac$ |
| Sum of Roots | $r_1 + r_2 = -\frac{b}{a}$ |
| Product of Roots | $r_1 \cdot r_2 = \frac{c}{a}$ |

## Key Takeaways

1. **Quadratic functions** $f(x) = ax^2 + bx + c$ create parabolas

2. **Vertex** at $x = -\frac{b}{2a}$ gives minimum ($a > 0$) or maximum ($a < 0$)

3. **Quadratic formula** solves for roots/intercepts

4. **Discriminant** determines nature and number of solutions

5. **Optimization**: Use vertex to find optimal values in business problems

---

## Next Week Preview

Week 4 covers **Algebra and Graphs of Polynomials** - we'll extend to higher-degree polynomials and their properties.

---

*IIT Madras BS Degree in Data Science*
