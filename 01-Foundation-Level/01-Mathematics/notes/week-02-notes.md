# Week 2: Coordinate Geometry and Straight Lines

**Date**: 2025-11-15  
**Course**: Mathematics for Data Science I (BSMA1001)

## Topics Covered

1. Rectangular Coordinate System
2. Distance and Midpoint Formulas
3. Slope of a Line
4. Parallel and Perpendicular Lines
5. Representations of a Line
6. General Equations of Lines
7. Straight-Line Fit (Linear Regression)

---

## Key Concepts

### 1. Rectangular Coordinate System (Cartesian Plane)

The **Cartesian coordinate system** represents points in 2D space using ordered pairs (x, y).

#### Components
- **X-axis (horizontal)**: Represents the first coordinate
- **Y-axis (vertical)**: Represents the second coordinate  
- **Origin (0, 0)**: Intersection of axes
- **Quadrants**: Four regions divided by the axes

```
Quadrant II (-,+)  |  Quadrant I (+,+)
        -------------------
Quadrant III (-,-) |  Quadrant IV (+,-)
```

#### Ordered Pairs
- **Point**: P = (x, y)
  - x-coordinate (abscissa): horizontal position
  - y-coordinate (ordinate): vertical position
- **(0, y)**: Points on y-axis
- **(x, 0)**: Points on x-axis

**Why important for DS**: Every data point can be plotted; scatter plots visualize relationships

### 2. Distance Formula

The **distance** between two points P₁ = (x₁, y₁) and P₂ = (x₂, y₂):

$$d = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$

**Derived from**: Pythagorean theorem

**Applications**:
- Euclidean distance in machine learning
- K-Nearest Neighbors (KNN) algorithm
- Clustering algorithms (K-Means)
- Similarity measures

### 3. Midpoint Formula

The **midpoint** M of line segment P₁P₂:

$$M = \left(\frac{x_1 + x_2}{2}, \frac{x_1 + y_2}{2}\right)$$

**Interpretation**: Average of coordinates

### 4. Slope of a Line

**Slope (m)** measures the steepness and direction of a line:

$$m = \frac{y_2 - y_1}{x_2 - x_1} = \frac{\text{rise}}{\text{run}} = \frac{\Delta y}{\Delta x}$$

#### Types of Slopes
- **m > 0**: Line rises (positive correlation)
- **m < 0**: Line falls (negative correlation)
- **m = 0**: Horizontal line (no change in y)
- **m = undefined**: Vertical line (no change in x)

**Why important for DS**: Slope = rate of change = derivative = gradient in ML!

### 5. Parallel Lines

Two non-vertical lines are **parallel** if and only if:

$$m_1 = m_2$$

- Same slope, different y-intercepts
- Never intersect
- **Example**: y = 2x + 3 and y = 2x - 5 are parallel

### 6. Perpendicular Lines

Two non-vertical lines are **perpendicular** if and only if:

$$m_1 \cdot m_2 = -1 \quad \text{or} \quad m_2 = -\frac{1}{m_1}$$

- Slopes are negative reciprocals
- Intersect at 90°
- **Example**: y = 2x + 1 ⊥ y = -½x + 3

**Special cases**:
- Horizontal (m=0) ⊥ Vertical (undefined)

### 7. Equations of Lines

#### Point-Slope Form
Given point (x₁, y₁) and slope m:

$$y - y_1 = m(x - x_1)$$

**When to use**: You know a point and the slope

#### Slope-Intercept Form
$$y = mx + b$$

where:
- m = slope
- b = y-intercept (where line crosses y-axis)

**When to use**: Easiest form for graphing; shows slope and intercept directly

**Why important for DS**: Standard form for linear regression!

#### Standard Form (General Form)
$$Ax + By + C = 0$$

where A, B, C are constants (A and B not both zero)

**When to use**: Mathematical proofs, systems of equations

#### Two-Point Form
Given two points (x₁, y₁) and (x₂, y₂):

$$\frac{y - y_1}{y_2 - y_1} = \frac{x - x_1}{x_2 - x_1}$$

**When to use**: You know two points

#### Intercept Form
Given x-intercept a and y-intercept b:

$$\frac{x}{a} + \frac{y}{b} = 1$$

### 8. Special Lines

- **Horizontal**: y = k (slope = 0)
- **Vertical**: x = h (slope undefined)
- **Identity**: y = x (slope = 1, passes through origin)
- **Origin through P**: y = (y₁/x₁)x

### 9. Linear Regression (Straight-Line Fit)

**Goal**: Find best-fitting line through data points

**Method**: Least Squares Regression
- Minimizes sum of squared vertical distances
- Line: ŷ = mx + b

**Formulas**:
$$m = \frac{n\sum xy - \sum x \sum y}{n\sum x^2 - (\sum x)^2}$$

$$b = \frac{\sum y - m\sum x}{n}$$

where n = number of data points

**Why fundamental**: Basis of all linear models in ML!

---

## Definitions

- **Term 1**: Definition
- **Term 2**: Definition

## Important Formulas

- Formula 1: 
- Formula 2: 

## Theorems & Proofs

### Theorem 1
<!-- Add theorem and proof -->

## Examples (Worked Problems)

### Example 1
**Problem**: 
**Solution**: 

## Questions/Doubts

- [ ] Question 1
- [ ] Question 2

## Action Items

- [ ] Review lecture slides
- [ ] Complete practice problems
- [ ] Work through notebook examples

## References

- Textbook: Chapter X
- Lecture video: [Link]

---

**Next Class**: Week 3
