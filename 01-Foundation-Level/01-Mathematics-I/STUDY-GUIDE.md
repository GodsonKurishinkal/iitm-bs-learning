# BSMA1001: Mathematics for Data Science I - Study Guide

**Course ID:** BSMA1001  
**Credits:** 4  
**Duration:** 12 weeks  
**Instructors:** Neelesh Upadhye, Madhavan Mukund  
**Prerequisites:** None

## 📚 Course Overview

This course introduces foundational mathematical concepts essential for data science, focusing on functions, algebra, and graph theory with practical applications in machine learning and data analysis.

## 🎯 Learning Objectives

By the end of this course, you will be able to:
- Identify and work with properties of linear, quadratic, polynomial, exponential, and logarithmic functions
- Find roots, maxima, and minima of various function types
- Apply graph theory concepts to real-world problems
- Use Python to visualize mathematical concepts
- Solve practical data science problems using mathematical foundations

## 📖 Reference Materials

**Required Books (Available for Download):**
- **Sets & Functions (Vol 1)** - [Download from course page](https://study.iitm.ac.in/ds/course_pages/BSMA1001.html)
- **Calculus (Vol 2)** - [Download from course page](https://study.iitm.ac.in/ds/course_pages/BSMA1001.html)
- **Graph Theory (Vol 3)** - [Download from course page](https://study.iitm.ac.in/ds/course_pages/BSMA1001.html)

**Video Lectures:** [YouTube Playlist](https://www.youtube.com/playlist?list=PLZ2ps__7DhBa7DqhN-HidvTC0x2PxFvZL)

---

## 📅 Week-by-Week Breakdown

### Week 1: Set Theory and Relations

**Topics Covered:**
- Introduction to sets and set operations
- Number systems (Natural numbers, Integers, Rationals, Reals)
- Relations and their properties
- Functions: Domain, Codomain, and Range
- Types of functions (Injective, Surjective, Bijective)

**Learning Activities:**
1. **Read:** Sets & Functions Vol 1, Chapter 1-2
2. **Watch:** Week 1 video lectures
3. **Practice:** Set operations problems
4. **Code:** Implement set operations in Python using `set` data type

**Practice Problems:**
- Define and visualize different types of relations
- Prove functions are injective/surjective/bijective
- Create Venn diagrams for complex set operations

**Python Applications:**
```python
# Practice with Python sets
A = {1, 2, 3, 4, 5}
B = {4, 5, 6, 7, 8}
# Union, Intersection, Difference, Symmetric Difference
```

**Weekly Notebook:** `week-01-sets-relations.ipynb`

---

### Week 2: Coordinate Systems and Straight Lines

**Topics Covered:**
- Cartesian coordinate system
- Distance formula
- Midpoint formula
- Slope of a line
- Equation of a line (slope-intercept, point-slope, two-point form)
- Parallel and perpendicular lines
- Distance between point and line

**Learning Activities:**
1. **Read:** Sets & Functions Vol 1, Chapter 3-4
2. **Watch:** Week 2 video lectures
3. **Practice:** Line equation problems
4. **Code:** Plot lines using matplotlib

**Practice Problems:**
- Find equations of lines given different conditions
- Calculate distances and angles between lines
- Determine if lines are parallel or perpendicular

**Python Applications:**
```python
import matplotlib.pyplot as plt
import numpy as np

# Plot straight lines
x = np.linspace(-10, 10, 100)
y1 = 2*x + 3  # Line 1
y2 = -0.5*x + 1  # Perpendicular line
```

**Weekly Notebook:** `week-02-coordinate-geometry.ipynb`

---

### Week 3: Quadratic Functions

**Topics Covered:**
- Standard form of quadratic equations
- Vertex form and completing the square
- Roots of quadratic equations
- Discriminant and nature of roots
- Maximum and minimum values
- Parabolas and their properties
- Applications in optimization

**Learning Activities:**
1. **Read:** Sets & Functions Vol 1, Chapter 5
2. **Watch:** Week 3 video lectures
3. **Practice:** Quadratic equation problems
4. **Code:** Visualize parabolas and find extrema

**Practice Problems:**
- Find roots using quadratic formula
- Convert between standard and vertex forms
- Solve real-world optimization problems

**Python Applications:**
```python
# Plot quadratic functions
def quadratic(x, a, b, c):
    return a*x**2 + b*x + c

# Find vertex, roots, and plot
```

**Weekly Notebook:** `week-03-quadratics.ipynb`

---

### Week 4: Algebra of Polynomials

**Topics Covered:**
- Definition of polynomials
- Degree of a polynomial
- Operations on polynomials (addition, multiplication, division)
- Remainder theorem
- Factor theorem
- Roots of polynomials
- Fundamental theorem of algebra

**Learning Activities:**
1. **Read:** Sets & Functions Vol 1, Chapter 6
2. **Watch:** Week 4 video lectures
3. **Practice:** Polynomial manipulation problems
4. **Code:** Use NumPy polynomial functions

**Practice Problems:**
- Factor higher-degree polynomials
- Apply remainder and factor theorems
- Find all roots of polynomials

**Python Applications:**
```python
import numpy as np

# Define polynomial
p = np.poly1d([1, -6, 11, -6])  # x^3 - 6x^2 + 11x - 6
roots = np.roots(p)
```

**Weekly Notebook:** `week-04-polynomials.ipynb`

---

### Week 5: Exponential and Logarithmic Functions

**Topics Covered:**
- Properties of exponential functions
- The number e and natural exponentials
- Logarithms and their properties
- Change of base formula
- Logarithmic scales
- Applications in data science (log transformations)

**Learning Activities:**
1. **Read:** Sets & Functions Vol 1, Chapter 7-8
2. **Watch:** Week 5 video lectures
3. **Practice:** Exponential and log problems
4. **Code:** Plot exponential growth/decay

**Practice Problems:**
- Solve exponential equations
- Apply logarithm rules
- Model real-world growth scenarios

**Python Applications:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Exponential growth and decay
x = np.linspace(0, 10, 100)
y_growth = np.exp(0.5*x)
y_decay = np.exp(-0.5*x)
y_log = np.log(x)
```

**Weekly Notebook:** `week-05-exp-log.ipynb`

---

### Week 6: Trigonometric Functions

**Topics Covered:**
- Angles and radians
- Sine, cosine, and tangent functions
- Unit circle
- Trigonometric identities
- Inverse trigonometric functions
- Applications in periodic data

**Learning Activities:**
1. **Read:** Calculus Vol 2, Chapter 1
2. **Watch:** Week 6 video lectures
3. **Practice:** Trigonometric problems
4. **Code:** Visualize sine and cosine waves

**Practice Problems:**
- Prove trigonometric identities
- Solve trigonometric equations
- Model periodic phenomena

**Python Applications:**
```python
# Plot sine and cosine
x = np.linspace(0, 4*np.pi, 1000)
y_sin = np.sin(x)
y_cos = np.cos(x)
```

**Weekly Notebook:** `week-06-trigonometry.ipynb`

---

### Week 7: Introduction to Calculus - Limits

**Topics Covered:**
- Concept of limits
- One-sided limits
- Limit laws
- Continuity
- Limits at infinity
- Indeterminate forms

**Learning Activities:**
1. **Read:** Calculus Vol 2, Chapter 2
2. **Watch:** Week 7 video lectures
3. **Practice:** Limit evaluation problems
4. **Code:** Visualize limits numerically

**Practice Problems:**
- Evaluate limits algebraically
- Determine continuity of functions
- Handle indeterminate forms

**Python Applications:**
```python
# Numerical limit approximation
def f(x):
    return (x**2 - 1)/(x - 1)

# Approach limit from both sides
```

**Weekly Notebook:** `week-07-limits.ipynb`

---

### Week 8: Derivatives

**Topics Covered:**
- Definition of derivative
- Differentiation rules (power, product, quotient, chain rule)
- Derivatives of common functions
- Higher-order derivatives
- Applications: rates of change

**Learning Activities:**
1. **Read:** Calculus Vol 2, Chapter 3-4
2. **Watch:** Week 8 video lectures
3. **Practice:** Differentiation problems
4. **Code:** Compute numerical derivatives

**Practice Problems:**
- Find derivatives using various rules
- Solve related rates problems
- Compute higher-order derivatives

**Python Applications:**
```python
from scipy.misc import derivative

# Numerical differentiation
def f(x):
    return x**3 - 2*x**2 + x - 5

derivative(f, 2.0, dx=1e-6)
```

**Weekly Notebook:** `week-08-derivatives.ipynb`

---

### Week 9: Applications of Derivatives

**Topics Covered:**
- Critical points
- Finding maxima and minima
- First and second derivative tests
- Concavity and inflection points
- Optimization problems
- Curve sketching

**Learning Activities:**
1. **Read:** Calculus Vol 2, Chapter 5
2. **Watch:** Week 9 video lectures
3. **Practice:** Optimization problems
4. **Code:** Visualize function behavior

**Practice Problems:**
- Find critical points and classify them
- Solve optimization problems
- Sketch curves using derivative information

**Python Applications:**
```python
from scipy.optimize import minimize

# Find minimum of function
def f(x):
    return x**4 - 4*x**3 + 4*x**2

result = minimize(f, x0=0)
```

**Weekly Notebook:** `week-09-optimization.ipynb`

---

### Week 10: Graph Theory - Basics

**Topics Covered:**
- Introduction to graphs
- Vertices and edges
- Degree of vertices
- Paths and cycles
- Connected graphs
- Trees

**Learning Activities:**
1. **Read:** Graph Theory Vol 3, Chapter 1-2
2. **Watch:** Week 10 video lectures
3. **Practice:** Graph problems
4. **Code:** Use NetworkX library

**Practice Problems:**
- Identify graph properties
- Determine if graphs are isomorphic
- Find paths and cycles

**Python Applications:**
```python
import networkx as nx
import matplotlib.pyplot as plt

# Create and visualize graphs
G = nx.Graph()
G.add_edges_from([(1,2), (2,3), (3,4), (4,1)])
nx.draw(G, with_labels=True)
```

**Weekly Notebook:** `week-10-graphs-basics.ipynb`

---

### Week 11: Graph Theory - Advanced Concepts

**Topics Covered:**
- Directed graphs (Digraphs)
- Weighted graphs
- Shortest path algorithms
- Minimum spanning trees
- Graph coloring
- Applications in networks and data structures

**Learning Activities:**
1. **Read:** Graph Theory Vol 3, Chapter 3-4
2. **Watch:** Week 11 video lectures
3. **Practice:** Advanced graph problems
4. **Code:** Implement graph algorithms

**Practice Problems:**
- Apply Dijkstra's algorithm
- Find minimum spanning trees
- Solve graph coloring problems

**Python Applications:**
```python
# Shortest path
G = nx.Graph()
G.add_weighted_edges_from([(1,2,4), (2,3,2), (1,3,7)])
path = nx.shortest_path(G, source=1, target=3, weight='weight')
```

**Weekly Notebook:** `week-11-graphs-advanced.ipynb`

---

### Week 12: Review and Applications

**Topics Covered:**
- Integration of concepts
- Real-world data science applications
- Mathematical modeling
- Problem-solving strategies
- Exam preparation

**Learning Activities:**
1. **Review:** All previous weeks' materials
2. **Practice:** Mixed problems from all topics
3. **Project:** Complete a mini data science project using mathematical concepts
4. **Code:** Comprehensive problem set

**Practice Problems:**
- Mixed problems covering all topics
- Application-based questions
- Past exam questions (if available)

**Project Ideas:**
1. Model population growth using exponential functions
2. Optimize delivery routes using graph theory
3. Analyze periodic sales data using trigonometric functions
4. Build a recommendation system using graph algorithms

**Weekly Notebook:** `week-12-comprehensive-review.ipynb`

---

## 🎯 Assessment Structure

- **Weekly Online Assignments:** 10-20% (varies by term)
- **Quiz 1 (In-person):** 15-20%
- **Quiz 2 (In-person):** 15-20%
- **End Term Exam (In-person):** 50-60%

**Passing Grade:** 40% overall with at least 40% in end-term exam

---

## 💡 Study Tips

1. **Daily Practice:** Spend 30-60 minutes daily on practice problems
2. **Visual Learning:** Always plot functions to understand behavior
3. **Code Integration:** Implement every concept in Python
4. **Connect Concepts:** See how each topic relates to data science
5. **Form Study Groups:** Discuss problems with peers
6. **Use Resources:** Download and refer to all three reference books
7. **Watch Lectures Multiple Times:** Rewatch difficult sections

---

## 🔗 Important Links

- **Course Page:** https://study.iitm.ac.in/ds/course_pages/BSMA1001.html
- **Video Lectures:** [YouTube Playlist](https://www.youtube.com/playlist?list=PLZ2ps__7DhBa7DqhN-HidvTC0x2PxFvZL)
- **Reference Books:** Available on course page
- **Discussion Forum:** Access through student portal

---

## 📝 Notes

All your detailed notes for each week should go in the `notes/` folder:
- `week-01-notes.md`
- `week-02-notes.md`
- ... and so on

All practice notebooks should go in the `notebooks/` folder following the naming convention shown above.

---

## ✅ Weekly Checklist Template

```markdown
### Week X Checklist
- [ ] Watched all video lectures
- [ ] Read assigned book chapters
- [ ] Completed practice problems
- [ ] Created Jupyter notebook with examples
- [ ] Took detailed notes
- [ ] Submitted online assignment
- [ ] Reviewed previous week's material
```

---

**Remember:** Mathematics is the language of data science. Master these fundamentals to excel in all future courses!
