# Course Conventions Reference

> IIT Madras BS Data Science - Naming and Organization Standards

## Weekly Notebook Structure

Each week's notebook should follow this structure:

### Header Cell (Markdown)
```markdown
# Week XX: Topic Name

## BSXXNNNN - Course Name

**Learning Objectives:**
- Objective 1
- Objective 2
- Objective 3

---
```

### Import Cell Pattern
```python
# Standard library imports
import logging
from pathlib import Path
from typing import TYPE_CHECKING

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Type hints (avoid runtime imports)
if TYPE_CHECKING:
    from pandas import DataFrame, Series

# Configuration
np.random.seed(42)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
```

## Course-Specific Contexts

### Mathematics Courses (BSMA)
- Use symbolic notation when explaining concepts
- Include worked examples with step-by-step solutions
- Visualize functions and geometric concepts
- Reference real-world applications

### Statistics Courses (BSMA - Statistics)
- Always use `ddof=1` for sample statistics
- Include confidence intervals where appropriate
- Visualize distributions and test results
- Use supply chain examples (demand variability, quality control)

### Programming Courses (BSCS)
- Focus on clean, readable code
- Include time/space complexity analysis where relevant
- Add comprehensive docstrings
- Test edge cases in examples

### Data Science Courses (BSDA)
- Use real-world datasets when possible
- Include EDA before modeling
- Compare multiple approaches
- Evaluate with appropriate metrics

## Exercise Section Template

```markdown
## Exercises

### Exercise 1: Basic Concept
**Difficulty:** ⭐ Easy
**Objective:** Practice fundamental concept

[Problem statement]

### Exercise 2: Application
**Difficulty:** ⭐⭐ Medium
**Objective:** Apply concept to real scenario

[Problem statement]

### Exercise 3: Challenge
**Difficulty:** ⭐⭐⭐ Hard
**Objective:** Extend understanding

[Problem statement]
```

## Solution Pattern

```python
# Solution Cell - Exercise N
def solution_exercise_n():
    """Solution for Exercise N.

    Approach:
        1. Step description
        2. Step description
        3. Step description
    """
    # Implementation
    pass

# Demonstrate
result = solution_exercise_n()
print(f"Result: {result}")
```

## Week-by-Week Topics Reference

### Foundation Level

#### Mathematics I (BSMA1001)
| Week | Topic |
|------|-------|
| 1 | Set Theory, Number Systems |
| 2 | Coordinate Systems, Straight Lines |
| 3 | Quadratic Functions |
| 4 | Polynomials, Algorithms |

#### Statistics I (BSMA1002)
| Week | Topic |
|------|-------|
| 1 | Data Types, Descriptive Statistics |
| 2 | Categorical Data, Frequency Distribution |
| 3 | Numerical Data, Central Tendency |
| 4 | Association, Correlation |

#### Python Programming (BSCS1002)
| Week | Topic |
|------|-------|
| 1 | Introduction to Algorithms |
| 2 | Conditionals |
| 3 | Conditionals (Continued) |
| 4 | Iterations and Ranges |

## Supply Chain Domain Examples

> **Philosophy**: Cover syllabus concepts thoroughly, then apply with retail supply chain examples.
> Both **time series** and **cross-sectional** data are common in retail supply chain.

### Time Series Data Examples (use `TimeSeriesSplit`)
Data where order matters - indexed by time.

| Domain | Example Dataset | Use Case |
|--------|----------------|----------|
| Demand Forecasting | Daily sales by SKU-Store | Predict next week's demand |
| Inventory Levels | Weekly stock positions | Replenishment planning |
| Seasonal Patterns | Monthly category sales | Promotional calendar |
| Lead Time Analysis | Order-to-delivery days | Supplier performance trends |

### Cross-Sectional Data Examples (use `KFold`/`StratifiedKFold`)
Snapshot data - each row is an independent entity.

| Domain | Example Dataset | Use Case |
|--------|----------------|----------|
| Store Analytics | Store attributes & KPIs | Performance clustering |
| SKU Profiling | Product characteristics | Category management |
| Supplier Scoring | Vendor metrics | Sourcing decisions |
| Price Optimization | SKU pricing data | Elasticity analysis |
| Assortment Planning | Store-SKU matrix | Localization |

### When to Use Each

```
┌─────────────────────────────────────────────────────────────┐
│  QUESTION: "Does the ORDER of observations matter?"        │
├─────────────────────────────────────────────────────────────┤
│  YES → Time Series                                          │
│        • "What will sales be NEXT week?"                   │
│        • "Is demand TRENDING up?"                          │
│        • Use: TimeSeriesSplit, lag features, rolling stats │
├─────────────────────────────────────────────────────────────┤
│  NO → Cross-Sectional                                       │
│       • "Which stores are HIGH performers?"                │
│       • "What DRIVES SKU profitability?"                   │
│       • Use: KFold, StratifiedKFold, standard scaling      │
└─────────────────────────────────────────────────────────────┘
```

### Mapping Course Topics to Real-World Examples

| Syllabus Topic | Time Series Example | Cross-Sectional Example |
|---------------|--------------------|-----------------------|
| **Sets & Venn** | Product overlap across weeks | Store format overlap |
| **Descriptive Stats** | Weekly sales summary | Store performance metrics |
| **Probability** | Stockout probability | Defect rate by category |
| **Distributions** | Demand distribution fit | Price point distribution |
| **Hypothesis Testing** | Pre/post promo effect | Regional sales difference |
| **Correlation** | Price-demand relationship | Store size vs sales |
| **Regression** | Demand forecasting model | Sales drivers analysis |
| **Classification** | Trend detection (up/down) | Store tier classification |

## Figure Standards

### Naming Convention
`week-XX-description.png`

Examples:
- `week-01-set-venn-diagram.png`
- `week-02-linear-regression-fit.png`
- `week-03-demand-forecast-comparison.png`

### Save Command
```python
fig.savefig(
    f'week-{week_number:02d}-{description}.png',
    dpi=150,
    bbox_inches='tight',
    facecolor='white',
    edgecolor='none'
)
```

## Assessment Artifacts

### Quiz Preparation Notes
Create summary cells with key formulas and concepts:

```markdown
## Quiz Preparation - Week XX

### Key Formulas
1. Formula 1: $equation$
2. Formula 2: $equation$

### Key Concepts
- Concept 1: Brief explanation
- Concept 2: Brief explanation

### Common Pitfalls
- Pitfall 1: How to avoid
- Pitfall 2: How to avoid
```

### Assignment Template
```python
# Assignment Week XX - Problem N
# Student: [Name]
# Date: YYYY-MM-DD

"""
Problem Statement:
[Copy problem statement here]

Approach:
[Describe your approach]
"""

# Your solution
def solve_problem_n():
    pass

# Test your solution
if __name__ == "__main__":
    result = solve_problem_n()
    print(f"Answer: {result}")
```
