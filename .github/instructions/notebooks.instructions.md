---
applyTo: "**/*.ipynb"
---

# Jupyter Notebook Instructions

## Cell Structure
- Start with markdown title cell
- Import cells at top, organized: stdlib → third-party → local
- Use markdown cells to explain concepts before code
- Keep code cells focused (one concept per cell)

## Data Science Notebook Pattern
```python
# Cell 1: Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pandas import DataFrame, Series

np.random.seed(42)

# Cell 2: Configuration
%matplotlib inline
plt.style.use('seaborn-v0_8-whitegrid')
```

---

# Weekly Study Notebook Generation Guide

This section applies ONLY to weekly study content notebooks (e.g., `week-XX-topic-name.ipynb`).
NOT for assignments, projects, or extra activities.

## Notebook Structure (Follow This Order)

### 1. Header Cell (Markdown)
```markdown
# Week [XX]: [Topic Name]
**Course**: [Course Code] - [Course Name]
**Level**: [Foundation/Diploma/BSc/BS/PG-Diploma/MTech]

## Learning Objectives
- [Objective 1]
- [Objective 2]

## Prerequisites
- [Prerequisite concept 1]
- [Prerequisite concept 2]
```

### 2. Setup Cell (Python)
```python
# Standard imports for weekly notebooks
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import TYPE_CHECKING, List, Dict, Set, Tuple, Callable, Any

if TYPE_CHECKING:
    from pandas import DataFrame, Series

# Reproducibility
np.random.seed(42)

# Visualization settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 11

# Additional imports as needed per topic:
# from scipy import stats, optimize
# import networkx as nx
# from matplotlib_venn import venn2, venn3
# from fractions import Fraction
```

### 3. For EACH Concept in the Week's Syllabus

Follow this 4-part pattern for every concept:

#### Part A: Theory Section (Markdown)
- Clear definitions with mathematical notation
- Key formulas in LaTeX
- Properties and theorems
- Intuitive explanations

#### Part B: Implementation Cell (Python)
- Demonstrate the concept with code
- Type hints on ALL functions
- Google-style docstrings
- Use `np.random.seed(42)`
- Print structured output with explanations

#### Part C: VISUALIZATION Cell (Python) - MANDATORY
**I learn best by SEEING what's happening. Create comprehensive visualizations for EVERY concept.**

```python
# Multi-panel visualization pattern
fig, axes = plt.subplots(2, 2, figsize=(14, 10))  # or 2x3, 1x3 as needed

# Panel 1: Main concept visualization
ax1 = axes[0, 0]
# ... visualization code ...
ax1.set_title('Concept Name', fontsize=12, fontweight='bold')

# Panel 2: Alternative representation
ax2 = axes[0, 1]
# ... different view of same concept ...

# Panel 3: Example/Application
ax3 = axes[1, 0]
# ... real-world example ...

# Panel 4: Summary or comparison
ax4 = axes[1, 1]
# ... key insights ...

plt.tight_layout()
plt.show()

# Print key insight summary
print("\n📊 Key Insights:")
print("   • Insight 1")
print("   • Insight 2")
```

**Visualization Requirements:**
- Multiple subplots showing different aspects (2x2, 2x3, or 1x3 layouts)
- Color-coded elements with clear legends
- Annotations explaining "why" not just "what"
- Use appropriate chart types:
  - Sets → Venn diagrams (`matplotlib_venn`)
  - Relations/Functions → Directed graphs, bipartite graphs (`networkx`)
  - Matrices → Heatmaps (`seaborn`)
  - Hierarchies → Nested circles, Hasse diagrams
  - Distributions → Histograms, box plots
  - Comparisons → Bar charts, before/after plots
- Always print a summary box with key takeaways after the figure

#### Part D: Supply Chain Application (Python + Markdown)
Connect EVERY concept to real-world retail/supply chain scenarios:

**Application Domains:**
- Inventory Management: Stock levels, reorder points, safety stock
- Demand Forecasting: Sales trends, seasonality, promotions
- Warehouse Operations: SKU distribution, storage optimization
- Supplier Management: Lead times, reliability, sourcing
- Pricing: Tiers, discounts, elasticity
- Logistics: Routes, delivery times, fulfillment

```python
# Create realistic supply chain DataFrame
inventory_df = pd.DataFrame({
    'SKU': ['SKU001', 'SKU002', ...],
    'warehouse': ['WH-North', 'WH-South', ...],
    'quantity': [...],
    'reorder_point': [...],
})

# Apply the mathematical concept to supply chain problem
# ... analysis code ...

# Display results professionally
display(results_df)
```

### 4. Practice Exercises Section (3 Exercises Minimum)

#### Exercise Pattern:
```markdown
### Exercise [N]: [Title with Supply Chain Context]
**Problem**: [Clear problem statement using supply chain scenario]
**Mathematical Concept**: [Which concept this tests]
**Difficulty**: [Basic/Intermediate/Advanced]
```

```python
# Exercise [N] Solution
# ... solution code with comments ...
print("Solution: ...")
```

```python
# Exercise [N] Visualization - MANDATORY
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
# ... visualization of the solution ...
plt.show()
print("\n📊 Exercise Insight: [Key learning from this exercise]")
```

### 5. Summary Cell (Markdown)
```markdown
## Week [XX] Summary

| Concept | Definition | Key Property | Supply Chain Application |
|---------|------------|--------------|-------------------------|
| [Concept 1] | [Brief def] | [Main property] | [Application] |
| [Concept 2] | [Brief def] | [Main property] | [Application] |

### Key Takeaways
1. [Takeaway 1]
2. [Takeaway 2]
3. [Takeaway 3]

### Next Week Preview
- [Topic 1]
- [Topic 2]
```

## Visualization Philosophy

**CRITICAL**: The user learns best through visual representations. For EVERY mathematical concept:

1. **Show the Structure**: Use diagrams that reveal underlying patterns
2. **Use Color Coding**: Different colors for different categories/types
3. **Add Annotations**: Explain insights directly on the figure
4. **Multiple Views**: Show the same concept from different angles
5. **Before/After**: When applicable, show transformations
6. **Real Data**: Use supply chain examples, not abstract numbers

### Visualization Toolkit by Topic:

| Topic | Primary Visualizations |
|-------|----------------------|
| Sets | Venn diagrams, nested circles, bar charts |
| Relations | Directed graphs, matrices as heatmaps, bipartite graphs |
| Functions | Mapping diagrams, bipartite graphs, domain-range plots |
| Number Systems | Number lines, nested hierarchy circles |
| Logic | Truth table heatmaps, logic gate diagrams |
| Sequences | Line plots, convergence animations |
| Probability | Histograms, density plots, tree diagrams |
| Statistics | Box plots, scatter plots, confidence intervals |
| Linear Algebra | Vector plots, transformation grids, matrix heatmaps |
| Calculus | Function plots with tangent lines, area under curves |

## Content Quality Checklist

Before completing a weekly notebook, verify:

- [ ] Every concept has a VISUALIZATION cell
- [ ] All visualizations have multiple panels (not single plots)
- [ ] Every example uses supply chain context
- [ ] All functions have type hints and docstrings
- [ ] `np.random.seed(42)` is set for reproducibility
- [ ] DataFrames are displayed with `display()`, not `print()`
- [ ] Summary insights are printed after each visualization
- [ ] Exercises have both solutions AND visualizations
- [ ] Summary table covers all concepts

## Visualization Standards
- Always include titles, axis labels, and legends
- Use `fig, ax = plt.subplots()` pattern
- Add `plt.tight_layout()` before showing
- Save figures: `fig.savefig('week-XX-description.png', dpi=150, bbox_inches='tight')`

## Time Series Validation
```python
from sklearn.model_selection import TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(X):
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
```

## Cross-Sectional Validation
```python
from sklearn.model_selection import StratifiedKFold, KFold

# For classification (e.g., store tiers, SKU categories)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()

# For regression (e.g., sales prediction across stores)
kf = KFold(n_splits=5, shuffle=True, random_state=42)
```

## Output Best Practices
- Clear unnecessary outputs before committing
- Use `display()` for DataFrames, not `print()`
- Limit DataFrame displays: `df.head()`, `df.sample(5)`
