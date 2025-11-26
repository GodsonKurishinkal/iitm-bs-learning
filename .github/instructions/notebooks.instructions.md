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
