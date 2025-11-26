# IIT Madras BS Data Science - Copilot Instructions

## Project Overview
Educational repository for IIT Madras BS Degree in Data Science. Focus: Supply Chain Analytics, Demand Forecasting, S&OP, Inventory Optimization.

## Folder Structure
- `01-Foundation-Level/` to `06-MTech-Level/`: Course materials by level
- Each level: `notebooks/` (Jupyter), `resources/` (JSON course data)
- `docs/`: Documentation and course catalog

## Tech Stack
- **Language**: Python 3.12+
- **Core**: NumPy, Pandas, Matplotlib, Seaborn, SciPy
- **ML/Stats**: Scikit-learn, Statsmodels
- **Notebooks**: Jupyter with interactive widgets

## Coding Standards

### Always
- Type hints + Google docstrings for all functions
- `np.random.seed(42)` for reproducibility
- `.copy()` when filtering DataFrames
- `TimeSeriesSplit` for time series data (never random split)
- `StratifiedKFold` or `KFold` for cross-sectional data
- `np.std(x, ddof=1)` for sample standard deviation
- `df.loc[mask, 'col']` for assignment (not chained indexing)

### Never
- Bare `except:` blocks
- Hardcoded file paths
- Magic numbers without constants
- `print()` for debugging (use `logging`)

## Naming Conventions
- Notebooks: `week-XX-topic-name.ipynb`
- Data files: `data_name_YYYYMMDD.csv`
- Figures: `week-XX-description.png`

## Examples Context
**Primary Goal**: Master IIT Madras syllabus with solid theoretical foundation.
**Application Context**: Retail Supply Chain Data Scientist real-world scenarios.

### Example Domains (use these for worked examples)
**Time Series**: Demand forecasting, sales trends, seasonal patterns, stock levels
**Cross-Sectional**: Store performance, SKU attributes, supplier analysis, pricing
