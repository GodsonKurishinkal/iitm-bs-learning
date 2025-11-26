# Python Data Science Patterns

> Extended patterns and examples for GitHub Copilot
> **Context**: Retail Supply Chain Data Scientist - Both Time Series & Cross-Sectional Data

---

## Data Type Selection Guide

| Data Type | When to Use | Retail SC Examples | Validation Method |
|-----------|-------------|-------------------|-------------------|
| **Time Series** | Temporal patterns, forecasting | Daily sales, inventory levels, demand | `TimeSeriesSplit` |
| **Cross-Sectional** | Snapshot comparisons, classification | Store attributes, SKU profiles, suppliers | `KFold`, `StratifiedKFold` |
| **Panel Data** | Both time + cross-section | Store-SKU daily sales | Grouped splits |

---

## Cross-Sectional Data Patterns (Retail Supply Chain)

### Store/SKU Analysis Template
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from typing import Tuple

np.random.seed(42)


def prepare_cross_sectional_features(
    df: pd.DataFrame,
    categorical_cols: list[str],
    numeric_cols: list[str]
) -> pd.DataFrame:
    """Prepare features for cross-sectional analysis.

    Use case: Analyzing store performance, SKU classification, supplier scoring.

    Args:
        df: DataFrame with entity-level data (stores, SKUs, suppliers).
        categorical_cols: Columns to one-hot encode.
        numeric_cols: Columns to standardize.

    Returns:
        Processed DataFrame ready for modeling.

    Example:
        >>> # Retail: Classify stores by performance tier
        >>> stores = prepare_cross_sectional_features(
        ...     store_df,
        ...     categorical_cols=['region', 'format'],
        ...     numeric_cols=['avg_sales', 'sqft', 'staff_count']
        ... )
    """
    result = df.copy()

    # One-hot encode categoricals
    result = pd.get_dummies(result, columns=categorical_cols, drop_first=True)

    # Standardize numerics
    scaler = StandardScaler()
    result[numeric_cols] = scaler.fit_transform(result[numeric_cols])

    return result


def cross_sectional_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    stratify: bool = True
) -> dict[str, float]:
    """Cross-validation for cross-sectional data.

    Use case: SKU classification, store clustering validation.

    Args:
        model: Scikit-learn compatible model.
        X: Feature DataFrame (each row = one entity).
        y: Target Series.
        n_splits: Number of CV folds.
        stratify: Use stratified splits for classification.

    Returns:
        Dictionary with mean and std of CV scores.
    """
    if stratify:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    else:
        from sklearn.model_selection import KFold
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

    return {
        'mean_score': round(scores.mean(), 4),
        'std_score': round(scores.std(), 4),
        'all_scores': scores.tolist()
    }
```

### ABC-XYZ Classification (Cross-Sectional)
```python
def abc_xyz_classification(
    df: pd.DataFrame,
    sku_col: str,
    revenue_col: str,
    demand_cv_col: str
) -> pd.DataFrame:
    """Classify SKUs using ABC-XYZ matrix.

    Cross-sectional analysis: Each SKU is one observation.
    ABC = Revenue contribution (Pareto)
    XYZ = Demand variability (CV)

    Args:
        df: DataFrame with SKU-level aggregated data.
        sku_col: SKU identifier column.
        revenue_col: Total revenue column.
        demand_cv_col: Coefficient of variation of demand.

    Returns:
        DataFrame with ABC and XYZ classifications.

    Example:
        >>> # Each row = one SKU (cross-sectional snapshot)
        >>> classified = abc_xyz_classification(
        ...     sku_summary, 'sku_id', 'annual_revenue', 'demand_cv'
        ... )
    """
    result = df.copy()

    # ABC Classification (cumulative revenue %)
    result = result.sort_values(revenue_col, ascending=False)
    result['revenue_cumsum'] = result[revenue_col].cumsum()
    result['revenue_pct'] = result['revenue_cumsum'] / result[revenue_col].sum()

    result['ABC'] = pd.cut(
        result['revenue_pct'],
        bins=[0, 0.7, 0.9, 1.0],
        labels=['A', 'B', 'C']
    )

    # XYZ Classification (demand variability)
    result['XYZ'] = pd.cut(
        result[demand_cv_col],
        bins=[0, 0.5, 1.0, float('inf')],
        labels=['X', 'Y', 'Z']
    )

    result['ABC_XYZ'] = result['ABC'].astype(str) + result['XYZ'].astype(str)

    return result[[sku_col, 'ABC', 'XYZ', 'ABC_XYZ', revenue_col, demand_cv_col]]
```

---

## Time Series Forecasting Patterns

### Demand Forecasting Template
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import Tuple

np.random.seed(42)


def prepare_time_series_features(
    df: pd.DataFrame,
    target_col: str,
    lags: list[int] = [1, 7, 14, 28]
) -> pd.DataFrame:
    """Create lag features for time series forecasting.

    Args:
        df: DataFrame with datetime index.
        target_col: Name of target column.
        lags: List of lag periods to create.

    Returns:
        DataFrame with lag features added.
    """
    result = df.copy()
    for lag in lags:
        result[f'{target_col}_lag_{lag}'] = result[target_col].shift(lag)
    return result.dropna()


def evaluate_forecast(
    y_true: pd.Series,
    y_pred: pd.Series
) -> dict[str, float]:
    """Calculate forecast accuracy metrics.

    Args:
        y_true: Actual values.
        y_pred: Predicted values.

    Returns:
        Dictionary with MAE, RMSE, MAPE metrics.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    return {
        'MAE': round(mae, 2),
        'RMSE': round(rmse, 2),
        'MAPE': round(mape, 2)
    }
```

### Cross-Validation for Time Series
```python
def time_series_cv(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5
) -> Tuple[list[float], list[float]]:
    """Perform time series cross-validation.

    Args:
        model: Scikit-learn compatible model.
        X: Feature DataFrame.
        y: Target Series.
        n_splits: Number of CV splits.

    Returns:
        Tuple of (train_scores, val_scores).
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    train_scores, val_scores = [], []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train = X.iloc[train_idx].copy()
        X_val = X.iloc[val_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_val = y.iloc[val_idx].copy()

        model.fit(X_train, y_train)

        train_scores.append(model.score(X_train, y_train))
        val_scores.append(model.score(X_val, y_val))

    return train_scores, val_scores
```

## Inventory Optimization Patterns

### Safety Stock Calculation
```python
def calculate_safety_stock(
    demand_std: float,
    lead_time: int,
    service_level: float = 0.95
) -> float:
    """Calculate safety stock based on service level.

    Args:
        demand_std: Standard deviation of demand.
        lead_time: Lead time in periods.
        service_level: Target service level (0-1).

    Returns:
        Safety stock quantity.

    Example:
        >>> safety = calculate_safety_stock(100, 7, 0.95)
    """
    from scipy import stats

    z_score = stats.norm.ppf(service_level)
    safety_stock = z_score * demand_std * np.sqrt(lead_time)

    return round(safety_stock, 0)


def economic_order_quantity(
    annual_demand: float,
    ordering_cost: float,
    holding_cost_rate: float,
    unit_cost: float
) -> float:
    """Calculate Economic Order Quantity (EOQ).

    Args:
        annual_demand: Annual demand in units.
        ordering_cost: Cost per order.
        holding_cost_rate: Annual holding cost as fraction of unit cost.
        unit_cost: Unit purchase cost.

    Returns:
        Optimal order quantity.
    """
    holding_cost = holding_cost_rate * unit_cost
    eoq = np.sqrt((2 * annual_demand * ordering_cost) / holding_cost)
    return round(eoq, 0)
```

## Statistical Analysis Patterns

### Descriptive Statistics
```python
def descriptive_summary(
    df: pd.DataFrame,
    numeric_cols: list[str] | None = None
) -> pd.DataFrame:
    """Generate comprehensive descriptive statistics.

    Args:
        df: Input DataFrame.
        numeric_cols: Columns to analyze (default: all numeric).

    Returns:
        Summary statistics DataFrame.
    """
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    stats = df[numeric_cols].agg([
        'count', 'mean', 'std', 'min',
        lambda x: x.quantile(0.25),
        'median',
        lambda x: x.quantile(0.75),
        'max'
    ])
    stats.index = ['count', 'mean', 'std', 'min', 'Q1', 'median', 'Q3', 'max']

    # Add sample std (ddof=1)
    sample_std = df[numeric_cols].std(ddof=1)
    stats.loc['sample_std'] = sample_std

    return stats.round(2)
```

### Hypothesis Testing Template
```python
from scipy import stats


def perform_t_test(
    group1: pd.Series,
    group2: pd.Series,
    alpha: float = 0.05,
    alternative: str = 'two-sided'
) -> dict:
    """Perform independent samples t-test.

    Args:
        group1: First sample.
        group2: Second sample.
        alpha: Significance level.
        alternative: 'two-sided', 'less', or 'greater'.

    Returns:
        Dictionary with test results.
    """
    t_stat, p_value = stats.ttest_ind(group1, group2, alternative=alternative)

    return {
        't_statistic': round(t_stat, 4),
        'p_value': round(p_value, 4),
        'alpha': alpha,
        'reject_null': p_value < alpha,
        'interpretation': 'Significant difference' if p_value < alpha else 'No significant difference'
    }
```

## Visualization Patterns

### Standard Plot Setup
```python
import matplotlib.pyplot as plt
import seaborn as sns


def setup_plot_style():
    """Configure consistent plot styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'axes.titlesize': 14,
        'axes.labelsize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'legend.fontsize': 10,
        'figure.dpi': 100
    })


def create_time_series_plot(
    df: pd.DataFrame,
    date_col: str,
    value_col: str,
    title: str,
    save_path: str | None = None
) -> plt.Figure:
    """Create standardized time series plot.

    Args:
        df: DataFrame with time series data.
        date_col: Column name for dates.
        value_col: Column name for values.
        title: Plot title.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(df[date_col], df[value_col], linewidth=1.5, color='#1f77b4')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Date')
    ax.set_ylabel(value_col)
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
```

---

## Retail Supply Chain Example Scenarios

### Time Series Examples (use `TimeSeriesSplit`)

| Scenario | Data Structure | Target Variable |
|----------|---------------|-----------------|
| Daily demand forecast | Date, SKU, Store, Qty | `daily_demand` |
| Inventory level prediction | Date, Warehouse, Stock | `stock_level` |
| Sales trend analysis | Week, Category, Revenue | `weekly_sales` |
| Seasonal pattern detection | Month, Product, Units | `monthly_units` |

### Cross-Sectional Examples (use `KFold`/`StratifiedKFold`)

| Scenario | Data Structure | Target Variable |
|----------|---------------|-----------------|
| Store performance tier | Store, Region, Sqft, Staff, Sales | `perf_tier` (A/B/C) |
| SKU profitability class | SKU, Category, Margin, Turnover | `profit_class` |
| Supplier reliability score | Supplier, LeadTime, Defects, OnTime | `reliability_score` |
| Markdown optimization | Product, Age, Stock, Price | `markdown_pct` |
| Assortment clustering | SKU, Attributes, Sales, Returns | `cluster_id` |

### Mapping Syllabus to Retail Examples

| Course Topic | Time Series Example | Cross-Sectional Example |
|-------------|--------------------|-----------------------|
| **Descriptive Stats** | Weekly sales distribution | Store attribute summary |
| **Probability** | Stockout probability | Defect rate by supplier |
| **Hypothesis Testing** | Pre/post promo sales | Regional performance diff |
| **Regression** | Demand vs price elasticity | Sales vs store features |
| **Classification** | Trend direction (up/down) | SKU category prediction |
| **Clustering** | Seasonal pattern groups | Store segmentation |
