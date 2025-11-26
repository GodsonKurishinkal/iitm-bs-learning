---
applyTo: "**/*.py"
---

# Python Data Science Instructions

## Function Template
```python
def function_name(
    param1: pd.DataFrame,
    param2: int = 10
) -> pd.DataFrame:
    """Short description.

    Args:
        param1: Description of param1.
        param2: Description with default.

    Returns:
        Description of return value.

    Raises:
        ValueError: When validation fails.

    Example:
        >>> result = function_name(df, param2=5)
    """
    if param1.empty:
        raise ValueError("DataFrame cannot be empty")

    result = param1.copy()
    # ... logic
    return result
```

## Error Handling
```python
try:
    result = process_data(df)
except FileNotFoundError as e:
    logger.error(f"Data file not found: {e}")
    raise
except pd.errors.EmptyDataError as e:
    logger.warning(f"Empty data file: {e}")
    return pd.DataFrame()
```

## Logging Setup
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
```

## Constants Pattern
```python
# Constants at module level
FORECAST_HORIZON: int = 12
SAFETY_STOCK_FACTOR: float = 1.65
MIN_DEMAND_THRESHOLD: int = 0
```

## Path Handling
```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
```
