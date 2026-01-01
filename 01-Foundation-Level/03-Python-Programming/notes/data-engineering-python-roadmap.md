# Data Engineering Python Roadmap

> **Goal:** Master Python for production data pipelines  
> **Timeline:** 8 weeks (parallel with DP-700 prep)  
> **Status:** 🟢 In Progress

---

## 📋 Curriculum Overview

| Week | Module | Focus | Your Status |
|------|--------|-------|-------------|
| 1 | Python Fundamentals | Variables, Types, Casting | ✅ Review |
| 2 | Control Flow | Loops, Conditionals, Nested Iteration | ✅ Review |
| 3 | Data Structures | Dicts, Lists for Config/Metadata | ✅ Review |
| 4 | File I/O & Connectivity | CSV, JSON, Parquet, SQL, Kafka | ✅ Strengthen |
| 5 | Error Handling & Logging | try/except, logging library | ✅ Strengthen |
| 6 | Modular Coding | Generic UDFs, ABC Patterns | ✅ Expert |
| 7-8 | PySpark | Sessions, DataFrames, SQL | 🔴 **PRIORITY** |

---

## 🎯 Module 1: Python Fundamentals (Review)

### Variables & Basic Data Types
```python
# Type hints for production code
from typing import Dict, List, Optional

# Configuration-driven approach (your pattern)
pipeline_name: str = "orders_etl"
batch_size: int = 10000
is_incremental: bool = True
threshold: float = 0.95
```

### Type Casting (Critical for Data Engineering)
```python
# String to numeric (common in ETL)
amount_str = "1234.56"
amount = float(amount_str)  # 1234.56
amount_cents = int(float(amount_str) * 100)  # 123456

# Date parsing
from datetime import datetime
date_str = "2026-01-01"
date_obj = datetime.strptime(date_str, "%Y-%m-%d")

# Safe casting with error handling
def safe_int(value, default=0):
    try:
        return int(value)
    except (ValueError, TypeError):
        return default
```

### Practice Exercise
- [ ] Create a config dictionary with mixed types
- [ ] Implement safe type casting functions
- [ ] Add type hints to existing code

---

## 🎯 Module 2: Control Flow

### Loops for Table/Column Iteration (Your Production Pattern)
```python
# Iterating tables from config
TABLES = ["orders", "inventory", "shipments"]
COLUMNS = {
    "orders": ["order_id", "amount", "created_at"],
    "inventory": ["sku", "quantity", "warehouse"]
}

# Single loop - table processing
for table in TABLES:
    df = extract(table)
    transform(df)
    load(df, f"bronze/{table}")

# Nested loop - column validation
for table, columns in COLUMNS.items():
    for column in columns:
        validate_column(table, column)
        print(f"✓ {table}.{column} validated")
```

### Conditional Statements (Pipeline Logic)
```python
# Load mode selection
load_mode = config.get("load_mode", "FULL")

if load_mode == "FULL":
    df = extract_full(table)
elif load_mode == "INCREMENTAL":
    df = extract_incremental(table, last_run_date)
else:
    raise ValueError(f"Unknown load mode: {load_mode}")

# Ternary for concise logic
output_path = f"data/{env}/output" if env != "prod" else "data/prod/output"
```

### Practice Exercise
- [ ] Build a multi-table ETL loop
- [ ] Implement FULL/INCREMENTAL mode switching
- [ ] Create nested validation loops

---

## 🎯 Module 3: Data Structures for Config/Metadata

### Dictionaries (Your Config Pattern)
```python
# Pipeline configuration (from your portfolio)
PIPELINE = {
    "load": {"mode": "FULL"},
    "paths": {"output": "data/landing/orders"},
    
    "schema": {
        "order_id": "string",
        "amount": "decimal(10,2)",
        "country": "string",
        "created_at": "timestamp"
    },
    
    "quality": {
        "required_columns": ["order_id", "amount"],
        "not_null": ["order_id"]
    }
}

# Accessing nested config
schema = PIPELINE["schema"]
required = PIPELINE["quality"]["required_columns"]

# Safe access with .get()
mode = PIPELINE.get("load", {}).get("mode", "FULL")
```

### Lists (Batch Processing)
```python
# Table list for processing
tables_to_process = ["orders", "inventory", "shipments"]

# List comprehension for transformation
columns_upper = [col.upper() for col in columns]
valid_tables = [t for t in tables if t not in EXCLUDE_LIST]

# Filtering with conditions
numeric_columns = [
    col for col, dtype in schema.items() 
    if dtype in ("int", "float", "decimal")
]
```

### Combining Structures
```python
# Metadata registry pattern
TABLE_REGISTRY = {
    "orders": {
        "source": "postgres",
        "columns": ["order_id", "amount", "created_at"],
        "primary_key": "order_id",
        "partition_by": "created_at"
    },
    "inventory": {
        "source": "mysql",
        "columns": ["sku", "qty", "warehouse"],
        "primary_key": "sku",
        "partition_by": None
    }
}

# Dynamic processing
for table, metadata in TABLE_REGISTRY.items():
    source = metadata["source"]
    pk = metadata["primary_key"]
    process_table(table, source, pk)
```

### Practice Exercise
- [ ] Create a TABLE_REGISTRY for your domain
- [ ] Implement config-driven schema validation
- [ ] Build metadata lookup functions

---

## 🎯 Module 4: File I/O & Connectivity

### CSV/JSON/Parquet (Your Daily Tools)
```python
import pandas as pd
import polars as pl
import json

# CSV
df_pd = pd.read_csv("data/orders.csv")
df_pl = pl.read_csv("data/orders.csv")

# JSON (config files)
with open("config/pipeline.json", "r") as f:
    config = json.load(f)

with open("output/metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

# Parquet (your preferred format)
df = pl.read_parquet("data/bronze/orders/*.parquet")
df.write_parquet(
    "data/silver/orders",
    partition_by=["year", "month"]
)
```

### SQL Database Connectivity
```python
import duckdb
from sqlalchemy import create_engine

# DuckDB (your go-to)
con = duckdb.connect("warehouse.duckdb")
df = con.execute("SELECT * FROM orders WHERE amount > 100").pl()

# PostgreSQL via SQLAlchemy
engine = create_engine("postgresql://user:pass@host:5432/db")
df = pd.read_sql("SELECT * FROM orders", engine)

# Connection from config
DB_CONFIG = {
    "host": "db.company.internal",
    "port": 5432,
    "database": "orders_db",
    "user": os.getenv("DB_USER"),
    "password": os.getenv("DB_PASS")
}

conn_string = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
```

### Kafka (Streaming - Learning)
```python
from confluent_kafka import Consumer, Producer

# Consumer config
consumer_config = {
    'bootstrap.servers': 'localhost:9092',
    'group.id': 'etl-consumer',
    'auto.offset.reset': 'earliest'
}

consumer = Consumer(consumer_config)
consumer.subscribe(['orders-topic'])

# Poll messages
while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    process_message(msg.value())
```

### Practice Exercise
- [ ] Build multi-format reader function
- [ ] Create database connection factory
- [ ] Implement config-driven file I/O

---

## 🎯 Module 5: Error Handling & Logging

### Production Error Handling Pattern
```python
import logging
from typing import Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def extract_table(table: str) -> Optional[pl.DataFrame]:
    """Extract table with comprehensive error handling."""
    logger.info(f"Starting extraction: {table}")
    
    try:
        df = read_from_source(table)
        logger.info(f"Extracted {len(df)} rows from {table}")
        return df
        
    except ConnectionError as e:
        logger.error(f"Connection failed for {table}: {e}")
        raise
        
    except FileNotFoundError as e:
        logger.warning(f"Source not found for {table}: {e}")
        return None
        
    except Exception as e:
        logger.exception(f"Unexpected error extracting {table}")
        raise
        
    finally:
        logger.debug(f"Extraction attempt completed for {table}")
```

### Retry Pattern (Production Essential)
```python
import time
from functools import wraps

def retry(max_attempts=3, delay=1, backoff=2):
    """Decorator for retry logic."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            current_delay = delay
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        logger.error(f"Failed after {max_attempts} attempts")
                        raise
                    logger.warning(f"Attempt {attempts} failed, retrying in {current_delay}s")
                    time.sleep(current_delay)
                    current_delay *= backoff
                    
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def connect_to_database():
    return create_connection(DB_CONFIG)
```

### Practice Exercise
- [ ] Add logging to existing pipeline
- [ ] Implement retry decorator
- [ ] Create error classification system

---

## 🎯 Module 6: Modular Coding (Your Strength)

### Abstract Base Class Pattern (Your Production Pattern)
```python
from abc import ABC, abstractmethod
from typing import Dict, Any

class BaseExtractor(ABC):
    """Abstract base for all extractors."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(self.__class__.__name__)
    
    @abstractmethod
    def connect(self) -> None:
        """Establish connection to source."""
        pass
    
    @abstractmethod
    def extract(self, query: str) -> pl.DataFrame:
        """Extract data from source."""
        pass
    
    def validate(self, df: pl.DataFrame) -> bool:
        """Common validation logic."""
        required = self.config.get("required_columns", [])
        return all(col in df.columns for col in required)


class PostgresExtractor(BaseExtractor):
    """PostgreSQL implementation."""
    
    def connect(self):
        self.conn = create_engine(self.config["connection_string"])
        self.logger.info("Connected to PostgreSQL")
    
    def extract(self, query: str) -> pl.DataFrame:
        return pl.read_database(query, self.conn)


class FileExtractor(BaseExtractor):
    """File-based implementation."""
    
    def connect(self):
        self.base_path = self.config["base_path"]
        self.logger.info(f"Using base path: {self.base_path}")
    
    def extract(self, query: str) -> pl.DataFrame:
        # query is filename pattern here
        return pl.read_parquet(f"{self.base_path}/{query}")
```

### Generic UDF Library
```python
# utils/transformations.py

def standardize_column_names(df: pl.DataFrame) -> pl.DataFrame:
    """Lowercase and snake_case all column names."""
    return df.rename({
        col: col.lower().replace(" ", "_").replace("-", "_")
        for col in df.columns
    })

def add_audit_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Add standard audit columns."""
    return df.with_columns([
        pl.lit(datetime.now()).alias("_loaded_at"),
        pl.lit(PIPELINE_RUN_ID).alias("_pipeline_run_id")
    ])

def validate_schema(df: pl.DataFrame, expected: Dict[str, str]) -> bool:
    """Validate DataFrame against expected schema."""
    for col, dtype in expected.items():
        if col not in df.columns:
            return False
        # Add type validation
    return True
```

### Practice Exercise
- [ ] Create ABC for your extractors
- [ ] Build generic transformation library
- [ ] Implement factory pattern for sources

---

## 🎯 Module 7-8: PySpark (Priority Focus)

### Session Management
```python
from pyspark.sql import SparkSession

# Create session
spark = SparkSession.builder \
    .appName("orders_etl") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# Session from config
def create_spark_session(config: Dict) -> SparkSession:
    builder = SparkSession.builder.appName(config["app_name"])
    
    for key, value in config.get("spark_config", {}).items():
        builder = builder.config(key, value)
    
    return builder.getOrCreate()
```

### DataFrame Operations (Mapping to Polars)
```python
from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DecimalType

# Read from JDBC (your portfolio pattern)
df = spark.read \
    .format("jdbc") \
    .option("url", DB_CONFIG["url"]) \
    .option("dbtable", "orders") \
    .option("user", DB_CONFIG["user"]) \
    .option("password", DB_CONFIG["password"]) \
    .load()

# Transformations (similar to Polars)
df_transformed = df \
    .filter(F.col("amount") > 0) \
    .withColumn("amount_usd", F.col("amount") * F.lit(3.67)) \
    .withColumn("year", F.year("created_at")) \
    .withColumn("month", F.month("created_at"))

# Aggregations
summary = df \
    .groupBy("country", "year") \
    .agg(
        F.sum("amount").alias("total_amount"),
        F.count("order_id").alias("order_count"),
        F.avg("amount").alias("avg_amount")
    )
```

### Joins (Critical for Data Engineering)
```python
# Read dimension tables
orders = spark.read.parquet("data/orders")
customers = spark.read.parquet("data/customers")
products = spark.read.parquet("data/products")

# Inner join
orders_with_customer = orders.join(
    customers,
    orders.customer_id == customers.id,
    "inner"
)

# Left join with alias
from pyspark.sql.functions import col

orders_enriched = orders.alias("o") \
    .join(
        customers.alias("c"),
        col("o.customer_id") == col("c.id"),
        "left"
    ) \
    .join(
        products.alias("p"),
        col("o.product_id") == col("p.id"),
        "left"
    ) \
    .select(
        col("o.order_id"),
        col("o.amount"),
        col("c.customer_name"),
        col("p.product_name")
    )
```

### SQL Integration in PySpark
```python
# Register as temp view
df.createOrReplaceTempView("orders")
customers.createOrReplaceTempView("customers")

# Use SQL directly
result = spark.sql("""
    SELECT 
        o.order_id,
        o.amount,
        c.customer_name,
        DATE_FORMAT(o.created_at, 'yyyy-MM') as order_month
    FROM orders o
    LEFT JOIN customers c ON o.customer_id = c.id
    WHERE o.amount > 100
    ORDER BY o.created_at DESC
""")

# Complex aggregation via SQL
monthly_summary = spark.sql("""
    SELECT 
        DATE_TRUNC('month', created_at) as month,
        country,
        COUNT(*) as order_count,
        SUM(amount) as total_amount,
        AVG(amount) as avg_amount
    FROM orders
    GROUP BY 1, 2
    ORDER BY 1 DESC, 3 DESC
""")
```

### Write Operations
```python
# Write to Parquet (partitioned)
df.write \
    .mode("overwrite") \
    .partitionBy("year", "month") \
    .parquet("data/silver/orders")

# Write to Delta (if using Databricks)
df.write \
    .format("delta") \
    .mode("merge") \
    .save("data/gold/orders")
```

### Practice Exercise
- [ ] Set up local PySpark environment
- [ ] Migrate one Polars pipeline to PySpark
- [ ] Practice multi-table joins
- [ ] Implement SQL-based transformations

---

## 📅 Weekly Study Schedule

| Day | Morning (1hr) | Evening (1hr) |
|-----|--------------|---------------|
| Mon | Theory/Concepts | Code Practice |
| Tue | Hands-on Lab | Review & Notes |
| Wed | Theory/Concepts | Code Practice |
| Thu | Hands-on Lab | Review & Notes |
| Fri | Project Work | Documentation |
| Sat | Deep Dive Topic | Portfolio Update |
| Sun | Review Week | Plan Next Week |

---

## 🔗 Resources

### Documentation
- [Polars User Guide](https://pola-rs.github.io/polars-book/)
- [PySpark Documentation](https://spark.apache.org/docs/latest/api/python/)
- [DuckDB Documentation](https://duckdb.org/docs/)

### Practice Platforms
- [LeetCode SQL](https://leetcode.com/problemset/database/)
- [Databricks Academy](https://www.databricks.com/learn)
- [Microsoft Learn - Fabric](https://learn.microsoft.com/en-us/training/paths/get-started-fabric/)

### Your Portfolio Projects
- Medallion Lakehouse (uses all patterns)
- Demand Forecasting (ML + data engineering)
- Anomaly Detection (data quality patterns)

---

## ✅ Progress Tracker

- [ ] Week 1: Python Fundamentals Review
- [ ] Week 2: Control Flow Mastery
- [ ] Week 3: Data Structures for Config
- [ ] Week 4: File I/O & Connectivity
- [ ] Week 5: Error Handling & Logging
- [ ] Week 6: Modular Coding Patterns
- [ ] Week 7: PySpark Basics
- [ ] Week 8: PySpark Advanced + SQL

---

*Last Updated: 2026-01-01*
