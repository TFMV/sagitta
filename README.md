# Sagitta 🏹

> ⚠️ **Experimental Project** - This project is in early development and uses features that are currently in preview/experimental status.

## Overview

Sagitta is a data pipeline tool that demonstrates efficient data transfer between Oracle and PostgreSQL databases using Apache Arrow. The project showcases the use of ADBC (Arrow Database Connectivity) for high-performance data movement.

## ⚠️ Important Notes

- **Experimental Status**: This project is in its infancy and should not be used in production environments without thorough testing.
- **Oracle Arrow Integration**: The Arrow integration with Oracle is currently in preview status. Features and APIs may change without notice.
- **Performance Testing**: While Arrow promises significant performance improvements, this implementation is primarily for demonstration purposes.

## Early Results

Initial performance testing shows promising results for data transfer between Oracle and PostgreSQL using Apache Arrow:

### Performance Breakdown

The latest performance metrics show the following breakdown:

| Operation           | Time (seconds) | Percentage |
|---------------------|----------------|------------|
| Oracle connection   | 0.50           | 60.6%      |
| Oracle data read    | 0.03           | 3.8%       |
| Postgres connection | 0.06           | 7.2%       |
| Postgres data write | 0.01           | 0.9%       |
| Polars DataFrame    | 0.02           | 2.3%       |
| Polars Series       | 0.02           | 1.9%       |
| Visualization       | 0.19           | 23.2%      |
| **Total**           | **0.82**       | **100%**   |

This demonstrates that most of the time is spent in Oracle connection operations (60.6%), while data processing operations are significantly faster. The Arrow-based data transfer to PostgreSQL is particularly efficient, taking less than 1% of the total execution time. Visualization operations account for 23.2% of the total time, making them the second most time-consuming component.

### Network Impact

A significant portion of the Oracle operation time is due to network latency when connecting to Oracle Cloud Infrastructure (OCI). Local PostgreSQL operations are much faster by comparison. Potential optimizations include:

- Batch processing to amortize connection costs
- Connection pooling for Oracle
- Implementing a persistent connection strategy
- Optimizing query patterns to reduce network roundtrips
- Considering data locality to reduce cross-cloud data movement

## Prerequisites

- Python 3.11+
- Oracle Instant Client
- PostgreSQL Server
- Oracle Database (with Arrow support)

## Dependencies

```bash
pyarrow
oracledb
python-dotenv
adbc-driver-postgresql
pandas
polars
matplotlib
seaborn
```

## Configuration

Create a `.env` file in the project root with the following variables:

```env
ORACLE_DSN=your_oracle_dsn
ORACLE_USERNAME=your_username
ORACLE_PASSWORD=your_password
ORACLE_TNS_ADMIN=path_to_oracle_wallet
POSTGRES_CONNECTION_STRING=postgresql://user:password@host:port/database
```

See .env.example for reference.

## Features

- Arrow-based data transfer between Oracle and PostgreSQL
- Environment-based configuration
- Proper connection management and error handling
- Structured logging
- Polars integration for high-performance data analysis
- Automatic data visualization with matplotlib and seaborn

## Advanced Features

### Polars Integration with Zero-Copy

Sagitta includes integration with [Polars](https://pola.rs/), a lightning-fast DataFrame library written in Rust, using a zero-copy approach for maximum performance:

```python
# Extract data from Oracle directly to Polars using zero-copy approach
with oracle_connection(oracle_cfg) as oracle_conn:
    # Get data as a Polars DataFrame
    customers_pl = ora_to_polars(oracle_conn, "SELECT * FROM customers")
    
    # Or get a single column as a Polars Series
    customer_ids = ora_to_polars_series(oracle_conn, "SELECT customer_id FROM customers", "customer_id")
    
    # Perform high-performance data operations
    filtered = customers_pl.filter(pl.col("status") == "ACTIVE")
    aggregated = customers_pl.group_by("region").agg([
        pl.count().alias("customer_count"),
        pl.sum("total_spend").alias("revenue")
    ])
    
    # Series operations
    if (customer_ids > 0).all():
        log_ids = customer_ids.log10()
```

#### Zero-Copy Data Flow

The data flows through the system with minimal copying:

1. Oracle → OracleDataFrame (using native Oracle Arrow support)
2. OracleDataFrame → PyArrow Table/Array (zero-copy conversion)
3. PyArrow → Polars DataFrame/Series (zero-copy conversion)

This approach maximizes performance by avoiding unnecessary data copies between memory regions.

#### Benefits of the Polars Integration

- **Performance**: 10-100x faster than pandas for many operations
- **Memory Efficiency**: Processes large datasets with minimal memory overhead
- **Zero-Copy**: Maintains data in Arrow format throughout the pipeline
- **Lazy Evaluation**: Supports deferred execution for complex query optimization
- **Expressive API**: Provides a rich, intuitive interface for data manipulation

### Automatic Visualization

Sagitta can automatically generate visualizations from your data:

```python
# Generate visualizations from Polars DataFrame
viz_path = visualize_data(customers_pl, "Customer Analysis")
```

The visualization engine:

- Automatically detects appropriate chart types based on data
- Creates correlation matrices for numeric columns
- Generates distribution plots and histograms
- Produces bar charts for categorical data
- Saves visualizations to files for easy sharing

## Limitations

1. Oracle Arrow support is in preview
2. Limited error recovery mechanisms
3. No automatic schema mapping
4. Basic data type support only

## License

MIT

## Disclaimer

This project uses experimental features and is not recommended for production use. The Oracle Arrow integration is in preview status and may have limitations or unexpected behaviors.
