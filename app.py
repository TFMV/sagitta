import logging
import os
import time
from dataclasses import dataclass
from typing import Any, List, Tuple
from contextlib import contextmanager
from pathlib import Path

import oracledb
import pandas as pd
import pyarrow as pa
import adbc_driver_postgresql.dbapi
from dotenv import load_dotenv
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns


# Configuration classes
@dataclass
class OracleConfig:
    username: str
    password: str
    dsn: str
    tns_admin: str

    @classmethod
    def from_env(cls) -> "OracleConfig":
        load_dotenv()
        return cls(
            username=os.getenv("ORACLE_USERNAME", ""),
            password=os.getenv("ORACLE_PASSWORD", ""),
            dsn=os.getenv("ORACLE_DSN", ""),
            tns_admin=os.getenv("ORACLE_TNS_ADMIN", ""),
        )


@dataclass
class PostgresConfig:
    connection_string: str

    @classmethod
    def from_env(cls) -> "PostgresConfig":
        load_dotenv()
        return cls(
            connection_string=os.getenv(
                "POSTGRES_CONNECTION_STRING",
                "postgresql://postgres:postgres@localhost:5432/postgres",
            )
        )


def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# Performance tracking
performance_metrics = {
    "oracle_connect": 0,
    "oracle_read": 0,
    "postgres_connect": 0,
    "postgres_write": 0,
    "ora_to_polars": 0,
    "ora_to_polars_series": 0,
    "visualization": 0,
    "total": 0,
}


def time_it(metric_name=None):
    """Decorator to measure execution time of functions and store in performance metrics."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Store in performance metrics if a metric name is provided
            if metric_name:
                # For 'total', just set the value directly
                if metric_name == "total":
                    performance_metrics[metric_name] = elapsed_time
                else:
                    # For other metrics, accumulate the values
                    performance_metrics[metric_name] += elapsed_time

            logging.info(f"{func.__name__} completed in {elapsed_time:.2f} seconds")
            return result

        return wrapper

    return decorator


@contextmanager
def oracle_connection(config: OracleConfig):
    """Create and manage an Oracle connection."""
    conn = None
    start_time = time.time()
    try:
        # Initialize the Oracle client with the wallet directory
        oracledb.init_oracle_client(lib_dir=None, config_dir=config.tns_admin)

        # Connect using the wallet configuration with SSL options
        conn = oracledb.connect(
            user=config.username,
            password=config.password,
            dsn=config.dsn,
            ssl_server_dn_match=False,  # Disable server DN matching for SSL
        )
        end_time = time.time()
        performance_metrics["oracle_connect"] = end_time - start_time
        logging.info(f"Connected to Oracle in {end_time - start_time:.2f} seconds")
        yield conn
    except Exception as e:
        logging.error(f"Oracle connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()


@contextmanager
def postgres_connection(config: PostgresConfig):
    """Create and manage a PostgreSQL connection."""
    conn = None
    start_time = time.time()
    try:
        conn = adbc_driver_postgresql.dbapi.connect(config.connection_string)
        end_time = time.time()
        performance_metrics["postgres_connect"] = end_time - start_time
        logging.info(f"Connected to PostgreSQL in {end_time - start_time:.2f} seconds")
        yield conn
    except Exception as e:
        logging.error(f"PostgreSQL connection error: {e}")
        raise
    finally:
        if conn:
            conn.close()


@time_it(metric_name="oracle_read")
def fetch_oracle_data(
    conn: oracledb.Connection, query: str
) -> Tuple[List[Any], List[str]]:
    """Fetch data from Oracle."""
    with conn.cursor() as cursor:
        cursor.execute(query)
        results = cursor.fetchall()
        # Convert column names to lowercase to avoid case sensitivity issues
        columns = [i[0].lower() for i in cursor.description]
        return results, columns


@time_it(metric_name="postgres_write")
def pg_ingest_data(conn: Any, table_name: str, data: pa.Table) -> int:
    """Ingest data into PostgreSQL using Arrow."""
    with conn.cursor() as cursor:
        # Set schema and drop existing table
        cursor.execute("SET search_path TO public")
        cursor.execute(f"DROP TABLE IF EXISTS {table_name}")

        # Now insert the data using COPY
        rows_affected = cursor.adbc_ingest(table_name, data, "create_append")
        conn.commit()

        return rows_affected


def print_performance_report():
    """Print a formatted performance report."""
    # Calculate the sum of all components except total
    component_sum = sum(v for k, v in performance_metrics.items() if k != "total")

    # Use the component sum for percentage calculations to ensure they add up to 100%
    logging.info("\n" + "=" * 50)
    logging.info("PERFORMANCE REPORT")
    logging.info("=" * 50)
    logging.info(
        f"Oracle connection:  {performance_metrics['oracle_connect']:.2f}s ({performance_metrics['oracle_connect']/component_sum*100:.1f}%)"
    )
    logging.info(
        f"Oracle data read:   {performance_metrics['oracle_read']:.2f}s ({performance_metrics['oracle_read']/component_sum*100:.1f}%)"
    )
    logging.info(
        f"Postgres connection:{performance_metrics['postgres_connect']:.2f}s ({performance_metrics['postgres_connect']/component_sum*100:.1f}%)"
    )
    logging.info(
        f"Postgres data write:{performance_metrics['postgres_write']:.2f}s ({performance_metrics['postgres_write']/component_sum*100:.1f}%)"
    )

    # Add Polars operations to the report
    if performance_metrics["ora_to_polars"] > 0:
        logging.info(
            f"Polars DataFrame:   {performance_metrics['ora_to_polars']:.2f}s ({performance_metrics['ora_to_polars']/component_sum*100:.1f}%)"
        )
    if performance_metrics["ora_to_polars_series"] > 0:
        logging.info(
            f"Polars Series:      {performance_metrics['ora_to_polars_series']:.2f}s ({performance_metrics['ora_to_polars_series']/component_sum*100:.1f}%)"
        )
    if performance_metrics["visualization"] > 0:
        logging.info(
            f"Visualization:      {performance_metrics['visualization']:.2f}s ({performance_metrics['visualization']/component_sum*100:.1f}%)"
        )

    logging.info("-" * 50)
    logging.info(f"Total:              {component_sum:.2f}s (100%)")
    logging.info("=" * 50)


@time_it(metric_name="ora_to_polars")
def ora_to_polars(conn: oracledb.Connection, query: str) -> pl.DataFrame:
    """
    Fetch data from Oracle and convert to Polars DataFrame using zero-copy approach.

    Args:
        conn: Oracle connection
        query: SQL query to execute

    Returns:
        Polars DataFrame with query results
    """
    try:
        # Fetch data from Oracle as OracleDataFrame
        logging.info(f"Executing query: {query}")
        odf = conn.fetch_df_all(statement=query, arraysize=1000)

        # Convert to PyArrow Table using column arrays
        arrow_table = pa.Table.from_arrays(
            odf.column_arrays(), names=odf.column_names()
        )

        # Convert PyArrow Table to Polars DataFrame
        p = pl.from_arrow(arrow_table)

        logging.info(
            f"Converted Oracle data to Polars DataFrame with {p.shape[0]} rows and {p.shape[1]} columns"
        )
        return p
    except Exception as e:
        logging.error(f"Error in ora_to_polars: {e}")
        raise


@time_it(metric_name="ora_to_polars_series")
def ora_to_polars_series(
    conn: oracledb.Connection, query: str, column_name: str
) -> pl.Series:
    """
    Fetch a single column from Oracle and convert to Polars Series using zero-copy approach.

    Args:
        conn: Oracle connection
        query: SQL query to execute (should return a single column)
        column_name: Name of the column to extract

    Returns:
        Polars Series with query results
    """
    try:
        # Fetch data from Oracle as OracleDataFrame
        logging.info(f"Executing query for series: {query}")
        odf = conn.fetch_df_all(statement=query, arraysize=1000)

        # Convert to PyArrow Array for the specific column
        arrow_array = pa.array(odf.get_column_by_name(column_name))

        # Convert PyArrow Array to Polars Series
        series = pl.from_arrow(arrow_array)

        logging.info(
            f"Converted Oracle data to Polars Series with {series.shape[0]} rows"
        )
        return series
    except Exception as e:
        logging.error(f"Error in ora_to_polars_series: {e}")
        raise


@time_it(metric_name="visualization")
def visualize_data(df: pl.DataFrame, title: str = "Data Visualization") -> str:
    """
    Create visualizations from a Polars DataFrame and save to a file.

    Args:
        df: Polars DataFrame to visualize
        title: Title for the visualization

    Returns:
        Path to the saved visualization file
    """
    try:
        # Create a temporary directory for visualizations
        viz_dir = Path("visualizations")
        viz_dir.mkdir(exist_ok=True)

        # Create a unique filename
        timestamp = int(time.time())
        filename = viz_dir / f"{title.lower().replace(' ', '_')}_{timestamp}.png"

        # Set up the plot
        plt.figure(figsize=(12, 8))
        sns.set_style("whitegrid")

        # Determine what kind of visualization to create based on the data
        numeric_cols = [
            col for col in df.columns if df[col].dtype in [pl.Int64, pl.Float64]
        ]

        if len(numeric_cols) >= 2:
            # Create a correlation heatmap for numeric columns
            corr_df = df.select(numeric_cols).to_pandas().corr()
            plt.subplot(2, 1, 1)
            sns.heatmap(corr_df, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
            plt.title(f"Correlation Matrix - {title}")

            # Create a histogram of a numeric column
            plt.subplot(2, 1, 2)
            sample_col = numeric_cols[0]
            sns.histplot(df[sample_col].to_numpy(), kde=True)
            plt.title(f"Distribution of {sample_col}")
        elif len(numeric_cols) == 1:
            # Create a histogram for the single numeric column
            sns.histplot(df[numeric_cols[0]].to_numpy(), kde=True)
            plt.title(f"Distribution of {numeric_cols[0]} - {title}")
        else:
            # Create a count plot for a categorical column
            if len(df.columns) > 0:
                sample_col = df.columns[0]
                # Get value counts and convert to pandas for plotting
                value_counts = df[sample_col].value_counts().to_pandas()
                # Limit to top 10 categories if there are many
                if len(value_counts) > 10:
                    value_counts = value_counts.head(10)
                sns.barplot(x=value_counts.index, y=value_counts.values)
                plt.title(f"Top Categories in {sample_col} - {title}")
                plt.xticks(rotation=45)

        # Save the plot
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

        logging.info(f"Visualization saved to {filename}")
        return str(filename)
    except Exception as e:
        logging.error(f"Error creating visualization: {e}")
        return "Visualization failed"


@time_it(metric_name="total")
def main():
    """Main execution function."""
    setup_logging()
    oracle_cfg = OracleConfig.from_env()
    postgres_cfg = PostgresConfig.from_env()

    # Reset performance metrics for this run
    for key in performance_metrics:
        if key != "total":  # Keep the total which will be set by the decorator
            performance_metrics[key] = 0

    try:
        # Extract data from Oracle using traditional method
        with oracle_connection(oracle_cfg) as oracle_conn:
            results, columns = fetch_oracle_data(
                oracle_conn, "SELECT * FROM tfmv.customers"
            )
            # Create DataFrame with lowercase column names
            customers_df = pd.DataFrame(results, columns=columns)
            logging.info(f"Retrieved {len(customers_df)} rows from Oracle")

        # Load data into PostgreSQL
        with postgres_connection(postgres_cfg) as pg_conn:
            customers_table = pa.Table.from_pandas(customers_df)
            rows_affected = pg_ingest_data(pg_conn, "c_customer", customers_table)
            logging.info(f"Inserted {rows_affected} rows into PostgreSQL")

        # Demo the Polars integration with visualization
        logging.info("\n=== POLARS INTEGRATION DEMO ===")

        # Track the second Oracle connection separately
        oracle_connect_start = time.time()
        with oracle_connection(oracle_cfg) as oracle_conn:
            oracle_connect_end = time.time()
            # Add this connection time to the existing metric
            performance_metrics["oracle_connect"] += (
                oracle_connect_end - oracle_connect_start
            )

            # Get customer data with Polars DataFrame
            logging.info("1. Loading data as Polars DataFrame:")
            customers_pl = ora_to_polars(oracle_conn, "SELECT * FROM tfmv.customers")
            logging.info(f"   DataFrame shape: {customers_pl.shape}")

            # Show summary statistics
            logging.info("2. Summary statistics:")
            logging.info(customers_pl.describe())

            # Demonstrate some Polars operations
            logging.info("3. Polars DataFrame operations:")

            # Get column names
            col_names = customers_pl.columns
            logging.info(f"   Column names: {col_names}")

            # If we have numeric columns, show some aggregations
            numeric_cols = [
                col
                for col in customers_pl.columns
                if customers_pl[col].dtype in [pl.Int64, pl.Float64]
            ]

            if numeric_cols:
                logging.info(f"   Numeric columns: {numeric_cols}")
                sample_col = numeric_cols[0]
                logging.info(
                    f"   Sum of {sample_col}: {customers_pl[sample_col].sum()}"
                )
                logging.info(
                    f"   Mean of {sample_col}: {customers_pl[sample_col].mean()}"
                )

            # Create visualization
            viz_path = visualize_data(customers_pl, "Customer Data Analysis")
            logging.info(f"4. Visualization created: {viz_path}")

            # Demo Polars Series operations
            if len(col_names) > 0:
                logging.info("\n=== POLARS SERIES DEMO ===")
                sample_col = col_names[0]

                # Get a single column as a Series
                logging.info(f"1. Loading column '{sample_col}' as Polars Series:")
                series_query = f"SELECT {sample_col} FROM tfmv.customers"
                series = ora_to_polars_series(oracle_conn, series_query, sample_col)
                logging.info(f"   Series shape: {series.shape}")

                # Show series operations
                logging.info("2. Series operations:")
                logging.info(f"   First 5 values: {series.head(5)}")

                # If numeric, show more operations
                if series.dtype in [pl.Int64, pl.Float64]:
                    logging.info(f"   Sum: {series.sum()}")
                    logging.info(f"   Mean: {series.mean()}")
                    if (series > 0).all():
                        logging.info(f"   Log10: {series.log10().head(5)}")
                else:
                    # For string series
                    if series.dtype == pl.Utf8:
                        logging.info(
                            f"   Unique values: {series.unique().sort().head(5)}"
                        )
                        logging.info(
                            f"   Value counts: {series.value_counts().head(5)}"
                        )

        # Print performance report
        print_performance_report()

    except Exception as e:
        logging.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
