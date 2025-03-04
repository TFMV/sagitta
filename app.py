import logging
import os
import time
from dataclasses import dataclass
from typing import Any, List, Tuple
from contextlib import contextmanager

import oracledb
import pandas as pd
import pyarrow as pa
import adbc_driver_postgresql.dbapi
from dotenv import load_dotenv


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
                performance_metrics[metric_name] = elapsed_time

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

        # Connect using the wallet configuration
        conn = oracledb.connect(
            user=config.username,
            password=config.password,
            dsn=config.dsn,
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
    total_time = sum(performance_metrics.values())
    performance_metrics["total"] = total_time

    logging.info("\n" + "=" * 50)
    logging.info("PERFORMANCE REPORT")
    logging.info("=" * 50)
    logging.info(
        f"Oracle connection:  {performance_metrics['oracle_connect']:.2f}s ({performance_metrics['oracle_connect']/total_time*100:.1f}%)"
    )
    logging.info(
        f"Oracle data read:   {performance_metrics['oracle_read']:.2f}s ({performance_metrics['oracle_read']/total_time*100:.1f}%)"
    )
    logging.info(
        f"Postgres connection:{performance_metrics['postgres_connect']:.2f}s ({performance_metrics['postgres_connect']/total_time*100:.1f}%)"
    )
    logging.info(
        f"Postgres data write:{performance_metrics['postgres_write']:.2f}s ({performance_metrics['postgres_write']/total_time*100:.1f}%)"
    )
    logging.info("-" * 50)
    logging.info(f"Total execution:    {total_time:.2f}s (100%)")
    logging.info("=" * 50)


@time_it(metric_name="total")
def main():
    """Main execution function."""
    setup_logging()
    oracle_cfg = OracleConfig.from_env()
    postgres_cfg = PostgresConfig.from_env()

    try:
        # Extract data from Oracle
        with oracle_connection(oracle_cfg) as oracle_conn:
            results, columns = fetch_oracle_data(
                oracle_conn, "SELECT * FROM tfmv.customer"
            )
            # Create DataFrame with lowercase column names
            customers_df = pd.DataFrame(results, columns=columns)
            logging.info(f"Retrieved {len(customers_df)} rows from Oracle")

        # Load data into PostgreSQL
        with postgres_connection(postgres_cfg) as pg_conn:
            customers_table = pa.Table.from_pandas(customers_df)
            rows_affected = pg_ingest_data(pg_conn, "c_customer", customers_table)
            logging.info(f"Inserted {rows_affected} rows into PostgreSQL")

        # Print performance report
        print_performance_report()

    except Exception as e:
        logging.error(f"Application error: {e}")
        raise


if __name__ == "__main__":
    main()
