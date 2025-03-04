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
| Oracle connection   | 1.12           | 32.8%      |
| Oracle data read    | 2.17           | 63.5%      |
| Postgres connection | 0.08           | 2.4%       |
| Postgres data write | 0.05           | 1.3%       |
| **Total execution** | **3.41**       | **100%**   |

This demonstrates that most of the time is spent in Oracle operations (connection and data retrieval), while PostgreSQL operations are significantly faster. The Arrow-based data transfer to PostgreSQL is particularly efficient, taking only 1.3% of the total execution time.

### Network Impact

A significant portion of the Oracle operation time (96.3% of total execution) is due to network latency when connecting to Oracle Cloud Infrastructure (OCI). Local PostgreSQL operations are much faster by comparison. Potential optimizations include:

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
oracledb
pandas
pyarrow
adbc-driver-postgresql
python-dotenv
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

## Limitations

1. Oracle Arrow support is in preview
2. Limited error recovery mechanisms
3. No automatic schema mapping
4. Basic data type support only

## License

MIT

## Disclaimer

This project uses experimental features and is not recommended for production use. The Oracle Arrow integration is in preview status and may have limitations or unexpected behaviors.
