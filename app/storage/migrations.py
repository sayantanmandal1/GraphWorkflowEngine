"""Database migrations for performance optimization."""

from sqlalchemy import text
from .database import engine, get_db
from .models import Base
from ..core.logging import get_logger

logger = get_logger(__name__)


def create_indexes_for_historical_queries():
    """Create database indexes to optimize historical data queries."""
    try:
        with engine.connect() as connection:
            # Index for workflow runs by status and completion date (for archiving/cleanup)
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_workflow_runs_status_completed 
                ON workflow_runs(status, completed_at)
            """))
            
            # Index for workflow runs by graph_id and started_at (for filtering)
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_workflow_runs_graph_started 
                ON workflow_runs(graph_id, started_at)
            """))
            
            # Index for workflow runs by started_at (for date range queries)
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_workflow_runs_started_at 
                ON workflow_runs(started_at DESC)
            """))
            
            # Index for log entries by run_id and timestamp (for log retrieval)
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_log_entries_run_timestamp 
                ON log_entries(run_id, timestamp)
            """))
            
            # Index for log entries by event_type (for archiving specific log types)
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_log_entries_event_type 
                ON log_entries(event_type)
            """))
            
            # Composite index for complex historical queries
            connection.execute(text("""
                CREATE INDEX IF NOT EXISTS idx_workflow_runs_composite 
                ON workflow_runs(status, graph_id, started_at DESC)
            """))
            
            connection.commit()
            logger.info("Successfully created database indexes for historical queries")
            
    except Exception as e:
        logger.error(f"Failed to create database indexes: {str(e)}")
        raise


def optimize_database_for_historical_queries():
    """Optimize database settings for historical data queries."""
    try:
        with engine.connect() as connection:
            # SQLite-specific optimizations
            if "sqlite" in str(engine.url):
                # Enable WAL mode for better concurrent read performance
                connection.execute(text("PRAGMA journal_mode=WAL"))
                
                # Increase cache size for better query performance
                connection.execute(text("PRAGMA cache_size=10000"))
                
                # Enable query optimization
                connection.execute(text("PRAGMA optimize"))
                
                logger.info("Applied SQLite optimizations for historical queries")
            
            connection.commit()
            
    except Exception as e:
        logger.error(f"Failed to optimize database: {str(e)}")
        raise


def run_historical_data_migrations():
    """Run all migrations for historical data functionality."""
    try:
        logger.info("Starting historical data migrations")
        
        # Create indexes
        create_indexes_for_historical_queries()
        
        # Apply database optimizations
        optimize_database_for_historical_queries()
        
        logger.info("Historical data migrations completed successfully")
        
    except Exception as e:
        logger.error(f"Historical data migrations failed: {str(e)}")
        raise


if __name__ == "__main__":
    # Run migrations when script is executed directly
    run_historical_data_migrations()