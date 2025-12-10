#!/usr/bin/env python3
"""Database initialization script."""

import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import get_config
from app.storage.database import create_tables, get_database_engine
from app.storage.migrations import run_historical_data_migrations
from app.core.logging import setup_logging, get_logger


def main():
    """Initialize the database."""
    # Load configuration
    config = get_config()
    
    # Setup logging
    logger = setup_logging(level=config.log_level.value)
    
    try:
        logger.info("Initializing database...")
        
        # Initialize database engine with configuration
        engine = get_database_engine(
            database_url=config.database_url,
            echo=config.database_echo
        )
        
        # Create tables
        create_tables()
        logger.info("Database tables created successfully")
        
        # Run migrations
        run_historical_data_migrations()
        logger.info("Database migrations completed successfully")
        
        logger.info("Database initialization completed")
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()