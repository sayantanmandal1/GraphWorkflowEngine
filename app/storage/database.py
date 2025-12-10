"""Database connection and session management."""

import os
from typing import Optional
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

# Global engine instance
_engine: Optional[Engine] = None

# Base class for all database models
Base = declarative_base()


def get_database_engine(database_url: Optional[str] = None, 
                       echo: bool = False,
                       connect_args: Optional[dict] = None) -> Engine:
    """Get or create database engine with configuration."""
    global _engine
    
    if _engine is None:
        # Use provided URL or fall back to environment variable
        if database_url is None:
            database_url = os.getenv("DATABASE_URL", "sqlite:///./workflow_engine.db")
        
        # Default connect args for SQLite
        if connect_args is None:
            if database_url.startswith("sqlite"):
                connect_args = {"check_same_thread": False}
            else:
                connect_args = {}
        
        # Create engine with appropriate settings
        if database_url.startswith("sqlite"):
            _engine = create_engine(
                database_url,
                connect_args=connect_args,
                poolclass=StaticPool,
                echo=echo
            )
        else:
            _engine = create_engine(
                database_url, 
                echo=echo,
                connect_args=connect_args
            )
    
    return _engine


def reset_database_engine():
    """Reset the global database engine (mainly for testing)."""
    global _engine
    if _engine:
        _engine.dispose()
    _engine = None


# Backward compatibility - create default engine
engine = get_database_engine()

# Create session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for all database models
Base = declarative_base()


def get_db():
    """Dependency to get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all database tables."""
    Base.metadata.create_all(bind=engine)


def drop_tables():
    """Drop all database tables."""
    Base.metadata.drop_all(bind=engine)