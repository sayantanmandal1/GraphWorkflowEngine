"""Configuration management for the Agent Workflow Engine."""

import os
from typing import Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator
from enum import Enum


class LogLevel(str, Enum):
    """Logging levels."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class DatabaseType(str, Enum):
    """Supported database types."""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


class AppConfig(BaseModel):
    """Application configuration settings."""
    
    # Application settings
    app_name: str = Field(default="Agent Workflow Engine", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Enable debug mode")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=False, description="Enable auto-reload in development")
    
    # Database settings
    database_url: str = Field(
        default="sqlite:///./workflow_engine.db",
        description="Database connection URL"
    )
    database_echo: bool = Field(default=False, description="Enable SQLAlchemy query logging")
    database_pool_size: int = Field(default=5, description="Database connection pool size")
    database_max_overflow: int = Field(default=10, description="Database connection pool overflow")
    
    # Execution engine settings
    max_concurrent_executions: int = Field(
        default=10,
        description="Maximum number of concurrent workflow executions"
    )
    execution_timeout: int = Field(
        default=3600,
        description="Default execution timeout in seconds"
    )
    node_timeout: int = Field(
        default=300,
        description="Default node execution timeout in seconds"
    )
    
    # WebSocket settings
    websocket_ping_interval: int = Field(
        default=20,
        description="WebSocket ping interval in seconds"
    )
    websocket_ping_timeout: int = Field(
        default=10,
        description="WebSocket ping timeout in seconds"
    )
    websocket_max_connections: int = Field(
        default=100,
        description="Maximum WebSocket connections"
    )
    
    # Logging settings
    log_level: LogLevel = Field(default=LogLevel.INFO, description="Logging level")
    log_format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log message format"
    )
    log_file: Optional[str] = Field(default=None, description="Log file path")
    log_max_size: int = Field(default=10485760, description="Maximum log file size in bytes")  # 10MB
    log_backup_count: int = Field(default=5, description="Number of log backup files to keep")
    
    # Health check settings
    health_check_timeout: float = Field(
        default=5.0,
        description="Health check timeout in seconds"
    )
    
    # Performance monitoring settings
    slow_request_threshold: float = Field(
        default=5.0,
        description="Slow request threshold in seconds"
    )
    enable_performance_monitoring: bool = Field(
        default=True,
        description="Enable performance monitoring middleware"
    )
    
    # Security settings
    cors_origins: list = Field(
        default=["*"],
        description="CORS allowed origins"
    )
    cors_methods: list = Field(
        default=["GET", "POST", "PUT", "DELETE"],
        description="CORS allowed methods"
    )
    
    # Storage settings
    enable_historical_data_cleanup: bool = Field(
        default=True,
        description="Enable automatic cleanup of old historical data"
    )
    historical_data_retention_days: int = Field(
        default=30,
        description="Number of days to retain historical data"
    )
    
    @field_validator('database_url')
    @classmethod
    def validate_database_url(cls, v):
        """Validate database URL format."""
        if not v:
            raise ValueError("Database URL cannot be empty")
        
        supported_schemes = ['sqlite', 'postgresql', 'mysql']
        scheme = v.split('://')[0].lower()
        
        if scheme not in supported_schemes:
            raise ValueError(f"Unsupported database scheme: {scheme}. Supported: {supported_schemes}")
        
        return v
    
    @field_validator('port')
    @classmethod
    def validate_port(cls, v):
        """Validate port number."""
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @field_validator('max_concurrent_executions')
    @classmethod
    def validate_max_concurrent_executions(cls, v):
        """Validate maximum concurrent executions."""
        if v < 1:
            raise ValueError("Maximum concurrent executions must be at least 1")
        return v
    
    @field_validator('execution_timeout', 'node_timeout')
    @classmethod
    def validate_timeouts(cls, v):
        """Validate timeout values."""
        if v < 1:
            raise ValueError("Timeout must be at least 1 second")
        return v
    
    @property
    def database_type(self) -> DatabaseType:
        """Get the database type from the URL."""
        scheme = self.database_url.split('://')[0].lower()
        if scheme == 'sqlite':
            return DatabaseType.SQLITE
        elif scheme.startswith('postgresql'):
            return DatabaseType.POSTGRESQL
        elif scheme.startswith('mysql'):
            return DatabaseType.MYSQL
        else:
            raise ValueError(f"Unknown database type: {scheme}")
    
    @property
    def is_sqlite(self) -> bool:
        """Check if using SQLite database."""
        return self.database_type == DatabaseType.SQLITE
    
    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return not self.debug and not self.reload
    
    def get_database_connect_args(self) -> Dict[str, Any]:
        """Get database connection arguments based on database type."""
        if self.is_sqlite:
            return {"check_same_thread": False}
        return {}
    
    def get_uvicorn_config(self) -> Dict[str, Any]:
        """Get Uvicorn server configuration."""
        return {
            "host": self.host,
            "port": self.port,
            "reload": self.reload,
            "log_level": self.log_level.lower(),
            "access_log": self.debug
        }
    
    @classmethod
    def from_env(cls) -> 'AppConfig':
        """Create configuration from environment variables."""
        def get_env(key: str, default=None, type_func=str):
            """Get environment variable with type conversion."""
            value = os.getenv(f"WORKFLOW_ENGINE_{key}")
            if value is None:
                return default
            if type_func == bool:
                return str(value).lower() in ('true', '1', 'yes', 'on')
            elif type_func == list:
                return value.split(',') if value else default
            return type_func(value)
        
        return cls(
            app_name=get_env("APP_NAME", "Agent Workflow Engine"),
            app_version=get_env("APP_VERSION", "1.0.0"),
            debug=get_env("DEBUG", False, bool),
            host=get_env("HOST", "0.0.0.0"),
            port=get_env("PORT", 8000, int),
            reload=get_env("RELOAD", False, bool),
            database_url=get_env("DATABASE_URL", "sqlite:///./workflow_engine.db"),
            database_echo=get_env("DATABASE_ECHO", False, bool),
            database_pool_size=get_env("DATABASE_POOL_SIZE", 5, int),
            database_max_overflow=get_env("DATABASE_MAX_OVERFLOW", 10, int),
            max_concurrent_executions=get_env("MAX_CONCURRENT_EXECUTIONS", 10, int),
            execution_timeout=get_env("EXECUTION_TIMEOUT", 3600, int),
            node_timeout=get_env("NODE_TIMEOUT", 300, int),
            websocket_ping_interval=get_env("WEBSOCKET_PING_INTERVAL", 20, int),
            websocket_ping_timeout=get_env("WEBSOCKET_PING_TIMEOUT", 10, int),
            websocket_max_connections=get_env("WEBSOCKET_MAX_CONNECTIONS", 100, int),
            log_level=LogLevel(get_env("LOG_LEVEL", "INFO")),
            log_format=get_env("LOG_FORMAT", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            log_file=get_env("LOG_FILE", None),
            log_max_size=get_env("LOG_MAX_SIZE", 10485760, int),
            log_backup_count=get_env("LOG_BACKUP_COUNT", 5, int),
            health_check_timeout=get_env("HEALTH_CHECK_TIMEOUT", 5.0, float),
            slow_request_threshold=get_env("SLOW_REQUEST_THRESHOLD", 5.0, float),
            enable_performance_monitoring=get_env("ENABLE_PERFORMANCE_MONITORING", True, bool),
            cors_origins=get_env("CORS_ORIGINS", ["*"], list),
            cors_methods=get_env("CORS_METHODS", ["GET", "POST", "PUT", "DELETE"], list),
            enable_historical_data_cleanup=get_env("ENABLE_HISTORICAL_DATA_CLEANUP", True, bool),
            historical_data_retention_days=get_env("HISTORICAL_DATA_RETENTION_DAYS", 30, int)
        )


# Global configuration instance
_config: Optional[AppConfig] = None


def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = AppConfig.from_env()
    return _config


def load_config(config_file: Optional[str] = None) -> AppConfig:
    """Load configuration from file or environment variables."""
    global _config
    
    # Load .env file if it exists
    if config_file and os.path.exists(config_file):
        from dotenv import load_dotenv
        load_dotenv(config_file)
    elif os.path.exists('.env'):
        from dotenv import load_dotenv
        load_dotenv('.env')
    
    # Create configuration from environment variables
    _config = AppConfig.from_env()
    
    return _config


def reset_config():
    """Reset the global configuration instance (mainly for testing)."""
    global _config
    _config = None


# Configuration validation
def validate_config(config: AppConfig) -> None:
    """Validate configuration settings."""
    errors = []
    
    # Check database connectivity requirements
    if config.is_sqlite:
        db_path = config.database_url.replace("sqlite:///", "")
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            try:
                os.makedirs(db_dir, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create database directory {db_dir}: {e}")
    
    # Validate log file directory
    if config.log_file:
        log_dir = os.path.dirname(config.log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create log directory {log_dir}: {e}")
    
    # Check resource limits
    if config.max_concurrent_executions > 100:
        errors.append("Warning: High concurrent execution limit may impact performance")
    
    if config.websocket_max_connections > 1000:
        errors.append("Warning: High WebSocket connection limit may impact performance")
    
    if errors:
        raise ValueError(f"Configuration validation failed: {'; '.join(errors)}")


# Environment-specific configurations
def get_development_config() -> AppConfig:
    """Get development configuration."""
    return AppConfig(
        debug=True,
        reload=True,
        log_level=LogLevel.DEBUG,
        database_echo=True,
        enable_performance_monitoring=True
    )


def get_production_config() -> AppConfig:
    """Get production configuration."""
    return AppConfig(
        debug=False,
        reload=False,
        log_level=LogLevel.INFO,
        database_echo=False,
        enable_performance_monitoring=True,
        cors_origins=[]  # Restrict CORS in production
    )


def get_testing_config() -> AppConfig:
    """Get testing configuration."""
    return AppConfig(
        debug=True,
        database_url="sqlite:///:memory:",
        log_level=LogLevel.WARNING,
        max_concurrent_executions=2,
        execution_timeout=30,
        node_timeout=10
    )