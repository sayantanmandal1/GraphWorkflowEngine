"""Application startup script and CLI interface."""

import sys
import os
import argparse
from typing import Optional

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import (
    AppConfig, 
    get_config, 
    load_config, 
    get_development_config,
    get_production_config,
    get_testing_config,
    validate_config
)
from app.factory import create_app
from app.core.logging import get_logger


def create_argument_parser() -> argparse.ArgumentParser:
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Agent Workflow Engine - A lightweight workflow orchestration system"
    )
    
    # Server configuration
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind the server to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind the server to (default: 8000)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development"
    )
    
    # Environment configuration
    parser.add_argument(
        "--env",
        choices=["development", "production", "testing"],
        help="Environment configuration preset"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file"
    )
    
    # Database configuration
    parser.add_argument(
        "--database-url",
        help="Database connection URL"
    )
    
    # Logging configuration
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level"
    )
    
    parser.add_argument(
        "--log-file",
        help="Path to log file"
    )
    
    # Debug mode
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    # Execution engine configuration
    parser.add_argument(
        "--max-concurrent-executions",
        type=int,
        help="Maximum number of concurrent workflow executions"
    )
    
    # Commands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run server command
    run_parser = subparsers.add_parser("run", help="Run the workflow engine server")
    run_parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    # Database commands
    db_parser = subparsers.add_parser("db", help="Database management commands")
    db_subparsers = db_parser.add_subparsers(dest="db_command", help="Database commands")
    
    db_subparsers.add_parser("init", help="Initialize database tables")
    db_subparsers.add_parser("migrate", help="Run database migrations")
    db_subparsers.add_parser("reset", help="Reset database (drop and recreate tables)")
    
    # Health check command
    health_parser = subparsers.add_parser("health", help="Run health checks")
    health_parser.add_argument(
        "--detailed",
        action="store_true",
        help="Run detailed health checks"
    )
    
    # Configuration commands
    config_parser = subparsers.add_parser("config", help="Configuration management")
    config_subparsers = config_parser.add_subparsers(dest="config_command", help="Config commands")
    
    config_subparsers.add_parser("show", help="Show current configuration")
    config_subparsers.add_parser("validate", help="Validate configuration")
    
    return parser


def load_configuration(args: argparse.Namespace) -> AppConfig:
    """Load configuration based on command line arguments."""
    
    # Load environment-specific configuration
    if args.env == "development":
        config = get_development_config()
    elif args.env == "production":
        config = get_production_config()
    elif args.env == "testing":
        config = get_testing_config()
    else:
        # Load from config file or environment
        config = load_config(args.config)
    
    # Override with command line arguments
    if args.host:
        config.host = args.host
    if args.port:
        config.port = args.port
    if args.reload:
        config.reload = args.reload
    if args.database_url:
        config.database_url = args.database_url
    if args.log_level:
        config.log_level = args.log_level
    if args.log_file:
        config.log_file = args.log_file
    if args.debug:
        config.debug = args.debug
    if args.max_concurrent_executions:
        config.max_concurrent_executions = args.max_concurrent_executions
    
    return config


def run_server(config: AppConfig, workers: int = 1):
    """Run the workflow engine server."""
    import uvicorn
    
    logger = get_logger(__name__)
    logger.info(f"Starting server with {workers} worker(s)")
    
    # Get uvicorn configuration
    uvicorn_config = config.get_uvicorn_config()
    
    if workers > 1:
        # Multi-worker mode
        uvicorn.run(
            "app.factory:create_app",
            factory=True,
            workers=workers,
            **uvicorn_config
        )
    else:
        # Single worker mode
        app = create_app(config)
        uvicorn.run(app, **uvicorn_config)


def run_database_command(command: str, config: AppConfig):
    """Run database management commands."""
    from app.storage.database import create_tables, drop_tables, get_database_engine
    from app.storage.migrations import run_historical_data_migrations
    
    logger = get_logger(__name__)
    
    if command == "init":
        logger.info("Initializing database tables...")
        create_tables()
        logger.info("Database tables created successfully")
        
    elif command == "migrate":
        logger.info("Running database migrations...")
        run_historical_data_migrations()
        logger.info("Database migrations completed successfully")
        
    elif command == "reset":
        logger.info("Resetting database...")
        drop_tables()
        create_tables()
        run_historical_data_migrations()
        logger.info("Database reset completed successfully")


async def run_health_check(config: AppConfig, detailed: bool = False):
    """Run health checks."""
    from app.core.error_recovery import health_checker
    
    logger = get_logger(__name__)
    
    if detailed:
        logger.info("Running detailed health checks...")
        results = await health_checker.run_all_checks()
        
        print(f"Overall Status: {results['overall_status']}")
        print(f"Timestamp: {results['timestamp']}")
        
        for check_name, result in results.get('checks', {}).items():
            status = result.get('status', 'unknown')
            message = result.get('message', 'No message')
            print(f"  {check_name}: {status} - {message}")
            
        if results['overall_status'] != 'healthy':
            sys.exit(1)
    else:
        logger.info("Running basic health check...")
        print("Service: Agent Workflow Engine")
        print("Status: Running")
        print(f"Version: {config.app_version}")


def show_configuration(config: AppConfig):
    """Show current configuration."""
    print("Current Configuration:")
    print(f"  App Name: {config.app_name}")
    print(f"  Version: {config.app_version}")
    print(f"  Debug: {config.debug}")
    print(f"  Host: {config.host}")
    print(f"  Port: {config.port}")
    print(f"  Database URL: {config.database_url}")
    print(f"  Log Level: {config.log_level}")
    print(f"  Max Concurrent Executions: {config.max_concurrent_executions}")
    print(f"  WebSocket Max Connections: {config.websocket_max_connections}")


def validate_configuration_command(config: AppConfig):
    """Validate configuration and show results."""
    try:
        validate_config(config)
        print("Configuration validation: PASSED")
        print("All configuration settings are valid.")
    except ValueError as e:
        print("Configuration validation: FAILED")
        print(f"Error: {e}")
        sys.exit(1)


def main():
    """Main entry point for the application."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    try:
        # Load configuration
        config = load_configuration(args)
        
        # Validate configuration
        validate_config(config)
        
        # Handle commands
        if args.command == "run" or args.command is None:
            # Default to running the server
            workers = getattr(args, 'workers', 1)
            run_server(config, workers)
            
        elif args.command == "db":
            if args.db_command:
                run_database_command(args.db_command, config)
            else:
                print("Database command required. Use --help for options.")
                sys.exit(1)
                
        elif args.command == "health":
            import asyncio
            asyncio.run(run_health_check(config, args.detailed))
            
        elif args.command == "config":
            if args.config_command == "show":
                show_configuration(config)
            elif args.config_command == "validate":
                validate_configuration_command(config)
            else:
                print("Configuration command required. Use --help for options.")
                sys.exit(1)
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()