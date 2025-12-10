"""Application factory for creating FastAPI instances."""

from datetime import datetime
from typing import Optional
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .config import AppConfig, get_config, validate_config
from .core.logging import setup_logging, get_logger
from .storage.database import create_tables, get_database_engine
from .core.graph_manager import GraphManager
from .core.execution_engine import ExecutionEngine
from .core.tool_registry import ToolRegistry
from .core.state_manager import StateManager
from .core.websocket_manager import WebSocketManager
from .api.endpoints import router, init_dependencies


class ApplicationState:
    """Container for application state and components."""
    
    def __init__(self):
        self.config: Optional[AppConfig] = None
        self.tool_registry: Optional[ToolRegistry] = None
        self.state_manager: Optional[StateManager] = None
        self.graph_manager: Optional[GraphManager] = None
        self.websocket_manager: Optional[WebSocketManager] = None
        self.execution_engine: Optional[ExecutionEngine] = None
        self.logger = None


# Global application state
app_state = ApplicationState()


def register_default_tools(tool_registry: ToolRegistry, logger) -> None:
    """Register default tools for the workflow engine."""
    try:
        # Register test tools
        from .tools.test_tools import (
            workflow_test_function,
            simple_math_function,
            conditional_function,
            state_logger_function
        )
        
        # Register code review tools
        from .tools.code_review_tools import (
            extract_functions,
            analyze_function_complexity,
            evaluate_quality_threshold,
            generate_improvement_suggestions,
            generate_final_report
        )
        
        # Define tools to register
        tools_to_register = [
            ("workflow_test_function", workflow_test_function, "A simple test function for workflow testing"),
            ("simple_math_function", simple_math_function, "Performs basic math operations on workflow state"),
            ("conditional_function", conditional_function, "Demonstrates conditional logic based on state values"),
            ("state_logger_function", state_logger_function, "Logs current workflow state for debugging"),
            ("extract_functions", extract_functions, "Extract individual functions from Python code for analysis"),
            ("analyze_function_complexity", analyze_function_complexity, "Calculate complexity metrics and identify code issues"),
            ("evaluate_quality_threshold", evaluate_quality_threshold, "Evaluate quality scores against threshold"),
            ("generate_improvement_suggestions", generate_improvement_suggestions, "Generate improvement suggestions for low-quality code"),
            ("generate_final_report", generate_final_report, "Generate comprehensive code review report")
        ]
        
        # Register each tool
        for tool_name, tool_func, tool_desc in tools_to_register:
            try:
                if not tool_registry.tool_exists(tool_name):
                    tool_registry.register_tool(tool_name, tool_func, tool_desc)
                    logger.info(f"Registered tool: {tool_name}")
                else:
                    logger.info(f"Tool already exists: {tool_name}")
            except Exception as e:
                logger.warning(f"Failed to register tool {tool_name}: {e}")
        
        logger.info("Default tools registration completed")
        
    except Exception as e:
        logger.error(f"Failed to register default tools: {e}")
        raise


def setup_health_checks(execution_engine: ExecutionEngine, tool_registry: ToolRegistry, 
                       websocket_manager: WebSocketManager, logger) -> None:
    """Set up health check functions."""
    try:
        from .core.error_recovery import health_checker
        
        # Database health check
        def check_database():
            try:
                from .storage.database import get_db
                db = next(get_db())
                db.execute("SELECT 1")
                return {"status": "healthy", "message": "Database connection successful"}
            except Exception as e:
                raise Exception(f"Database connection failed: {str(e)}")
        
        # Execution engine health check
        def check_execution_engine():
            try:
                stats = execution_engine.get_execution_statistics()
                active_count = len(execution_engine._active_executions)
                return {
                    "status": "healthy",
                    "message": "Execution engine operational",
                    "active_executions": active_count,
                    "max_concurrent": execution_engine._max_concurrent_executions
                }
            except Exception as e:
                raise Exception(f"Execution engine check failed: {str(e)}")
        
        # Tool registry health check
        def check_tool_registry():
            try:
                tools = tool_registry.list_tools()
                return {
                    "status": "healthy",
                    "message": "Tool registry operational",
                    "registered_tools": len(tools)
                }
            except Exception as e:
                raise Exception(f"Tool registry check failed: {str(e)}")
        
        # WebSocket manager health check
        def check_websocket_manager():
            try:
                if websocket_manager:
                    info = websocket_manager.get_connection_info()
                    return {
                        "status": "healthy",
                        "message": "WebSocket manager operational",
                        **info
                    }
                else:
                    return {
                        "status": "healthy",
                        "message": "WebSocket manager not configured"
                    }
            except Exception as e:
                raise Exception(f"WebSocket manager check failed: {str(e)}")
        
        # Register all health checks
        health_checker.register_check("database", check_database, timeout=5.0)
        health_checker.register_check("execution_engine", check_execution_engine, timeout=3.0)
        health_checker.register_check("tool_registry", check_tool_registry, timeout=2.0)
        health_checker.register_check("websocket_manager", check_websocket_manager, timeout=2.0)
        
        logger.info("Health checks registered")
        
    except Exception as e:
        logger.error(f"Failed to setup health checks: {e}")
        raise


def initialize_database(config: AppConfig, logger) -> None:
    """Initialize database and run migrations."""
    try:
        # Create database tables
        create_tables()
        logger.info("Database tables created")
        
        # Run historical data migrations for performance optimization
        try:
            from .storage.migrations import run_historical_data_migrations
            run_historical_data_migrations()
            logger.info("Historical data migrations completed")
        except Exception as e:
            logger.warning(f"Historical data migrations failed: {str(e)}")
            # Continue startup even if migrations fail
            
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


def initialize_core_components(config: AppConfig, logger) -> tuple:
    """Initialize core application components."""
    try:
        # Initialize core components
        tool_registry = ToolRegistry()
        state_manager = StateManager()
        graph_manager = GraphManager()
        websocket_manager = WebSocketManager()
        execution_engine = ExecutionEngine(
            tool_registry=tool_registry,
            state_manager=state_manager,
            graph_manager=graph_manager,
            max_concurrent_executions=config.max_concurrent_executions,
            websocket_manager=websocket_manager
        )
        
        logger.info("Core components initialized")
        
        return tool_registry, state_manager, graph_manager, websocket_manager, execution_engine
        
    except Exception as e:
        logger.error(f"Core components initialization failed: {e}")
        raise


def graceful_shutdown(execution_engine: ExecutionEngine, websocket_manager: WebSocketManager, logger) -> None:
    """Handle graceful shutdown of application components."""
    logger.info("Shutting down Agent Workflow Engine")
    
    # Stop WebSocket broadcast processor
    try:
        websocket_manager.stop_broadcast_processor()
        logger.info("WebSocket broadcast processor stopped")
    except Exception as e:
        logger.error(f"Error stopping WebSocket broadcast processor: {str(e)}")
    
    # Shutdown execution engine
    try:
        execution_engine.shutdown()
        logger.info("Execution engine shutdown completed")
    except Exception as e:
        logger.error(f"Error during execution engine shutdown: {str(e)}")


@asynccontextmanager
async def create_lifespan_handler(config: AppConfig):
    """Create application lifespan handler."""
    
    # Startup
    logger = setup_logging(
        level=config.log_level.value,
        log_file=config.log_file,
        log_format=config.log_format,
        max_size=config.log_max_size,
        backup_count=config.log_backup_count
    )
    
    logger.info(f"Starting {config.app_name} v{config.app_version}")
    
    try:
        # Initialize database
        initialize_database(config, logger)
        
        # Initialize core components
        components = initialize_core_components(config, logger)
        tool_registry, state_manager, graph_manager, websocket_manager, execution_engine = components
        
        # Store components in global state
        app_state.config = config
        app_state.tool_registry = tool_registry
        app_state.state_manager = state_manager
        app_state.graph_manager = graph_manager
        app_state.websocket_manager = websocket_manager
        app_state.execution_engine = execution_engine
        app_state.logger = logger
        
        # Register default tools
        register_default_tools(tool_registry, logger)
        
        # Initialize API dependencies
        init_dependencies(
            graph_manager=graph_manager,
            execution_engine=execution_engine,
            tool_registry=tool_registry,
            state_manager=state_manager,
            websocket_manager=websocket_manager
        )
        
        # Setup health checks
        setup_health_checks(execution_engine, tool_registry, websocket_manager, logger)
        
        # Start WebSocket broadcast processor
        websocket_manager.start_broadcast_processor()
        logger.info("WebSocket broadcast processor started")
        
        logger.info("Application startup completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Application startup failed: {e}")
        raise
    
    # Shutdown
    try:
        graceful_shutdown(execution_engine, websocket_manager, logger)
    except Exception as e:
        logger.error(f"Error during graceful shutdown: {e}")


def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """Create and configure FastAPI application instance."""
    
    # Use provided config or load from environment
    if config is None:
        config = get_config()
    
    # Validate configuration
    validate_config(config)
    
    # Create lifespan handler
    lifespan_handler = create_lifespan_handler(config)
    
    # Create FastAPI application
    app = FastAPI(
        title=config.app_name,
        description="A lightweight workflow engine for defining, executing, and monitoring computational workflows",
        version=config.app_version,
        debug=config.debug,
        lifespan=lifespan_handler
    )
    
    # Add CORS middleware
    if config.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.cors_origins,
            allow_credentials=True,
            allow_methods=config.cors_methods,
            allow_headers=["*"],
        )
    
    # Add custom middleware
    if config.enable_performance_monitoring:
        from .core.middleware import (
            ErrorHandlingMiddleware, 
            RequestLoggingMiddleware, 
            PerformanceMonitoringMiddleware
        )
        
        app.add_middleware(ErrorHandlingMiddleware)
        app.add_middleware(PerformanceMonitoringMiddleware, slow_request_threshold=config.slow_request_threshold)
        app.add_middleware(RequestLoggingMiddleware)
    
    # Include API router
    app.include_router(router)
    
    # Add health check endpoints
    add_health_endpoints(app, config)
    
    return app


def add_health_endpoints(app: FastAPI, config: AppConfig) -> None:
    """Add health check endpoints to the application."""
    
    @app.get("/")
    async def root():
        """Root endpoint for basic health check."""
        return {"message": f"{config.app_name} is running", "version": config.app_version}
    
    @app.get("/health")
    async def health_check():
        """Basic health check endpoint."""
        return {
            "status": "healthy", 
            "service": config.app_name.lower().replace(" ", "-"),
            "version": config.app_version
        }
    
    @app.get("/health/detailed")
    async def detailed_health_check():
        """Detailed health check endpoint with component status."""
        from .core.error_recovery import health_checker
        
        try:
            results = await health_checker.run_all_checks()
            status_code = 200 if results["overall_status"] == "healthy" else 503
            
            return JSONResponse(
                status_code=status_code,
                content={
                    "service": config.app_name.lower().replace(" ", "-"),
                    "version": config.app_version,
                    **results
                }
            )
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Health check failed: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={
                    "service": config.app_name.lower().replace(" ", "-"),
                    "overall_status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    @app.get("/health/ready")
    async def readiness_check():
        """Readiness check endpoint for container orchestration."""
        from .core.error_recovery import health_checker
        
        try:
            # Check critical components only
            critical_checks = ["database", "execution_engine"]
            results = {}
            
            for check_name in critical_checks:
                if check_name in health_checker.checks:
                    results[check_name] = await health_checker.run_check(check_name)
            
            # Determine if service is ready
            ready = all(
                result.get("status") == "healthy" 
                for result in results.values()
            )
            
            status_code = 200 if ready else 503
            
            return JSONResponse(
                status_code=status_code,
                content={
                    "ready": ready,
                    "checks": results,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Readiness check failed: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={
                    "ready": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    @app.get("/health/live")
    async def liveness_check():
        """Liveness check endpoint for container orchestration."""
        try:
            # Simple check that the application is responsive
            return {
                "alive": True,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            logger = get_logger(__name__)
            logger.error(f"Liveness check failed: {str(e)}")
            return JSONResponse(
                status_code=503,
                content={
                    "alive": False,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
            )


def get_app_state() -> ApplicationState:
    """Get the current application state."""
    return app_state