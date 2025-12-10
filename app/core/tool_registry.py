"""Tool Registry component for managing reusable workflow tools."""

import importlib
import inspect
from typing import Callable, Dict, Optional, Any
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError

from ..storage.database import get_db
from ..storage.models import ToolRegistryModel
from .exceptions import ToolRegistryError
from .logging import get_logger

logger = get_logger(__name__)


class ToolRegistry:
    """Registry for managing reusable tools that can be called by workflow nodes."""
    
    def __init__(self, db_session: Optional[Session] = None):
        """Initialize the tool registry.
        
        Args:
            db_session: Optional database session. If not provided, will create new sessions as needed.
        """
        self._db_session = db_session
        self._memory_cache: Dict[str, Callable] = {}
    
    def _get_session(self) -> Session:
        """Get database session, creating a new one if needed."""
        if self._db_session:
            return self._db_session
        return next(get_db())
    
    def register_tool(self, name: str, function: Callable, description: str = "") -> None:
        """Register a Python function as a reusable tool.
        
        Args:
            name: Unique identifier for the tool
            function: Python function to register
            description: Optional description of the tool's purpose
            
        Raises:
            ToolRegistryError: If tool name already exists or function is invalid
        """
        if not name or not name.strip():
            raise ToolRegistryError("Tool name cannot be empty")
        
        name = name.strip()
        
        if not callable(function):
            raise ToolRegistryError(f"Tool '{name}' must be a callable function")
        
        # Validate function has proper signature
        try:
            sig = inspect.signature(function)
            # Function should accept at least a state parameter
            if len(sig.parameters) == 0:
                logger.warning(f"Tool '{name}' has no parameters - it won't be able to access workflow state")
        except (ValueError, TypeError) as e:
            raise ToolRegistryError(f"Cannot inspect function signature for tool '{name}': {e}")
        
        # Get function module and name for storage
        function_module = function.__module__
        function_name = function.__name__
        
        if not function_module or not function_name:
            raise ToolRegistryError(f"Cannot determine module or name for function '{name}'")
        
        session = self._get_session()
        try:
            # Check if tool already exists
            existing_tool = session.query(ToolRegistryModel).filter_by(name=name).first()
            if existing_tool:
                raise ToolRegistryError(f"Tool '{name}' is already registered")
            
            # Create new tool entry
            tool_model = ToolRegistryModel(
                name=name,
                description=description.strip() if description else "",
                function_module=function_module,
                function_name=function_name
            )
            
            session.add(tool_model)
            session.commit()
            
            # Cache the function in memory for faster access
            self._memory_cache[name] = function
            
            logger.info(f"Successfully registered tool '{name}' from {function_module}.{function_name}")
            
        except IntegrityError:
            session.rollback()
            raise ToolRegistryError(f"Tool '{name}' is already registered")
        except Exception as e:
            session.rollback()
            raise ToolRegistryError(f"Failed to register tool '{name}': {e}")
        finally:
            if not self._db_session:
                session.close()
    
    def get_tool(self, name: str) -> Callable:
        """Retrieve a registered tool by name.
        
        Args:
            name: Name of the tool to retrieve
            
        Returns:
            The callable function associated with the tool name
            
        Raises:
            ToolRegistryError: If tool is not found or cannot be loaded
        """
        if not name or not name.strip():
            raise ToolRegistryError("Tool name cannot be empty")
        
        name = name.strip()
        
        # Check memory cache first
        if name in self._memory_cache:
            return self._memory_cache[name]
        
        session = self._get_session()
        try:
            # Query database for tool
            tool_model = session.query(ToolRegistryModel).filter_by(name=name).first()
            if not tool_model:
                raise ToolRegistryError(f"Tool '{name}' is not registered")
            
            # Load the function from its module
            try:
                module = importlib.import_module(tool_model.function_module)
                function = getattr(module, tool_model.function_name)
                
                if not callable(function):
                    raise ToolRegistryError(f"Tool '{name}' is not callable")
                
                # Cache for future use
                self._memory_cache[name] = function
                
                logger.debug(f"Loaded tool '{name}' from {tool_model.function_module}.{tool_model.function_name}")
                return function
                
            except ImportError as e:
                raise ToolRegistryError(f"Cannot import module for tool '{name}': {e}")
            except AttributeError as e:
                raise ToolRegistryError(f"Function not found in module for tool '{name}': {e}")
            
        except Exception as e:
            if isinstance(e, ToolRegistryError):
                raise
            raise ToolRegistryError(f"Failed to retrieve tool '{name}': {e}")
        finally:
            if not self._db_session:
                session.close()
    
    def list_tools(self) -> Dict[str, str]:
        """List all registered tools with their descriptions.
        
        Returns:
            Dictionary mapping tool names to their descriptions
        """
        session = self._get_session()
        try:
            tools = session.query(ToolRegistryModel).all()
            return {tool.name: tool.description for tool in tools}
        except Exception as e:
            raise ToolRegistryError(f"Failed to list tools: {e}")
        finally:
            if not self._db_session:
                session.close()
    
    def unregister_tool(self, name: str) -> bool:
        """Remove a tool from the registry.
        
        Args:
            name: Name of the tool to remove
            
        Returns:
            True if tool was removed, False if tool was not found
            
        Raises:
            ToolRegistryError: If removal operation fails
        """
        if not name or not name.strip():
            raise ToolRegistryError("Tool name cannot be empty")
        
        name = name.strip()
        
        session = self._get_session()
        try:
            tool_model = session.query(ToolRegistryModel).filter_by(name=name).first()
            if not tool_model:
                return False
            
            session.delete(tool_model)
            session.commit()
            
            # Remove from memory cache
            self._memory_cache.pop(name, None)
            
            logger.info(f"Successfully unregistered tool '{name}'")
            return True
            
        except Exception as e:
            session.rollback()
            raise ToolRegistryError(f"Failed to unregister tool '{name}': {e}")
        finally:
            if not self._db_session:
                session.close()
    
    def call_tool(self, name: str, *args, **kwargs) -> Any:
        """Call a registered tool with the provided arguments.
        
        Args:
            name: Name of the tool to call
            *args: Positional arguments to pass to the tool
            **kwargs: Keyword arguments to pass to the tool
            
        Returns:
            The result of calling the tool function
            
        Raises:
            ToolRegistryError: If tool is not found or execution fails
        """
        try:
            tool_function = self.get_tool(name)
            return tool_function(*args, **kwargs)
        except Exception as e:
            if isinstance(e, ToolRegistryError):
                raise
            raise ToolRegistryError(f"Failed to execute tool '{name}': {e}")
    
    def tool_exists(self, name: str) -> bool:
        """Check if a tool is registered.
        
        Args:
            name: Name of the tool to check
            
        Returns:
            True if tool exists, False otherwise
        """
        if not name or not name.strip():
            return False
        
        name = name.strip()
        
        # Check memory cache first
        if name in self._memory_cache:
            return True
        
        session = self._get_session()
        try:
            tool_model = session.query(ToolRegistryModel).filter_by(name=name).first()
            return tool_model is not None
        except Exception:
            return False
        finally:
            if not self._db_session:
                session.close()
    
    def clear_cache(self) -> None:
        """Clear the in-memory function cache."""
        self._memory_cache.clear()
        logger.debug("Tool registry memory cache cleared")
    
    def get_tool_info(self, name: str) -> Dict[str, str]:
        """Get detailed information about a registered tool.
        
        Args:
            name: Name of the tool
            
        Returns:
            Dictionary with tool information including name, description, module, and function name
            
        Raises:
            ToolRegistryError: If tool is not found
        """
        if not name or not name.strip():
            raise ToolRegistryError("Tool name cannot be empty")
        
        name = name.strip()
        
        session = self._get_session()
        try:
            tool_model = session.query(ToolRegistryModel).filter_by(name=name).first()
            if not tool_model:
                raise ToolRegistryError(f"Tool '{name}' is not registered")
            
            return {
                "name": tool_model.name,
                "description": tool_model.description,
                "module": tool_model.function_module,
                "function": tool_model.function_name,
                "created_at": tool_model.created_at.isoformat() if tool_model.created_at else None,
                "updated_at": tool_model.updated_at.isoformat() if tool_model.updated_at else None,
            }
        except Exception as e:
            if isinstance(e, ToolRegistryError):
                raise
            raise ToolRegistryError(f"Failed to get tool info for '{name}': {e}")
        finally:
            if not self._db_session:
                session.close()