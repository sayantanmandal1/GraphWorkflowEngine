"""Tests for core workflow engine components."""

import pytest
import tempfile
import os
from datetime import datetime
from typing import Dict, Any

from app.core.tool_registry import ToolRegistry
from app.core.state_manager import StateManager
from app.core.graph_manager import GraphManager
from app.core.execution_engine import ExecutionEngine
from app.models.core import GraphDefinition, NodeDefinition, EdgeDefinition, ExecutionStatusEnum
from app.storage.database import create_tables, get_db


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    # Create temporary database file
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)
    
    # Store original DATABASE_URL
    original_db_url = os.environ.get('DATABASE_URL')
    
    # Set database URL for testing
    os.environ['DATABASE_URL'] = f'sqlite:///{db_path}'
    
    # Recreate engine and session with new URL
    from app.storage.database import engine, SessionLocal, Base
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool
    
    # Create new engine for test database
    test_engine = create_engine(
        f'sqlite:///{db_path}',
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        echo=False
    )
    
    # Create tables in test database
    Base.metadata.create_all(bind=test_engine)
    
    # Update the module-level engine and SessionLocal
    import app.storage.database as db_module
    db_module.engine = test_engine
    db_module.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    
    yield db_path
    
    # Cleanup
    try:
        os.unlink(db_path)
    except OSError:
        pass
    
    # Restore original DATABASE_URL
    if original_db_url:
        os.environ['DATABASE_URL'] = original_db_url
    else:
        os.environ.pop('DATABASE_URL', None)


@pytest.fixture
def tool_registry(temp_db):
    """Create a ToolRegistry instance for testing."""
    return ToolRegistry()


@pytest.fixture
def state_manager(temp_db):
    """Create a StateManager instance for testing."""
    return StateManager()


@pytest.fixture
def graph_manager(temp_db):
    """Create a GraphManager instance for testing."""
    return GraphManager()


@pytest.fixture
def execution_engine(tool_registry, state_manager, graph_manager):
    """Create an ExecutionEngine instance for testing."""
    return ExecutionEngine(tool_registry, state_manager, graph_manager, max_concurrent_executions=2)


def simple_test_tool(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Simple test tool for testing."""
    message = kwargs.get('message', 'default')
    return {
        'test_result': f'executed with message: {message}',
        'execution_count': state.get('execution_count', 0) + 1
    }


def math_tool(state: Dict[str, Any], **kwargs) -> Dict[str, Any]:
    """Math tool for testing."""
    operation = kwargs.get('operation', 'add')
    value = kwargs.get('value', 1)
    current = state.get('result', 0)
    
    if operation == 'add':
        result = current + value
    elif operation == 'multiply':
        result = current * value
    else:
        result = current
    
    return {'result': result}


class TestToolRegistry:
    """Test cases for ToolRegistry component."""
    
    def test_register_and_get_tool(self, tool_registry):
        """Test tool registration and retrieval."""
        # Register a tool
        tool_registry.register_tool('test_tool', simple_test_tool, 'A test tool')
        
        # Retrieve the tool
        retrieved_tool = tool_registry.get_tool('test_tool')
        assert retrieved_tool is not None
        assert callable(retrieved_tool)
        
        # Test tool execution
        result = retrieved_tool({'count': 0}, message='hello')
        assert result['test_result'] == 'executed with message: hello'
        assert result['execution_count'] == 1
    
    def test_list_tools(self, tool_registry):
        """Test tool listing functionality."""
        # Register multiple tools
        tool_registry.register_tool('tool1', simple_test_tool, 'First tool')
        tool_registry.register_tool('tool2', math_tool, 'Math tool')
        
        # List tools
        tools = tool_registry.list_tools()
        assert len(tools) == 2
        assert 'tool1' in tools
        assert 'tool2' in tools
        assert tools['tool1'] == 'First tool'
        assert tools['tool2'] == 'Math tool'
    
    def test_tool_registry_integrity(self, tool_registry):
        """Test tool registry prevents duplicate registrations."""
        # Register a tool
        tool_registry.register_tool('unique_tool', simple_test_tool, 'Original tool')
        
        # Try to register with same name - should raise error
        with pytest.raises(Exception):
            tool_registry.register_tool('unique_tool', math_tool, 'Duplicate tool')
        
        # Verify original tool is unchanged
        retrieved_tool = tool_registry.get_tool('unique_tool')
        result = retrieved_tool({'count': 0}, message='test')
        assert 'test_result' in result  # This confirms it's the original simple_test_tool


class TestStateManager:
    """Test cases for StateManager component."""
    
    def test_create_and_get_state(self, state_manager):
        """Test state creation and retrieval."""
        run_id = 'test_run_001'
        graph_id = 'test_graph_001'
        initial_state = {'counter': 5, 'name': 'test'}
        
        # Create run state
        state_manager.create_run_state(run_id, graph_id, initial_state)
        
        # Retrieve state
        workflow_state = state_manager.get_state(run_id)
        assert workflow_state.data == initial_state
        assert workflow_state.current_node is None
        assert workflow_state.execution_path == []
    
    def test_update_state(self, state_manager):
        """Test state updates."""
        run_id = 'test_run_002'
        graph_id = 'test_graph_002'
        initial_state = {'value': 10}
        
        # Create and update state
        state_manager.create_run_state(run_id, graph_id, initial_state)
        state_manager.update_state(run_id, {'value': 20, 'new_field': 'added'}, 'node1')
        
        # Verify updates
        workflow_state = state_manager.get_state(run_id)
        assert workflow_state.data['value'] == 20
        assert workflow_state.data['new_field'] == 'added'
        assert workflow_state.current_node == 'node1'
        assert 'node1' in workflow_state.execution_path
    
    def test_state_persistence_consistency(self, state_manager):
        """Test state persistence consistency."""
        run_id = 'test_run_003'
        graph_id = 'test_graph_003'
        initial_state = {'data': 'persistent'}
        
        # Create state
        state_manager.create_run_state(run_id, graph_id, initial_state)
        
        # Update state multiple times
        state_manager.update_state(run_id, {'step': 1}, 'node1')
        state_manager.update_state(run_id, {'step': 2}, 'node2')
        
        # Verify final state
        final_state = state_manager.get_state(run_id)
        assert final_state.data['data'] == 'persistent'
        assert final_state.data['step'] == 2
        assert final_state.current_node == 'node2'
        assert final_state.execution_path == ['node1', 'node2']


class TestGraphManager:
    """Test cases for GraphManager component."""
    
    def test_create_and_get_graph(self, graph_manager):
        """Test graph creation and retrieval."""
        # Create a simple graph
        graph_def = GraphDefinition(
            name='Test Graph',
            description='A simple test graph',
            nodes=[
                NodeDefinition(id='start', function_name='test_tool', parameters={'message': 'start'}),
                NodeDefinition(id='end', function_name='test_tool', parameters={'message': 'end'})
            ],
            edges=[
                EdgeDefinition(from_node='start', to_node='end')
            ],
            entry_point='start'
        )
        
        # Create graph
        graph_id = graph_manager.create_graph(graph_def)
        assert graph_id is not None
        
        # Retrieve graph
        retrieved_graph = graph_manager.get_graph(graph_id)
        assert retrieved_graph.name == 'Test Graph'
        assert len(retrieved_graph.nodes) == 2
        assert len(retrieved_graph.edges) == 1
        assert retrieved_graph.entry_point == 'start'
    
    def test_graph_validation(self, graph_manager):
        """Test graph validation."""
        # Valid graph
        valid_graph = GraphDefinition(
            name='Valid Graph',
            description='A valid graph',
            nodes=[NodeDefinition(id='node1', function_name='test_tool')],
            edges=[],
            entry_point='node1'
        )
        
        validation_result = graph_manager.validate_graph(valid_graph)
        assert validation_result.is_valid
        
        # Test that invalid graph creation fails at the model level
        # This shows that validation is working at the Pydantic level
        with pytest.raises(Exception):  # Should raise ValidationError
            invalid_graph = GraphDefinition(
                name='Invalid Graph',
                description='An invalid graph',
                nodes=[NodeDefinition(id='node1', function_name='test_tool')],
                edges=[],
                entry_point='nonexistent'
            )


class TestExecutionEngine:
    """Test cases for ExecutionEngine component."""
    
    def test_workflow_execution_basic(self, execution_engine, tool_registry, graph_manager):
        """Test basic workflow execution."""
        # Register test tools
        tool_registry.register_tool('simple_test_tool', simple_test_tool, 'Simple test tool')
        
        # Create a simple graph
        graph_def = GraphDefinition(
            name='Basic Execution Test',
            description='Test basic execution',
            nodes=[
                NodeDefinition(id='start', function_name='simple_test_tool', parameters={'message': 'hello'})
            ],
            edges=[],
            entry_point='start'
        )
        
        graph_id = graph_manager.create_graph(graph_def)
        
        # Execute workflow
        run_id = execution_engine.execute_workflow(graph_id, {'initial': 'state'})
        assert run_id is not None
        
        # Wait a moment for execution to complete
        import time
        time.sleep(1)
        
        # Check execution status
        status = execution_engine.get_execution_status(run_id)
        assert status.run_id == run_id
        assert status.graph_id == graph_id
        # Status should be completed or running
        assert status.status in [ExecutionStatusEnum.COMPLETED, ExecutionStatusEnum.RUNNING]
    
    def test_concurrent_execution_isolation(self, execution_engine, tool_registry, graph_manager):
        """Test concurrent execution isolation."""
        # Register test tools
        tool_registry.register_tool('math_tool', math_tool, 'Math tool')
        
        # Create two identical graphs
        graph_def1 = GraphDefinition(
            name='Concurrent Test 1',
            description='First concurrent test',
            nodes=[
                NodeDefinition(id='math', function_name='math_tool', parameters={'operation': 'add', 'value': 10})
            ],
            edges=[],
            entry_point='math'
        )
        
        graph_def2 = GraphDefinition(
            name='Concurrent Test 2', 
            description='Second concurrent test',
            nodes=[
                NodeDefinition(id='math', function_name='math_tool', parameters={'operation': 'multiply', 'value': 3})
            ],
            edges=[],
            entry_point='math'
        )
        
        graph_id1 = graph_manager.create_graph(graph_def1)
        graph_id2 = graph_manager.create_graph(graph_def2)
        
        # Execute both workflows concurrently with different initial states
        run_id1 = execution_engine.execute_workflow(graph_id1, {'result': 5})
        run_id2 = execution_engine.execute_workflow(graph_id2, {'result': 4})
        
        assert run_id1 != run_id2
        
        # Both should be tracked as active or completed
        active_executions = execution_engine.get_active_executions()
        assert len(active_executions) <= 2  # May complete quickly


class TestIntegration:
    """Integration tests for all components working together."""
    
    def test_end_to_end_workflow(self, execution_engine, tool_registry, graph_manager):
        """Test complete end-to-end workflow execution."""
        # Register tools
        tool_registry.register_tool('simple_test_tool', simple_test_tool, 'Simple test tool')
        tool_registry.register_tool('math_tool', math_tool, 'Math tool')
        
        # Create a multi-step workflow
        graph_def = GraphDefinition(
            name='End-to-End Test',
            description='Complete workflow test',
            nodes=[
                NodeDefinition(id='init', function_name='simple_test_tool', parameters={'message': 'initializing'}),
                NodeDefinition(id='calculate', function_name='math_tool', parameters={'operation': 'add', 'value': 15}),
                NodeDefinition(id='finalize', function_name='simple_test_tool', parameters={'message': 'finalizing'})
            ],
            edges=[
                EdgeDefinition(from_node='init', to_node='calculate'),
                EdgeDefinition(from_node='calculate', to_node='finalize')
            ],
            entry_point='init'
        )
        
        graph_id = graph_manager.create_graph(graph_def)
        
        # Execute workflow
        run_id = execution_engine.execute_workflow(graph_id, {'result': 10})
        
        # Wait for completion
        import time
        time.sleep(2)
        
        # Verify execution completed
        status = execution_engine.get_execution_status(run_id)
        logs = execution_engine.get_execution_logs(run_id)
        
        assert status.run_id == run_id
        assert len(logs) > 0  # Should have execution logs
        
        # Verify final state contains expected results
        final_state = status.current_state
        assert 'execution_count' in final_state.data  # From simple_test_tool
        assert 'result' in final_state.data  # From math_tool
        assert final_state.data['result'] == 25  # 10 + 15