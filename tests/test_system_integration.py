"""System integration tests with real database operations."""

import pytest
import tempfile
import os
import time
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, List
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from app.factory import create_app
from app.config import AppConfig
from app.storage.database import Base, get_db
from app.storage.models import WorkflowRunModel, LogEntryModel, GraphModel
from app.models.core import ExecutionStatusEnum, LogEventType


@pytest.fixture
def integration_db():
    """Create a real database for integration testing."""
    db_fd, db_path = tempfile.mkstemp(suffix='.db')
    os.close(db_fd)
    
    # Store original DATABASE_URL
    original_db_url = os.environ.get('DATABASE_URL')
    
    # Set database URL for testing
    test_db_url = f'sqlite:///{db_path}'
    os.environ['DATABASE_URL'] = test_db_url
    
    # Create engine and tables
    engine = create_engine(test_db_url, connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    
    yield db_path, engine
    
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
def integration_config():
    """Create configuration for integration testing."""
    return AppConfig(
        app_name="Integration Test Engine",
        debug=True,
        log_level="INFO",
        max_concurrent_executions=10,
        enable_performance_monitoring=True
    )


@pytest.fixture
def integration_client(integration_db, integration_config):
    """Create FastAPI client for integration testing."""
    db_path, engine = integration_db
    
    app = create_app(integration_config)
    client = TestClient(app)
    
    # Wait for startup
    time.sleep(2)
    
    yield client, engine
    
    # Cleanup
    try:
        from app.factory import get_app_state
        app_state = get_app_state()
        if app_state.execution_engine:
            app_state.execution_engine.shutdown()
    except Exception as e:
        print(f"Cleanup warning: {e}")


class TestDatabaseIntegration:
    """Test integration with database operations."""
    
    def test_workflow_persistence_lifecycle(self, integration_client):
        """Test complete workflow persistence lifecycle."""
        client, engine = integration_client
        
        # Create workflow
        workflow_data = {
            "name": "Database Integration Test",
            "description": "Test database persistence",
            "nodes": [
                {
                    "id": "persist_test",
                    "function_name": "workflow_test_function",
                    "parameters": {"message": "Database persistence test"}
                }
            ],
            "edges": [],
            "entry_point": "persist_test"
        }
        
        # Create graph via API
        response = client.post("/api/v1/graph/create", json=workflow_data)
        assert response.status_code == 200
        graph_id = response.json()["graph_id"]
        
        # Verify graph was persisted in database
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        try:
            graph_record = db.query(GraphModel).filter(GraphModel.id == graph_id).first()
            assert graph_record is not None
            assert graph_record.name == "Database Integration Test"
            assert graph_record.description == "Test database persistence"
        finally:
            db.close()
        
        # Execute workflow
        execution_data = {
            "graph_id": graph_id,
            "initial_state": {"test_data": "persistence_test"}
        }
        
        response = client.post("/api/v1/graph/run", json=execution_data)
        assert response.status_code == 200
        run_id = response.json()["run_id"]
        
        # Wait for completion
        max_attempts = 30
        for attempt in range(max_attempts):
            response = client.get(f"/api/v1/graph/state/{run_id}")
            if response.status_code == 200:
                status_data = response.json()
                if status_data["status"] in ["completed", "failed"]:
                    break
            time.sleep(1)
        
        assert status_data["status"] == "completed"
        
        # Verify workflow run was persisted
        db = SessionLocal()
        try:
            run_record = db.query(WorkflowRunModel).filter(WorkflowRunModel.id == run_id).first()
            assert run_record is not None
            assert run_record.graph_id == graph_id
            assert run_record.status == ExecutionStatusEnum.COMPLETED
            assert run_record.final_state is not None
            
            # Verify execution logs were persisted
            log_records = db.query(LogEntryModel).filter(LogEntryModel.run_id == run_id).all()
            assert len(log_records) > 0
            
            # Verify log chronological ordering
            log_timestamps = [log.timestamp for log in log_records]
            assert log_timestamps == sorted(log_timestamps)
            
        finally:
            db.close()
    
    def test_concurrent_database_operations(self, integration_client):
        """Test concurrent database operations maintain consistency."""
        client, engine = integration_client
        
        # Create workflow
        workflow_data = {
            "name": "Concurrent DB Test",
            "description": "Test concurrent database operations",
            "nodes": [
                {
                    "id": "db_task",
                    "function_name": "simple_math_function",
                    "parameters": {"operation": "add", "value": 1}
                }
            ],
            "edges": [],
            "entry_point": "db_task"
        }
        
        response = client.post("/api/v1/graph/create", json=workflow_data)
        graph_id = response.json()["graph_id"]
        
        # Start multiple concurrent executions
        num_concurrent = 10
        run_ids = []
        
        for i in range(num_concurrent):
            execution_data = {
                "graph_id": graph_id,
                "initial_state": {"result": i, "execution_id": i}
            }
            
            response = client.post("/api/v1/graph/run", json=execution_data)
            assert response.status_code == 200
            run_ids.append(response.json()["run_id"])
        
        # Wait for all to complete
        completed_runs = set()
        max_wait = 60
        start_time = time.time()
        
        while len(completed_runs) < num_concurrent and (time.time() - start_time) < max_wait:
            for run_id in run_ids:
                if run_id not in completed_runs:
                    response = client.get(f"/api/v1/graph/state/{run_id}")
                    if response.status_code == 200:
                        status_data = response.json()
                        if status_data["status"] in ["completed", "failed"]:
                            completed_runs.add(run_id)
            time.sleep(0.5)
        
        assert len(completed_runs) == num_concurrent
        
        # Verify database consistency
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        try:
            # Check all runs are in database
            run_records = db.query(WorkflowRunModel).filter(
                WorkflowRunModel.id.in_(run_ids)
            ).all()
            assert len(run_records) == num_concurrent
            
            # Check all have unique states
            final_states = []
            for record in run_records:
                if record.final_state:
                    state_data = json.loads(record.final_state)
                    final_states.append(state_data.get("execution_id"))
            
            # Should have unique execution IDs
            assert len(set(final_states)) == len(final_states)
            
            # Check logs for all runs
            log_counts = {}
            for run_id in run_ids:
                log_count = db.query(LogEntryModel).filter(
                    LogEntryModel.run_id == run_id
                ).count()
                log_counts[run_id] = log_count
                assert log_count > 0  # Each run should have logs
            
        finally:
            db.close()
    
    def test_database_error_recovery(self, integration_client):
        """Test system recovery from database errors."""
        client, engine = integration_client
        
        # Create workflow
        workflow_data = {
            "name": "DB Error Recovery Test",
            "description": "Test database error recovery",
            "nodes": [
                {
                    "id": "recovery_task",
                    "function_name": "workflow_test_function",
                    "parameters": {"message": "Recovery test"}
                }
            ],
            "edges": [],
            "entry_point": "recovery_task"
        }
        
        response = client.post("/api/v1/graph/create", json=workflow_data)
        graph_id = response.json()["graph_id"]
        
        # Start normal execution
        execution_data = {
            "graph_id": graph_id,
            "initial_state": {"test": "normal_execution"}
        }
        
        response = client.post("/api/v1/graph/run", json=execution_data)
        run_id = response.json()["run_id"]
        
        # Wait for completion
        max_attempts = 30
        for attempt in range(max_attempts):
            response = client.get(f"/api/v1/graph/state/{run_id}")
            if response.status_code == 200:
                status_data = response.json()
                if status_data["status"] in ["completed", "failed"]:
                    break
            time.sleep(1)
        
        assert status_data["status"] == "completed"
        
        # Verify system can still operate after potential database stress
        # Start another execution to verify recovery
        recovery_execution_data = {
            "graph_id": graph_id,
            "initial_state": {"test": "recovery_execution"}
        }
        
        response = client.post("/api/v1/graph/run", json=recovery_execution_data)
        assert response.status_code == 200
        recovery_run_id = response.json()["run_id"]
        
        # Wait for recovery execution
        for attempt in range(max_attempts):
            response = client.get(f"/api/v1/graph/state/{recovery_run_id}")
            if response.status_code == 200:
                status_data = response.json()
                if status_data["status"] in ["completed", "failed"]:
                    break
            time.sleep(1)
        
        assert status_data["status"] == "completed"
        
        # Verify both executions are in database
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        try:
            original_run = db.query(WorkflowRunModel).filter(
                WorkflowRunModel.id == run_id
            ).first()
            recovery_run = db.query(WorkflowRunModel).filter(
                WorkflowRunModel.id == recovery_run_id
            ).first()
            
            assert original_run is not None
            assert recovery_run is not None
            assert original_run.status == ExecutionStatusEnum.COMPLETED
            assert recovery_run.status == ExecutionStatusEnum.COMPLETED
            
        finally:
            db.close()


class TestCompleteSystemIntegration:
    """Test complete system integration across all components."""
    
    def test_end_to_end_code_review_workflow(self, integration_client):
        """Test complete code review workflow integration."""
        client, engine = integration_client
        
        # Test code for review
        test_code = '''
def simple_function(x):
    """A simple function."""
    return x * 2

def complex_function(a, b, c, d, e, f):
    """A complex function with high cyclomatic complexity."""
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    return e + f
                else:
                    return e - f
            else:
                return e * f
        else:
            return e / f if f != 0 else 0
    else:
        return 0
'''
        
        # Create code review workflow
        workflow_data = {
            "name": "Complete Code Review Integration",
            "description": "End-to-end code review workflow test",
            "nodes": [
                {
                    "id": "extract",
                    "function_name": "extract_functions",
                    "parameters": {}
                },
                {
                    "id": "analyze",
                    "function_name": "analyze_function_complexity",
                    "parameters": {}
                },
                {
                    "id": "evaluate",
                    "function_name": "evaluate_quality_threshold",
                    "parameters": {"quality_threshold": 7.0}
                },
                {
                    "id": "suggest",
                    "function_name": "generate_improvement_suggestions",
                    "parameters": {}
                },
                {
                    "id": "report",
                    "function_name": "generate_final_report",
                    "parameters": {}
                }
            ],
            "edges": [
                {"from_node": "extract", "to_node": "analyze"},
                {"from_node": "analyze", "to_node": "evaluate"},
                {
                    "from_node": "evaluate", 
                    "to_node": "suggest",
                    "condition": "state.get('threshold_met', False) == False"
                },
                {
                    "from_node": "evaluate", 
                    "to_node": "report",
                    "condition": "state.get('threshold_met', False) == True"
                },
                {"from_node": "suggest", "to_node": "report"}
            ],
            "entry_point": "extract"
        }
        
        response = client.post("/api/v1/graph/create", json=workflow_data)
        assert response.status_code == 200
        graph_id = response.json()["graph_id"]
        
        # Execute code review workflow
        execution_data = {
            "graph_id": graph_id,
            "initial_state": {
                "code_input": test_code,
                "workflow_name": "Integration Test Code Review"
            }
        }
        
        response = client.post("/api/v1/graph/run", json=execution_data)
        assert response.status_code == 200
        run_id = response.json()["run_id"]
        
        # Monitor execution progress
        execution_path = []
        max_attempts = 60
        
        for attempt in range(max_attempts):
            response = client.get(f"/api/v1/graph/state/{run_id}")
            assert response.status_code == 200
            
            status_data = response.json()
            current_path = status_data.get("current_state", {}).get("execution_path", [])
            
            if len(current_path) > len(execution_path):
                execution_path = current_path
                print(f"Execution progress: {' -> '.join(execution_path)}")
            
            if status_data["status"] in ["completed", "failed", "cancelled"]:
                break
            
            time.sleep(2)
        
        # Verify successful completion
        assert status_data["status"] == "completed"
        final_state = status_data["current_state"]["data"]
        
        # Verify code review results
        assert "final_report" in final_state
        final_report = final_state["final_report"]
        
        assert "quality_metrics" in final_report
        assert "function_details" in final_report
        assert "overall_status" in final_report
        
        quality_metrics = final_report["quality_metrics"]
        assert quality_metrics["total_functions"] == 2  # simple_function and complex_function
        assert quality_metrics["total_issues"] >= 0
        
        function_details = final_report["function_details"]
        assert len(function_details) == 2
        
        # Verify functions were analyzed
        function_names = [func["name"] for func in function_details]
        assert "simple_function" in function_names
        assert "complex_function" in function_names
        
        # Verify database persistence of complex workflow
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        try:
            run_record = db.query(WorkflowRunModel).filter(
                WorkflowRunModel.id == run_id
            ).first()
            
            assert run_record is not None
            assert run_record.status == ExecutionStatusEnum.COMPLETED
            
            # Verify execution logs capture the full workflow
            log_records = db.query(LogEntryModel).filter(
                LogEntryModel.run_id == run_id
            ).order_by(LogEntryModel.timestamp).all()
            
            assert len(log_records) >= 5  # At least one log per node
            
            # Verify log types
            log_types = [log.event_type for log in log_records]
            assert LogEventType.WORKFLOW_START in log_types
            assert LogEventType.WORKFLOW_COMPLETE in log_types
            assert LogEventType.NODE_START in log_types
            assert LogEventType.NODE_COMPLETE in log_types
            
        finally:
            db.close()
    
    def test_system_resilience_under_stress(self, integration_client):
        """Test system resilience under various stress conditions."""
        client, engine = integration_client
        
        # Create multiple different workflows
        workflows = []
        
        # Simple workflow
        simple_workflow = {
            "name": "Stress Test Simple",
            "description": "Simple workflow for stress testing",
            "nodes": [
                {
                    "id": "simple_task",
                    "function_name": "workflow_test_function",
                    "parameters": {"message": "Simple stress test"}
                }
            ],
            "edges": [],
            "entry_point": "simple_task"
        }
        
        # Complex workflow
        complex_workflow = {
            "name": "Stress Test Complex",
            "description": "Complex workflow for stress testing",
            "nodes": [
                {
                    "id": "init",
                    "function_name": "simple_math_function",
                    "parameters": {"operation": "add", "value": 5}
                },
                {
                    "id": "process",
                    "function_name": "simple_math_function",
                    "parameters": {"operation": "multiply", "value": 3}
                },
                {
                    "id": "check",
                    "function_name": "conditional_function",
                    "parameters": {"threshold": 10}
                },
                {
                    "id": "finalize",
                    "function_name": "workflow_test_function",
                    "parameters": {"message": "Complex stress test complete"}
                }
            ],
            "edges": [
                {"from_node": "init", "to_node": "process"},
                {"from_node": "process", "to_node": "check"},
                {"from_node": "check", "to_node": "finalize"}
            ],
            "entry_point": "init"
        }
        
        # Create workflows
        for workflow_data in [simple_workflow, complex_workflow]:
            response = client.post("/api/v1/graph/create", json=workflow_data)
            assert response.status_code == 200
            workflows.append(response.json()["graph_id"])
        
        # Execute multiple workflows concurrently with different patterns
        import threading
        import random
        
        execution_results = []
        execution_lock = threading.Lock()
        
        def execute_random_workflows(thread_id: int, num_executions: int):
            """Execute random workflows from different threads."""
            thread_results = []
            
            for i in range(num_executions):
                try:
                    # Randomly select workflow
                    graph_id = random.choice(workflows)
                    
                    execution_data = {
                        "graph_id": graph_id,
                        "initial_state": {
                            "thread_id": thread_id,
                            "execution_id": i,
                            "result": random.randint(1, 20),
                            "value": random.randint(1, 30)
                        }
                    }
                    
                    response = client.post("/api/v1/graph/run", json=execution_data)
                    if response.status_code == 200:
                        run_id = response.json()["run_id"]
                        thread_results.append({
                            "thread_id": thread_id,
                            "execution_id": i,
                            "run_id": run_id,
                            "graph_id": graph_id,
                            "started": True
                        })
                    else:
                        thread_results.append({
                            "thread_id": thread_id,
                            "execution_id": i,
                            "error": "Failed to start",
                            "started": False
                        })
                    
                    # Random delay to simulate realistic usage
                    time.sleep(random.uniform(0.1, 0.5))
                    
                except Exception as e:
                    thread_results.append({
                        "thread_id": thread_id,
                        "execution_id": i,
                        "error": str(e),
                        "started": False
                    })
            
            with execution_lock:
                execution_results.extend(thread_results)
        
        # Start stress test with multiple threads
        num_threads = 5
        executions_per_thread = 10
        threads = []
        
        start_time = time.time()
        
        for thread_id in range(num_threads):
            thread = threading.Thread(
                target=execute_random_workflows,
                args=(thread_id, executions_per_thread)
            )
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        stress_duration = time.time() - start_time
        
        # Analyze stress test results
        total_executions = len(execution_results)
        successful_starts = sum(1 for result in execution_results if result.get("started", False))
        failed_starts = total_executions - successful_starts
        
        print(f"\n=== Stress Test Results ===")
        print(f"Duration: {stress_duration:.2f}s")
        print(f"Total execution attempts: {total_executions}")
        print(f"Successful starts: {successful_starts}")
        print(f"Failed starts: {failed_starts}")
        print(f"Success rate: {successful_starts / total_executions:.2%}")
        
        # Wait for executions to complete and verify database consistency
        run_ids = [result["run_id"] for result in execution_results if result.get("started", False)]
        completed_count = 0
        max_wait = 120  # seconds
        wait_start = time.time()
        
        while completed_count < len(run_ids) and (time.time() - wait_start) < max_wait:
            completed_count = 0
            for run_id in run_ids:
                response = client.get(f"/api/v1/graph/state/{run_id}")
                if response.status_code == 200:
                    status_data = response.json()
                    if status_data["status"] in ["completed", "failed", "cancelled"]:
                        completed_count += 1
            
            if completed_count < len(run_ids):
                time.sleep(2)
        
        completion_rate = completed_count / len(run_ids) if run_ids else 0
        print(f"Completion rate: {completion_rate:.2%}")
        
        # Verify database integrity after stress test
        SessionLocal = sessionmaker(bind=engine)
        db = SessionLocal()
        try:
            # Check database consistency
            total_runs_in_db = db.query(WorkflowRunModel).count()
            total_logs_in_db = db.query(LogEntryModel).count()
            
            print(f"Total runs in database: {total_runs_in_db}")
            print(f"Total logs in database: {total_logs_in_db}")
            
            # Verify no orphaned logs (logs without corresponding runs)
            orphaned_logs = db.execute(text("""
                SELECT COUNT(*) FROM log_entries l 
                WHERE NOT EXISTS (
                    SELECT 1 FROM workflow_runs w WHERE w.id = l.run_id
                )
            """)).scalar()
            
            assert orphaned_logs == 0, f"Found {orphaned_logs} orphaned log entries"
            
            # Verify referential integrity
            runs_with_logs = db.execute(text("""
                SELECT COUNT(DISTINCT w.id) FROM workflow_runs w
                JOIN log_entries l ON w.id = l.run_id
            """)).scalar()
            
            print(f"Runs with logs: {runs_with_logs}")
            
        finally:
            db.close()
        
        # Stress test assertions
        assert successful_starts / total_executions >= 0.90, f"Too many failed starts: {failed_starts}/{total_executions}"
        assert completion_rate >= 0.85, f"Too many incomplete executions: {completion_rate:.2%}"
        assert stress_duration < 180, f"Stress test took too long: {stress_duration:.2f}s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])