"""Performance tests for concurrent execution scenarios."""

import pytest
import time
import threading
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from fastapi.testclient import TestClient

from app.factory import create_app
from app.config import AppConfig
from app.models.core import GraphDefinition, NodeDefinition, EdgeDefinition


@pytest.fixture
def performance_config():
    """Create configuration optimized for performance testing."""
    return AppConfig(
        app_name="Performance Test Engine",
        debug=False,  # Disable debug for better performance
        log_level="WARNING",  # Reduce logging overhead
        max_concurrent_executions=20,  # Higher concurrency limit
        enable_performance_monitoring=True,
        slow_request_threshold=5.0
    )


@pytest.fixture
def perf_client(performance_config):
    """Create FastAPI test client optimized for performance testing."""
    app = create_app(performance_config)
    client = TestClient(app)
    
    # Wait for startup
    time.sleep(2)
    
    yield client
    
    # Cleanup
    try:
        from app.factory import get_app_state
        app_state = get_app_state()
        if app_state.execution_engine:
            app_state.execution_engine.shutdown()
    except Exception as e:
        print(f"Cleanup warning: {e}")


class PerformanceMetrics:
    """Helper class to collect and analyze performance metrics."""
    
    def __init__(self):
        self.execution_times: List[float] = []
        self.throughput_data: List[Dict[str, Any]] = []
        self.error_count = 0
        self.success_count = 0
    
    def record_execution(self, duration: float, success: bool = True):
        """Record an execution time and success status."""
        self.execution_times.append(duration)
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    def record_throughput(self, timestamp: datetime, completed_count: int, total_count: int):
        """Record throughput data point."""
        self.throughput_data.append({
            'timestamp': timestamp,
            'completed': completed_count,
            'total': total_count,
            'completion_rate': completed_count / total_count if total_count > 0 else 0
        })
    
    def get_statistics(self) -> Dict[str, Any]:
        """Calculate performance statistics."""
        if not self.execution_times:
            return {"error": "No execution data recorded"}
        
        return {
            'total_executions': len(self.execution_times),
            'success_count': self.success_count,
            'error_count': self.error_count,
            'success_rate': self.success_count / len(self.execution_times),
            'avg_execution_time': statistics.mean(self.execution_times),
            'median_execution_time': statistics.median(self.execution_times),
            'min_execution_time': min(self.execution_times),
            'max_execution_time': max(self.execution_times),
            'std_dev_execution_time': statistics.stdev(self.execution_times) if len(self.execution_times) > 1 else 0,
            'p95_execution_time': self._percentile(self.execution_times, 95),
            'p99_execution_time': self._percentile(self.execution_times, 99)
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile value."""
        sorted_data = sorted(data)
        index = int((percentile / 100) * len(sorted_data))
        return sorted_data[min(index, len(sorted_data) - 1)]


class TestConcurrentExecutionPerformance:
    """Test performance under various concurrent execution scenarios."""
    
    def test_high_concurrency_simple_workflows(self, perf_client):
        """Test performance with high concurrency on simple workflows."""
        # Create a simple, fast-executing workflow
        workflow_data = {
            "name": "High Concurrency Test",
            "description": "Simple workflow for concurrency testing",
            "nodes": [
                {
                    "id": "fast_task",
                    "function_name": "simple_math_function",
                    "parameters": {"operation": "add", "value": 1}
                }
            ],
            "edges": [],
            "entry_point": "fast_task"
        }
        
        response = perf_client.post("/api/v1/graph/create", json=workflow_data)
        assert response.status_code == 200
        graph_id = response.json()["graph_id"]
        
        # Performance test parameters
        num_concurrent_executions = 50
        metrics = PerformanceMetrics()
        
        def execute_workflow(execution_id: int) -> Dict[str, Any]:
            """Execute a single workflow and measure performance."""
            start_time = time.time()
            
            try:
                # Start execution
                execution_data = {
                    "graph_id": graph_id,
                    "initial_state": {"result": execution_id}
                }
                
                response = perf_client.post("/api/v1/graph/run", json=execution_data)
                if response.status_code != 200:
                    return {"success": False, "error": "Failed to start execution"}
                
                run_id = response.json()["run_id"]
                
                # Wait for completion
                max_wait = 30  # seconds
                start_wait = time.time()
                
                while (time.time() - start_wait) < max_wait:
                    response = perf_client.get(f"/api/v1/graph/state/{run_id}")
                    if response.status_code == 200:
                        status_data = response.json()
                        if status_data["status"] in ["completed", "failed", "cancelled"]:
                            end_time = time.time()
                            duration = end_time - start_time
                            
                            return {
                                "success": status_data["status"] == "completed",
                                "duration": duration,
                                "run_id": run_id,
                                "final_result": status_data.get("current_state", {}).get("data", {}).get("result")
                            }
                    
                    time.sleep(0.1)
                
                return {"success": False, "error": "Timeout waiting for completion"}
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                return {"success": False, "error": str(e), "duration": duration}
        
        # Execute workflows concurrently
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            # Submit all tasks
            future_to_id = {
                executor.submit(execute_workflow, i): i 
                for i in range(num_concurrent_executions)
            }
            
            # Collect results
            completed_count = 0
            for future in as_completed(future_to_id):
                execution_id = future_to_id[future]
                try:
                    result = future.result()
                    
                    if "duration" in result:
                        metrics.record_execution(result["duration"], result["success"])
                    else:
                        metrics.record_execution(0, False)
                    
                    completed_count += 1
                    
                    # Record throughput periodically
                    if completed_count % 10 == 0:
                        metrics.record_throughput(
                            datetime.utcnow(), 
                            completed_count, 
                            num_concurrent_executions
                        )
                    
                except Exception as e:
                    print(f"Execution {execution_id} failed: {e}")
                    metrics.record_execution(0, False)
                    completed_count += 1
        
        total_time = time.time() - start_time
        
        # Analyze performance
        stats = metrics.get_statistics()
        
        print(f"\n=== High Concurrency Performance Results ===")
        print(f"Total executions: {stats['total_executions']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average execution time: {stats['avg_execution_time']:.3f}s")
        print(f"Median execution time: {stats['median_execution_time']:.3f}s")
        print(f"95th percentile: {stats['p95_execution_time']:.3f}s")
        print(f"99th percentile: {stats['p99_execution_time']:.3f}s")
        print(f"Throughput: {stats['total_executions'] / total_time:.2f} executions/second")
        
        # Performance assertions
        assert stats['success_rate'] >= 0.95, f"Success rate too low: {stats['success_rate']:.2%}"
        assert stats['avg_execution_time'] < 5.0, f"Average execution time too high: {stats['avg_execution_time']:.3f}s"
        assert total_time < 60, f"Total execution time too high: {total_time:.2f}s"
        assert stats['p95_execution_time'] < 10.0, f"95th percentile too high: {stats['p95_execution_time']:.3f}s"
    
    def test_complex_workflow_performance(self, perf_client):
        """Test performance with complex multi-step workflows."""
        # Create a more complex workflow
        workflow_data = {
            "name": "Complex Performance Test",
            "description": "Multi-step workflow for performance testing",
            "nodes": [
                {
                    "id": "init",
                    "function_name": "simple_math_function",
                    "parameters": {"operation": "add", "value": 10}
                },
                {
                    "id": "process1",
                    "function_name": "simple_math_function",
                    "parameters": {"operation": "multiply", "value": 2}
                },
                {
                    "id": "process2",
                    "function_name": "simple_math_function",
                    "parameters": {"operation": "add", "value": 5}
                },
                {
                    "id": "check",
                    "function_name": "conditional_function",
                    "parameters": {"threshold": 30}
                },
                {
                    "id": "finalize_high",
                    "function_name": "workflow_test_function",
                    "parameters": {"message": "High value result"}
                },
                {
                    "id": "finalize_low",
                    "function_name": "workflow_test_function",
                    "parameters": {"message": "Low value result"}
                }
            ],
            "edges": [
                {"from_node": "init", "to_node": "process1"},
                {"from_node": "process1", "to_node": "process2"},
                {"from_node": "process2", "to_node": "check"},
                {
                    "from_node": "check", 
                    "to_node": "finalize_high",
                    "condition": "state.get('meets_threshold', False) == True"
                },
                {
                    "from_node": "check", 
                    "to_node": "finalize_low",
                    "condition": "state.get('meets_threshold', False) == False"
                }
            ],
            "entry_point": "init"
        }
        
        response = perf_client.post("/api/v1/graph/create", json=workflow_data)
        assert response.status_code == 200
        graph_id = response.json()["graph_id"]
        
        # Test with moderate concurrency
        num_executions = 20
        metrics = PerformanceMetrics()
        
        def execute_complex_workflow(execution_id: int) -> Dict[str, Any]:
            """Execute complex workflow and measure performance."""
            start_time = time.time()
            
            try:
                execution_data = {
                    "graph_id": graph_id,
                    "initial_state": {"result": execution_id, "value": execution_id}
                }
                
                response = perf_client.post("/api/v1/graph/run", json=execution_data)
                if response.status_code != 200:
                    return {"success": False, "error": "Failed to start execution"}
                
                run_id = response.json()["run_id"]
                
                # Wait for completion with longer timeout for complex workflow
                max_wait = 60
                start_wait = time.time()
                
                while (time.time() - start_wait) < max_wait:
                    response = perf_client.get(f"/api/v1/graph/state/{run_id}")
                    if response.status_code == 200:
                        status_data = response.json()
                        if status_data["status"] in ["completed", "failed", "cancelled"]:
                            end_time = time.time()
                            duration = end_time - start_time
                            
                            return {
                                "success": status_data["status"] == "completed",
                                "duration": duration,
                                "run_id": run_id,
                                "execution_path": status_data.get("current_state", {}).get("execution_path", [])
                            }
                    
                    time.sleep(0.2)
                
                return {"success": False, "error": "Timeout waiting for completion"}
                
            except Exception as e:
                end_time = time.time()
                duration = end_time - start_time
                return {"success": False, "error": str(e), "duration": duration}
        
        # Execute complex workflows
        start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_id = {
                executor.submit(execute_complex_workflow, i): i 
                for i in range(num_executions)
            }
            
            for future in as_completed(future_to_id):
                result = future.result()
                if "duration" in result:
                    metrics.record_execution(result["duration"], result["success"])
                else:
                    metrics.record_execution(0, False)
        
        total_time = time.time() - start_time
        stats = metrics.get_statistics()
        
        print(f"\n=== Complex Workflow Performance Results ===")
        print(f"Total executions: {stats['total_executions']}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Average execution time: {stats['avg_execution_time']:.3f}s")
        print(f"Median execution time: {stats['median_execution_time']:.3f}s")
        print(f"95th percentile: {stats['p95_execution_time']:.3f}s")
        
        # Performance assertions for complex workflows
        assert stats['success_rate'] >= 0.90, f"Success rate too low: {stats['success_rate']:.2%}"
        assert stats['avg_execution_time'] < 15.0, f"Average execution time too high: {stats['avg_execution_time']:.3f}s"
        assert total_time < 120, f"Total execution time too high: {total_time:.2f}s"
    
    def test_sustained_load_performance(self, perf_client):
        """Test performance under sustained load over time."""
        # Create workflow for sustained testing
        workflow_data = {
            "name": "Sustained Load Test",
            "description": "Workflow for sustained load testing",
            "nodes": [
                {
                    "id": "load_task",
                    "function_name": "simple_math_function",
                    "parameters": {"operation": "multiply", "value": 2}
                }
            ],
            "edges": [],
            "entry_point": "load_task"
        }
        
        response = perf_client.post("/api/v1/graph/create", json=workflow_data)
        assert response.status_code == 200
        graph_id = response.json()["graph_id"]
        
        # Sustained load parameters
        test_duration = 30  # seconds
        executions_per_second = 5
        total_expected_executions = test_duration * executions_per_second
        
        metrics = PerformanceMetrics()
        execution_counter = 0
        start_time = time.time()
        
        def continuous_execution():
            """Continuously execute workflows for the test duration."""
            nonlocal execution_counter
            
            while (time.time() - start_time) < test_duration:
                execution_start = time.time()
                
                try:
                    execution_data = {
                        "graph_id": graph_id,
                        "initial_state": {"result": execution_counter}
                    }
                    
                    response = perf_client.post("/api/v1/graph/run", json=execution_data)
                    if response.status_code == 200:
                        run_id = response.json()["run_id"]
                        
                        # Quick check for completion (don't wait long)
                        max_quick_wait = 5
                        quick_start = time.time()
                        
                        while (time.time() - quick_start) < max_quick_wait:
                            response = perf_client.get(f"/api/v1/graph/state/{run_id}")
                            if response.status_code == 200:
                                status_data = response.json()
                                if status_data["status"] in ["completed", "failed"]:
                                    execution_end = time.time()
                                    duration = execution_end - execution_start
                                    metrics.record_execution(duration, status_data["status"] == "completed")
                                    break
                            time.sleep(0.1)
                        else:
                            # Timeout - record as ongoing
                            metrics.record_execution(time.time() - execution_start, True)
                    else:
                        metrics.record_execution(time.time() - execution_start, False)
                    
                    execution_counter += 1
                    
                    # Rate limiting
                    elapsed = time.time() - execution_start
                    target_interval = 1.0 / executions_per_second
                    if elapsed < target_interval:
                        time.sleep(target_interval - elapsed)
                        
                except Exception as e:
                    metrics.record_execution(time.time() - execution_start, False)
                    execution_counter += 1
        
        # Run sustained load test
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Start multiple threads for sustained load
            futures = [executor.submit(continuous_execution) for _ in range(3)]
            
            # Wait for completion
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Sustained load thread error: {e}")
        
        total_time = time.time() - start_time
        stats = metrics.get_statistics()
        
        print(f"\n=== Sustained Load Performance Results ===")
        print(f"Test duration: {total_time:.2f}s")
        print(f"Total executions: {stats['total_executions']}")
        print(f"Target executions: {total_expected_executions}")
        print(f"Success rate: {stats['success_rate']:.2%}")
        print(f"Actual throughput: {stats['total_executions'] / total_time:.2f} executions/second")
        print(f"Target throughput: {executions_per_second} executions/second")
        print(f"Average execution time: {stats['avg_execution_time']:.3f}s")
        
        # Sustained load assertions
        assert stats['success_rate'] >= 0.85, f"Success rate too low under sustained load: {stats['success_rate']:.2%}"
        actual_throughput = stats['total_executions'] / total_time
        assert actual_throughput >= executions_per_second * 0.7, f"Throughput too low: {actual_throughput:.2f} < {executions_per_second * 0.7:.2f}"
        assert stats['avg_execution_time'] < 10.0, f"Average execution time degraded: {stats['avg_execution_time']:.3f}s"


class TestMemoryAndResourceUsage:
    """Test memory usage and resource management under load."""
    
    def test_memory_stability_under_load(self, perf_client):
        """Test that memory usage remains stable under sustained load."""
        import psutil
        import os
        
        # Get current process for memory monitoring
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create workflow
        workflow_data = {
            "name": "Memory Stability Test",
            "description": "Test memory usage stability",
            "nodes": [
                {
                    "id": "memory_task",
                    "function_name": "workflow_test_function",
                    "parameters": {"message": "Memory test execution"}
                }
            ],
            "edges": [],
            "entry_point": "memory_task"
        }
        
        response = perf_client.post("/api/v1/graph/create", json=workflow_data)
        graph_id = response.json()["graph_id"]
        
        # Execute many workflows and monitor memory
        num_executions = 100
        memory_samples = []
        
        for i in range(num_executions):
            execution_data = {
                "graph_id": graph_id,
                "initial_state": {"execution": i}
            }
            
            response = perf_client.post("/api/v1/graph/run", json=execution_data)
            assert response.status_code == 200
            
            # Sample memory every 10 executions
            if i % 10 == 0:
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_samples.append(current_memory)
                print(f"Execution {i}: Memory usage: {current_memory:.2f} MB")
            
            # Brief pause to allow cleanup
            if i % 20 == 0:
                time.sleep(0.5)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_growth = final_memory - initial_memory
        
        print(f"\n=== Memory Stability Results ===")
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"Final memory: {final_memory:.2f} MB")
        print(f"Memory growth: {memory_growth:.2f} MB")
        print(f"Memory samples: {memory_samples}")
        
        # Memory stability assertions
        # Allow some growth but not excessive
        max_acceptable_growth = 100  # MB
        assert memory_growth < max_acceptable_growth, f"Memory growth too high: {memory_growth:.2f} MB"
        
        # Check for memory leaks (continuous growth)
        if len(memory_samples) >= 3:
            # Calculate trend - should not be consistently increasing
            recent_samples = memory_samples[-3:]
            growth_trend = recent_samples[-1] - recent_samples[0]
            assert growth_trend < 50, f"Potential memory leak detected: {growth_trend:.2f} MB growth in recent samples"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-s"])