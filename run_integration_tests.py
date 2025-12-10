#!/usr/bin/env python3
"""
Comprehensive integration test runner for the Agent Workflow Engine.

This script demonstrates the complete system integration by running various
test scenarios that validate end-to-end functionality, concurrent execution,
database persistence, and system resilience.
"""

import sys
import os
import time
import subprocess
from datetime import datetime

def run_command(command, description, timeout=120):
    """Run a command and capture its output."""
    print(f"\n{'='*60}")
    print(f"🧪 {description}")
    print(f"{'='*60}")
    print(f"Command: {command}")
    print()
    
    start_time = time.time()
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout
        )
        
        duration = time.time() - start_time
        
        if result.returncode == 0:
            print(f"✅ PASSED ({duration:.2f}s)")
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
        else:
            print(f"❌ FAILED ({duration:.2f}s)")
            if result.stderr:
                print("\nError:")
                print(result.stderr)
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"⏰ TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"💥 EXCEPTION: {str(e)}")
        return False


def main():
    """Run comprehensive integration tests."""
    print("🚀 Agent Workflow Engine - Final Integration Testing")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test scenarios to run
    test_scenarios = [
        {
            "command": "python -m pytest tests/test_core_components.py::TestIntegration::test_end_to_end_workflow -v",
            "description": "End-to-End Workflow Integration Test",
            "timeout": 60
        },
        {
            "command": "python -m pytest tests/test_core_components.py::TestExecutionEngine::test_concurrent_execution_isolation -v",
            "description": "Concurrent Execution Isolation Test",
            "timeout": 60
        },
        {
            "command": "python -m pytest tests/test_core_components.py::TestStateManager::test_state_persistence_consistency -v",
            "description": "State Persistence Consistency Test",
            "timeout": 60
        },
        {
            "command": "python -m pytest tests/test_core_components.py::TestToolRegistry::test_tool_registry_integrity -v",
            "description": "Tool Registry Integrity Test",
            "timeout": 30
        },
        {
            "command": "python -m pytest tests/test_core_components.py::TestGraphManager::test_graph_validation -v",
            "description": "Graph Validation Test",
            "timeout": 30
        },
        {
            "command": "python test_code_review_integration.py",
            "description": "Code Review Workflow Integration Test",
            "timeout": 120
        },
        {
            "command": "python -m pytest tests/test_websocket_comprehensive.py::TestWebSocketIntegration::test_workflow_monitoring -v",
            "description": "WebSocket Real-time Monitoring Test",
            "timeout": 60
        }
    ]
    
    # Run all test scenarios
    passed_tests = 0
    total_tests = len(test_scenarios)
    
    for i, scenario in enumerate(test_scenarios, 1):
        print(f"\n📋 Running Test {i}/{total_tests}")
        
        success = run_command(
            scenario["command"],
            scenario["description"],
            scenario.get("timeout", 120)
        )
        
        if success:
            passed_tests += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 FINAL INTEGRATION TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {(passed_tests / total_tests) * 100:.1f}%")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL INTEGRATION TESTS PASSED!")
        print("✅ The Agent Workflow Engine is fully integrated and operational.")
        print("\nKey capabilities verified:")
        print("  • End-to-end workflow execution")
        print("  • Concurrent execution with state isolation")
        print("  • Database persistence and consistency")
        print("  • Tool registry management")
        print("  • Graph validation and creation")
        print("  • Code review workflow implementation")
        print("  • Real-time WebSocket monitoring")
        
        return 0
    else:
        print(f"\n⚠️  {total_tests - passed_tests} INTEGRATION TESTS FAILED")
        print("❌ Some system components may need attention.")
        
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)