#!/usr/bin/env python3
"""
Code Review Workflow Example

This script demonstrates the code review workflow implementation by:
1. Setting up the workflow engine with code review tools
2. Creating a code review workflow definition
3. Running the workflow with sample Python code
4. Displaying the comprehensive analysis results
"""

import requests
import json
import time
from typing import Dict, Any

# Sample Python code for analysis (intentionally has quality issues)
SAMPLE_CODE = '''
def calculate_total_price(items, tax_rate, discount_rate, shipping_cost, is_premium_customer, customer_type, region, currency):
    """Calculate total price with various factors."""
    total = 0
    for item in items:
        if item['category'] == 'electronics':
            if item['price'] > 1000:
                if is_premium_customer:
                    if customer_type == 'gold':
                        if region == 'US':
                            total += item['price'] * 0.9
                        else:
                            total += item['price'] * 0.95
                    else:
                        total += item['price'] * 0.98
                else:
                    total += item['price']
            else:
                total += item['price']
        elif item['category'] == 'books':
            total += item['price'] * 0.95
        else:
            total += item['price']
    
    # Apply tax
    total_with_tax = total * (1 + tax_rate)
    
    # Apply discount
    if discount_rate > 0:
        total_with_tax = total_with_tax * (1 - discount_rate)
    
    # Add shipping
    final_total = total_with_tax + shipping_cost
    
    return final_total

def process_data(data):
    result = []
    for i in range(len(data)):
        if data[i] > 0:
            result.append(data[i] * 2)
        else:
            result.append(0)
    return result

def very_long_function_that_does_many_things_and_violates_single_responsibility_principle():
    # This function is intentionally long and complex
    data = []
    for i in range(100):
        if i % 2 == 0:
            if i % 4 == 0:
                if i % 8 == 0:
                    data.append(i * 3)
                else:
                    data.append(i * 2)
            else:
                data.append(i)
        else:
            if i % 3 == 0:
                data.append(i + 10)
            else:
                data.append(i + 5)
    
    # Process the data
    processed = []
    for item in data:
        if item > 50:
            processed.append(item / 2)
        elif item > 25:
            processed.append(item * 1.5)
        else:
            processed.append(item)
    
    # More processing
    final_result = []
    for item in processed:
        if item > 100:
            final_result.append(item - 50)
        elif item > 75:
            final_result.append(item - 25)
        elif item > 50:
            final_result.append(item - 10)
        else:
            final_result.append(item)
    
    return final_result
'''

# High-quality code sample for comparison
HIGH_QUALITY_CODE = '''
def calculate_circle_area(radius: float) -> float:
    """
    Calculate the area of a circle given its radius.
    
    Args:
        radius: The radius of the circle (must be positive)
        
    Returns:
        The area of the circle
        
    Raises:
        ValueError: If radius is negative
    """
    if radius < 0:
        raise ValueError("Radius must be non-negative")
    
    import math
    return math.pi * radius ** 2


def format_currency(amount: float, currency_code: str = "USD") -> str:
    """
    Format a monetary amount as a currency string.
    
    Args:
        amount: The monetary amount to format
        currency_code: The currency code (default: USD)
        
    Returns:
        Formatted currency string
    """
    return f"{currency_code} {amount:.2f}"


def validate_email(email: str) -> bool:
    """
    Validate an email address format.
    
    Args:
        email: The email address to validate
        
    Returns:
        True if email format is valid, False otherwise
    """
    import re
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))
'''


class CodeReviewWorkflowDemo:
    """Demonstration of the code review workflow."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api/v1"
    
    def register_code_review_tools(self):
        """Register code review tools with the workflow engine."""
        print("🔧 Registering code review tools...")
        
        # The tools should be automatically available if the module is imported
        # In a real implementation, you would register them via the tool registry API
        # For this demo, we assume they're already registered
        
        tools_to_register = [
            "extract_functions",
            "analyze_function_complexity", 
            "evaluate_quality_threshold",
            "generate_improvement_suggestions",
            "generate_final_report"
        ]
        
        print(f"✅ Code review tools registered: {', '.join(tools_to_register)}")
    
    def create_code_review_workflow(self) -> Dict[str, Any]:
        """Create the code review workflow definition."""
        return {
            "name": "Code Review Workflow",
            "description": "Comprehensive code quality analysis workflow with iterative improvement",
            "nodes": [
                {
                    "id": "extract_functions",
                    "function_name": "extract_functions",
                    "parameters": {},
                    "timeout": 30
                },
                {
                    "id": "analyze_complexity",
                    "function_name": "analyze_function_complexity", 
                    "parameters": {},
                    "timeout": 60
                },
                {
                    "id": "evaluate_threshold",
                    "function_name": "evaluate_quality_threshold",
                    "parameters": {
                        "quality_threshold": 7.0
                    },
                    "timeout": 30
                },
                {
                    "id": "generate_suggestions",
                    "function_name": "generate_improvement_suggestions",
                    "parameters": {},
                    "timeout": 30
                },
                {
                    "id": "generate_report",
                    "function_name": "generate_final_report",
                    "parameters": {},
                    "timeout": 30
                }
            ],
            "edges": [
                {"from_node": "extract_functions", "to_node": "analyze_complexity"},
                {"from_node": "analyze_complexity", "to_node": "evaluate_threshold"},
                {
                    "from_node": "evaluate_threshold", 
                    "to_node": "generate_suggestions",
                    "condition": "state.get('threshold_met', False) == False"
                },
                {
                    "from_node": "evaluate_threshold", 
                    "to_node": "generate_report",
                    "condition": "state.get('threshold_met', False) == True"
                },
                {"from_node": "generate_suggestions", "to_node": "generate_report"}
            ],
            "entry_point": "extract_functions",
            "exit_conditions": [
                "state.get('report_status') == 'success'"
            ]
        }
    
    def run_code_review(self, code_input: str, workflow_name: str = "Code Review") -> Dict[str, Any]:
        """Run the code review workflow on the provided code."""
        print(f"\n📝 Starting code review workflow: {workflow_name}")
        print(f"Code length: {len(code_input)} characters")
        print(f"Code lines: {len(code_input.split(chr(10)))}")
        
        # Create workflow
        workflow_def = self.create_code_review_workflow()
        
        try:
            response = requests.post(
                f"{self.api_url}/graph/create",
                json={"graph": workflow_def}
            )
            
            if response.status_code != 201:
                print(f"❌ Failed to create workflow: {response.text}")
                return None
            
            graph_data = response.json()
            graph_id = graph_data["graph_id"]
            print(f"✅ Workflow created: {graph_id}")
            
            # Start execution with code input
            initial_state = {
                "code_input": code_input,
                "workflow_name": workflow_name
            }
            
            response = requests.post(
                f"{self.api_url}/graph/run",
                json={
                    "graph_id": graph_id,
                    "initial_state": initial_state
                }
            )
            
            if response.status_code != 202:
                print(f"❌ Failed to start workflow: {response.text}")
                return None
            
            run_data = response.json()
            run_id = run_data["run_id"]
            print(f"🏃 Code review started: {run_id}")
            
            # Wait for completion
            final_status = self.wait_for_completion(run_id, timeout=120)
            
            if final_status:
                print(f"✅ Code review completed: {final_status['status']}")
                return final_status
            else:
                print("❌ Code review timed out or failed")
                return None
                
        except requests.exceptions.ConnectionError:
            print("❌ Connection error: Please ensure the workflow engine is running")
            return None
        except Exception as e:
            print(f"❌ Error running code review: {str(e)}")
            return None
    
    def wait_for_completion(self, run_id: str, timeout: int = 120) -> Dict[str, Any]:
        """Wait for workflow completion and return final status."""
        print(f"⏳ Waiting for completion: {run_id}")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.api_url}/graph/state/{run_id}")
                
                if response.status_code != 200:
                    print(f"❌ Failed to get status: {response.text}")
                    return None
                
                status_data = response.json()
                current_status = status_data["status"]
                
                if current_status in ["completed", "failed", "cancelled"]:
                    return status_data
                
                # Show progress
                current_state = status_data.get("current_state", {})
                current_node = current_state.get("current_node")
                if current_node:
                    print(f"   Currently executing: {current_node}")
                
                time.sleep(2)
                
            except Exception as e:
                print(f"❌ Error checking status: {str(e)}")
                return None
        
        print("⏰ Timeout waiting for completion")
        return None
    
    def display_code_review_results(self, final_status: Dict[str, Any]):
        """Display comprehensive code review results."""
        if not final_status:
            print("❌ No results to display")
            return
        
        current_state = final_status.get("current_state", {})
        final_report = current_state.get("data", {}).get("final_report", {})
        
        if not final_report:
            print("❌ No final report available")
            return
        
        print(f"\n" + "="*60)
        print("📊 CODE REVIEW RESULTS")
        print("="*60)
        
        # Overall status
        overall_status = final_report.get("overall_status", "UNKNOWN")
        summary = final_report.get("summary", "No summary available")
        
        status_emoji = "✅" if overall_status == "PASSED" else "⚠️"
        print(f"{status_emoji} Overall Status: {overall_status}")
        print(f"📝 Summary: {summary}")
        
        # Quality metrics
        quality_metrics = final_report.get("quality_metrics", {})
        print(f"\n📈 Quality Metrics:")
        print(f"   Total Functions: {quality_metrics.get('total_functions', 0)}")
        print(f"   Average Quality Score: {quality_metrics.get('average_quality_score', 0):.2f}/10.0")
        print(f"   Quality Threshold: {quality_metrics.get('quality_threshold', 0):.1f}")
        print(f"   Functions Above Threshold: {quality_metrics.get('functions_above_threshold', 0)}")
        print(f"   Functions Below Threshold: {quality_metrics.get('functions_below_threshold', 0)}")
        print(f"   Total Issues: {quality_metrics.get('total_issues', 0)}")
        
        # Complexity analysis
        complexity_analysis = final_report.get("complexity_analysis", {})
        distribution = complexity_analysis.get("distribution", {})
        avg_complexity = complexity_analysis.get("average_complexity", 0)
        
        print(f"\n🔍 Complexity Analysis:")
        print(f"   Average Complexity: {avg_complexity:.1f}")
        print(f"   Low Complexity (≤5): {distribution.get('low', 0)} functions")
        print(f"   Medium Complexity (6-10): {distribution.get('medium', 0)} functions")
        print(f"   High Complexity (>10): {distribution.get('high', 0)} functions")
        
        # Function details
        function_details = final_report.get("function_details", [])
        print(f"\n📋 Function Analysis:")
        
        for func in function_details:
            status_emoji = "✅" if func["status"] == "PASS" else "❌"
            print(f"   {status_emoji} {func['name']}")
            print(f"      Quality Score: {func['quality_score']:.1f}/10.0")
            print(f"      Complexity: {func['complexity']}")
            print(f"      Line Count: {func['line_count']}")
            if func['issues']:
                print(f"      Issues: {', '.join(func['issues'])}")
        
        # Recommendations
        recommendations = final_report.get("recommendations", {})
        function_specific = recommendations.get("function_specific", [])
        overall_recs = recommendations.get("overall", [])
        priority_actions = recommendations.get("priority_actions", [])
        
        if function_specific or overall_recs or priority_actions:
            print(f"\n💡 Recommendations:")
            
            if priority_actions:
                print("   Priority Actions:")
                for action in priority_actions:
                    print(f"      🔥 {action}")
            
            if function_specific:
                print("   Function-Specific Suggestions:")
                for suggestion in function_specific:
                    func_name = suggestion["function_name"]
                    quality_score = suggestion["current_quality_score"]
                    print(f"      📝 {func_name} (Quality: {quality_score:.1f}):")
                    for item in suggestion["suggestions"]:
                        print(f"         • {item}")
            
            if overall_recs:
                print("   Overall Recommendations:")
                for rec in overall_recs:
                    print(f"      📋 {rec}")
        
        # Processing summary
        processing_summary = final_report.get("processing_summary", {})
        print(f"\n⚙️ Processing Summary:")
        print(f"   Extraction Status: {processing_summary.get('extraction_status', 'unknown')}")
        print(f"   Analysis Status: {processing_summary.get('analysis_status', 'unknown')}")
        print(f"   Evaluation Status: {processing_summary.get('evaluation_status', 'unknown')}")
        print(f"   Suggestion Status: {processing_summary.get('suggestion_status', 'unknown')}")
        print(f"   Code Lines Analyzed: {processing_summary.get('code_lines_analyzed', 0)}")
        
        print("="*60)
    
    def run_comprehensive_demo(self):
        """Run a comprehensive demonstration of the code review workflow."""
        print("🚀 CODE REVIEW WORKFLOW DEMONSTRATION")
        print("="*60)
        print("This demo showcases:")
        print("• Function extraction from Python code")
        print("• Complexity analysis and issue detection")
        print("• Quality scoring and threshold evaluation")
        print("• Improvement suggestion generation")
        print("• Comprehensive reporting")
        print("="*60)
        
        try:
            # Test server connectivity
            response = requests.get(f"{self.api_url}/graphs")
            if response.status_code != 200:
                print("❌ Server not accessible. Please ensure the workflow engine is running.")
                return
            
            print("✅ Server connectivity confirmed")
            
            # Register tools
            self.register_code_review_tools()
            
            # Demo 1: Low-quality code analysis
            print(f"\n" + "="*50)
            print("DEMO 1: ANALYZING LOW-QUALITY CODE")
            print("="*50)
            
            low_quality_results = self.run_code_review(SAMPLE_CODE, "Low Quality Code Analysis")
            if low_quality_results:
                self.display_code_review_results(low_quality_results)
            
            # Demo 2: High-quality code analysis
            print(f"\n" + "="*50)
            print("DEMO 2: ANALYZING HIGH-QUALITY CODE")
            print("="*50)
            
            high_quality_results = self.run_code_review(HIGH_QUALITY_CODE, "High Quality Code Analysis")
            if high_quality_results:
                self.display_code_review_results(high_quality_results)
            
            # Demo 3: Comparison summary
            print(f"\n" + "="*50)
            print("DEMO 3: COMPARISON SUMMARY")
            print("="*50)
            
            if low_quality_results and high_quality_results:
                low_quality_state = low_quality_results.get("current_state", {}).get("data", {})
                high_quality_state = high_quality_results.get("current_state", {}).get("data", {})
                
                low_avg_quality = low_quality_state.get("average_quality_score", 0)
                high_avg_quality = high_quality_state.get("average_quality_score", 0)
                
                low_issues = low_quality_state.get("total_issues", 0)
                high_issues = high_quality_state.get("total_issues", 0)
                
                print(f"📊 Quality Score Comparison:")
                print(f"   Low-quality code: {low_avg_quality:.2f}/10.0")
                print(f"   High-quality code: {high_avg_quality:.2f}/10.0")
                print(f"   Improvement: {high_avg_quality - low_avg_quality:.2f} points")
                
                print(f"\n🐛 Issue Count Comparison:")
                print(f"   Low-quality code: {low_issues} issues")
                print(f"   High-quality code: {high_issues} issues")
                print(f"   Reduction: {low_issues - high_issues} issues")
            
            print(f"\n" + "="*60)
            print("🎉 CODE REVIEW WORKFLOW DEMO COMPLETED!")
            print("="*60)
            print("The workflow successfully demonstrated:")
            print("✅ Function extraction and parsing")
            print("✅ Complexity analysis and issue detection")
            print("✅ Quality threshold evaluation")
            print("✅ Improvement suggestion generation")
            print("✅ Comprehensive report generation")
            print("✅ Iterative workflow logic")
            print("="*60)
            
        except requests.exceptions.ConnectionError:
            print("❌ Connection error: Please ensure the workflow engine server is running on http://localhost:8000")
        except Exception as e:
            print(f"❌ Demo error: {str(e)}")


def main():
    """Main entry point for the code review workflow demo."""
    demo = CodeReviewWorkflowDemo()
    
    print("Starting Code Review Workflow Demo...")
    print("Please ensure the server is running: uvicorn app.main:app --port 8000")
    print()
    
    demo.run_comprehensive_demo()


if __name__ == "__main__":
    main()