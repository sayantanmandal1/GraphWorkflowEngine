"""
Workflow Engine Code Review Demo
Demonstrates the workflow engine's ability to analyze and review code quality.
"""

import asyncio
import logging
from pathlib import Path


def read_file_content(file_path: str) -> str:
    """Read content from a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def analyze_code_quality(file_path: str) -> dict:
    """Simple code quality analysis."""
    content = read_file_content(file_path)
    if not content:
        return {"issues": [], "score": 0}
    
    issues = []
    lines = content.split('\n')
    
    # Check for common issues
    has_docstrings = '"""' in content or "'''" in content
    has_type_hints = ': ' in content and '->' in content
    has_error_handling = 'try:' in content or 'except' in content
    has_logging = 'logging' in content or 'logger' in content
    
    # Count functions and classes
    function_count = content.count('def ')
    class_count = content.count('class ')
    
    # Identify issues
    if not has_docstrings and function_count > 0:
        issues.append("Missing docstrings")
    
    if not has_type_hints and function_count > 0:
        issues.append("Missing type hints")
    
    if not has_error_handling and function_count > 0:
        issues.append("No error handling")
    
    if not has_logging and function_count > 0:
        issues.append("No logging")
    
    # Check for long lines
    long_lines = [i+1 for i, line in enumerate(lines) if len(line) > 100]
    if long_lines:
        issues.append(f"Long lines: {len(long_lines)} lines > 100 chars")
    
    # Check for naming conventions
    if any(line.strip().startswith('class ') and not line.split()[1][0].isupper() for line in lines):
        issues.append("Class names should be PascalCase")
    
    # Calculate score (10 - number of issue types)
    score = max(0, 10 - len(issues))
    
    return {
        "issues": issues,
        "score": score,
        "function_count": function_count,
        "class_count": class_count,
        "line_count": len(lines)
    }


def check_code_style(file_path: str) -> dict:
    """Check code style issues."""
    content = read_file_content(file_path)
    if not content:
        return {"issues": []}
    
    issues = []
    lines = content.split('\n')
    
    # Check indentation
    for i, line in enumerate(lines, 1):
        if line.strip():
            indent_level = len(line) - len(line.lstrip())
            if indent_level % 4 != 0:
                issues.append(f"Line {i}: Inconsistent indentation (not multiple of 4)")
                break
    
    # Check for trailing whitespace
    trailing_whitespace = [i+1 for i, line in enumerate(lines) if line.rstrip() != line]
    if trailing_whitespace:
        issues.append(f"Trailing whitespace on {len(trailing_whitespace)} lines")
    
    # Check for multiple blank lines
    blank_line_groups = 0
    in_blank_group = False
    for line in lines:
        if not line.strip():
            if not in_blank_group:
                blank_line_groups += 1
                in_blank_group = True
        else:
            in_blank_group = False
    
    return {"issues": issues}


def security_scan(file_path: str) -> dict:
    """Basic security scan."""
    content = read_file_content(file_path)
    if not content:
        return {"issues": []}
    
    issues = []
    
    # Check for potential security issues
    if 'eval(' in content:
        issues.append("Use of eval() function - potential security risk")
    
    if 'exec(' in content:
        issues.append("Use of exec() function - potential security risk")
    
    if 'input(' in content and 'int(' in content:
        issues.append("Direct conversion of user input - potential security risk")
    
    if 'pickle.load' in content:
        issues.append("Use of pickle.load - potential security risk")
    
    return {"issues": issues}


async def run_code_review_workflow():
    """
    Demonstrates the workflow engine performing code review on both perfect and faulty code.
    """
    print("=== Workflow Engine Code Review Demo ===\n")
    
    # Define the code review workflow
    workflow_steps = [
        {
            "id": "analyze_perfect_code",
            "tool": "analyze_code_quality",
            "params": {"file_path": "perfect_code_example.py"},
            "description": "Analyzing perfect code example"
        },
        {
            "id": "style_check_perfect",
            "tool": "check_code_style", 
            "params": {"file_path": "perfect_code_example.py"},
            "description": "Style checking perfect code"
        },
        {
            "id": "security_scan_perfect",
            "tool": "security_scan",
            "params": {"file_path": "perfect_code_example.py"},
            "description": "Security scanning perfect code"
        },
        {
            "id": "analyze_faulty_code",
            "tool": "analyze_code_quality",
            "params": {"file_path": "faulty_code_example.py"},
            "description": "Analyzing faulty code example"
        },
        {
            "id": "style_check_faulty",
            "tool": "check_code_style",
            "params": {"file_path": "faulty_code_example.py"}, 
            "description": "Style checking faulty code"
        },
        {
            "id": "security_scan_faulty",
            "tool": "security_scan",
            "params": {"file_path": "faulty_code_example.py"},
            "description": "Security scanning faulty code"
        }
    ]
    
    # Tool mapping
    tools = {
        "analyze_code_quality": analyze_code_quality,
        "check_code_style": check_code_style,
        "security_scan": security_scan
    }
    
    results = {}
    
    # Execute workflow steps
    for step in workflow_steps:
        print(f"🔄 {step['description']}...")
        
        try:
            # Execute the tool
            tool_func = tools[step['tool']]
            result = tool_func(**step['params'])
            
            # Store result
            results[step['id']] = result
            
            # Display results
            print(f"✅ {step['description']} completed")
            if result.get('issues'):
                print(f"   Issues found: {len(result['issues'])}")
                for issue in result['issues'][:3]:  # Show first 3 issues
                    print(f"   - {issue}")
                if len(result['issues']) > 3:
                    print(f"   ... and {len(result['issues']) - 3} more issues")
            else:
                print("   No issues found")
            
            if 'score' in result:
                print(f"   Quality score: {result['score']}/10")
            print()
            
        except Exception as e:
            print(f"❌ Error in {step['description']}: {e}")
            results[step['id']] = {"error": str(e)}
            print()
    
    # Generate summary report
    print("=== Code Review Summary ===")
    
    # Perfect code summary
    perfect_results = [results.get(f"{tool}_perfect_code", {}) for tool in ["analyze", "style_check", "security_scan"]]
    perfect_issues = sum(len(r.get('issues', [])) for r in perfect_results)
    perfect_score = results.get('analyze_perfect_code', {}).get('score', 0)
    
    print(f"📋 Perfect Code Analysis:")
    print(f"   Total issues found: {perfect_issues}")
    print(f"   Quality score: {perfect_score}/10")
    print(f"   Code quality: {'Excellent' if perfect_issues == 0 else 'Good' if perfect_issues < 3 else 'Needs improvement'}")
    
    # Faulty code summary  
    faulty_results = [results.get(f"{tool}_faulty_code", {}) for tool in ["analyze", "style_check", "security_scan"]]
    faulty_issues = sum(len(r.get('issues', [])) for r in faulty_results)
    faulty_score = results.get('analyze_faulty_code', {}).get('score', 0)
    
    print(f"📋 Faulty Code Analysis:")
    print(f"   Total issues found: {faulty_issues}")
    print(f"   Quality score: {faulty_score}/10")
    print(f"   Code quality: {'Excellent' if faulty_issues == 0 else 'Good' if faulty_issues < 3 else 'Needs improvement'}")
    
    print(f"\n🎯 Workflow completed successfully!")
    print(f"   The workflow engine successfully identified {faulty_issues - perfect_issues} more issues in the faulty code.")
    print(f"   Quality score difference: {perfect_score - faulty_score} points")


if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Check if example files exist
    if not Path("perfect_code_example.py").exists():
        print("❌ perfect_code_example.py not found. Please ensure both example files exist.")
        exit(1)
        
    if not Path("faulty_code_example.py").exists():
        print("❌ faulty_code_example.py not found. Please ensure both example files exist.")
        exit(1)
    
    # Run the demo
    asyncio.run(run_code_review_workflow())