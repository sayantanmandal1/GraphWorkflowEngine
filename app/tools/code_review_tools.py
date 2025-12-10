"""Code Review Workflow tools for analyzing Python code quality."""

import ast
import re
import math
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from ..core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FunctionInfo:
    """Information about a parsed function."""
    name: str
    line_start: int
    line_end: int
    source_code: str
    complexity: int
    issues: List[str]
    quality_score: float


@dataclass
class CodeAnalysisResult:
    """Result of code analysis."""
    functions: List[FunctionInfo]
    overall_quality_score: float
    total_issues: int
    suggestions: List[str]


def extract_functions(state: Dict[str, Any], context: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
    """
    Extract individual functions from Python code input for analysis.
    
    Args:
        state: Current workflow state containing 'code_input'
        context: Execution context (optional)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with extracted function information
        
    Validates: Requirements 5.1
    """
    logger.info("Starting function extraction from code input")
    
    code_input = state.get("code_input", "")
    if not code_input or not code_input.strip():
        logger.warning("No code input provided for function extraction")
        return {
            "functions": [],
            "extraction_error": "No code input provided",
            "extraction_status": "failed"
        }
    
    try:
        # Parse the code into an AST
        tree = ast.parse(code_input)
        
        # Extract function definitions
        functions = []
        lines = code_input.split('\n')
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Get function source code
                start_line = node.lineno - 1  # Convert to 0-based indexing
                end_line = node.end_lineno if hasattr(node, 'end_lineno') else start_line + 1
                
                # Extract source code for this function
                function_lines = lines[start_line:end_line]
                function_source = '\n'.join(function_lines)
                
                function_info = {
                    "name": node.name,
                    "line_start": start_line + 1,  # Convert back to 1-based for display
                    "line_end": end_line,
                    "source_code": function_source
                }
                
                functions.append(function_info)
                logger.debug(f"Extracted function: {node.name} (lines {start_line + 1}-{end_line})")
        
        logger.info(f"Successfully extracted {len(functions)} functions from code input")
        
        return {
            "functions": functions,
            "function_count": len(functions),
            "extraction_status": "success",
            "code_lines": len(lines)
        }
        
    except SyntaxError as e:
        error_msg = f"Syntax error in code input: {str(e)}"
        logger.error(error_msg)
        return {
            "functions": [],
            "extraction_error": error_msg,
            "extraction_status": "failed"
        }
    except Exception as e:
        error_msg = f"Error extracting functions: {str(e)}"
        logger.error(error_msg)
        return {
            "functions": [],
            "extraction_error": error_msg,
            "extraction_status": "failed"
        }


def analyze_function_complexity(state: Dict[str, Any], context: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
    """
    Calculate complexity metrics and identify potential issues for extracted functions.
    
    Args:
        state: Current workflow state containing extracted functions
        context: Execution context (optional)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with complexity analysis results
        
    Validates: Requirements 5.2
    """
    logger.info("Starting complexity analysis of extracted functions")
    
    functions = state.get("functions", [])
    if not functions:
        logger.warning("No functions available for complexity analysis")
        return {
            "analyzed_functions": [],
            "analysis_error": "No functions to analyze",
            "analysis_status": "failed"
        }
    
    try:
        analyzed_functions = []
        
        for func_info in functions:
            source_code = func_info.get("source_code", "")
            
            if not source_code.strip():
                logger.warning(f"No source code for function {func_info.get('name', 'unknown')}")
                continue
            
            try:
                # Parse the function source code to get AST node
                func_ast = ast.parse(source_code)
                # Find the function definition node
                ast_node = None
                for node in ast.walk(func_ast):
                    if isinstance(node, ast.FunctionDef):
                        ast_node = node
                        break
                
                if not ast_node:
                    logger.warning(f"Could not parse function {func_info.get('name', 'unknown')}")
                    continue
                
                # Calculate cyclomatic complexity
                complexity = _calculate_cyclomatic_complexity(ast_node)
                
                # Identify issues
                issues = _identify_code_issues(ast_node, source_code)
                
            except Exception as e:
                logger.error(f"Error parsing function {func_info.get('name', 'unknown')}: {str(e)}")
                # Use default values if parsing fails
                complexity = 1
                issues = [f"Could not analyze function: {str(e)}"]
            
            # Calculate quality score based on complexity and issues
            quality_score = _calculate_quality_score(complexity, len(issues), len(source_code.split('\n')))
            
            analyzed_function = {
                "name": func_info["name"],
                "line_start": func_info["line_start"],
                "line_end": func_info["line_end"],
                "source_code": source_code,
                "complexity": complexity,
                "issues": issues,
                "quality_score": quality_score,
                "line_count": len(source_code.split('\n'))
            }
            
            analyzed_functions.append(analyzed_function)
            logger.debug(f"Analyzed function {func_info['name']}: complexity={complexity}, issues={len(issues)}, quality={quality_score:.2f}")
        
        # Calculate overall metrics
        total_issues = sum(len(func["issues"]) for func in analyzed_functions)
        avg_quality = sum(func["quality_score"] for func in analyzed_functions) / len(analyzed_functions) if analyzed_functions else 0
        
        logger.info(f"Completed complexity analysis: {len(analyzed_functions)} functions, {total_issues} total issues, avg quality: {avg_quality:.2f}")
        
        return {
            "analyzed_functions": analyzed_functions,
            "total_functions": len(analyzed_functions),
            "total_issues": total_issues,
            "average_quality_score": avg_quality,
            "analysis_status": "success"
        }
        
    except Exception as e:
        error_msg = f"Error during complexity analysis: {str(e)}"
        logger.error(error_msg)
        return {
            "analyzed_functions": [],
            "analysis_error": error_msg,
            "analysis_status": "failed"
        }


def evaluate_quality_threshold(state: Dict[str, Any], context: Optional[Any] = None, 
                             quality_threshold: float = 7.0, **kwargs) -> Dict[str, Any]:
    """
    Evaluate quality scores against threshold and determine next action.
    
    Args:
        state: Current workflow state containing analyzed functions
        context: Execution context (optional)
        quality_threshold: Minimum quality score threshold (default: 7.0)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with threshold evaluation results
        
    Validates: Requirements 5.3, 5.4
    """
    logger.info(f"Evaluating quality threshold: {quality_threshold}")
    
    analyzed_functions = state.get("analyzed_functions", [])
    if not analyzed_functions:
        logger.warning("No analyzed functions available for threshold evaluation")
        return {
            "threshold_met": False,
            "evaluation_error": "No analyzed functions available",
            "evaluation_status": "failed"
        }
    
    try:
        # Calculate overall quality metrics
        avg_quality = state.get("average_quality_score", 0.0)
        total_issues = state.get("total_issues", 0)
        
        # Determine if threshold is met
        threshold_met = avg_quality >= quality_threshold and total_issues == 0
        
        # Count functions below threshold
        functions_below_threshold = [
            func for func in analyzed_functions 
            if func["quality_score"] < quality_threshold or len(func["issues"]) > 0
        ]
        
        evaluation_result = {
            "threshold_met": threshold_met,
            "quality_threshold": quality_threshold,
            "average_quality_score": avg_quality,
            "total_issues": total_issues,
            "functions_below_threshold": len(functions_below_threshold),
            "functions_below_threshold_details": functions_below_threshold,
            "evaluation_status": "success",
            "next_action": "complete" if threshold_met else "improve"
        }
        
        if threshold_met:
            logger.info(f"Quality threshold met: avg_quality={avg_quality:.2f} >= {quality_threshold}, issues={total_issues}")
        else:
            logger.info(f"Quality threshold not met: avg_quality={avg_quality:.2f} < {quality_threshold} or issues={total_issues} > 0")
        
        return evaluation_result
        
    except Exception as e:
        error_msg = f"Error during threshold evaluation: {str(e)}"
        logger.error(error_msg)
        return {
            "threshold_met": False,
            "evaluation_error": error_msg,
            "evaluation_status": "failed"
        }


def generate_improvement_suggestions(state: Dict[str, Any], context: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
    """
    Generate improvement suggestions for functions that don't meet quality standards.
    
    Args:
        state: Current workflow state containing analyzed functions
        context: Execution context (optional)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with improvement suggestions
        
    Validates: Requirements 5.3
    """
    logger.info("Generating improvement suggestions for low-quality functions")
    
    analyzed_functions = state.get("analyzed_functions", [])
    functions_below_threshold = state.get("functions_below_threshold_details", [])
    
    if not functions_below_threshold:
        logger.info("No functions below threshold - no suggestions needed")
        return {
            "suggestions": [],
            "suggestion_count": 0,
            "suggestion_status": "success"
        }
    
    try:
        suggestions = []
        
        for func in functions_below_threshold:
            func_name = func["name"]
            complexity = func["complexity"]
            issues = func["issues"]
            quality_score = func["quality_score"]
            
            func_suggestions = []
            
            # Complexity-based suggestions
            if complexity > 10:
                func_suggestions.append(f"High complexity ({complexity}): Consider breaking down into smaller functions")
            elif complexity > 5:
                func_suggestions.append(f"Moderate complexity ({complexity}): Review for simplification opportunities")
            
            # Issue-based suggestions
            for issue in issues:
                if "too long" in issue.lower():
                    func_suggestions.append("Function is too long: Consider splitting into smaller, focused functions")
                elif "no docstring" in issue.lower():
                    func_suggestions.append("Add docstring: Document function purpose, parameters, and return value")
                elif "too many parameters" in issue.lower():
                    func_suggestions.append("Too many parameters: Consider using a configuration object or reducing parameters")
                elif "nested" in issue.lower():
                    func_suggestions.append("Deeply nested code: Reduce nesting levels using early returns or helper functions")
                elif "naming" in issue.lower():
                    func_suggestions.append("Improve naming: Use more descriptive variable and function names")
            
            # Quality score based suggestions
            if quality_score < 5.0:
                func_suggestions.append("Low quality score: Comprehensive refactoring recommended")
            elif quality_score < 7.0:
                func_suggestions.append("Below threshold: Focus on addressing identified issues")
            
            if func_suggestions:
                suggestion_entry = {
                    "function_name": func_name,
                    "current_quality_score": quality_score,
                    "complexity": complexity,
                    "issue_count": len(issues),
                    "suggestions": func_suggestions
                }
                suggestions.append(suggestion_entry)
        
        # Generate overall suggestions
        overall_suggestions = []
        total_issues = state.get("total_issues", 0)
        avg_quality = state.get("average_quality_score", 0.0)
        
        if total_issues > 10:
            overall_suggestions.append("High issue count: Consider establishing coding standards and review processes")
        if avg_quality < 6.0:
            overall_suggestions.append("Low average quality: Implement systematic code quality improvements")
        
        logger.info(f"Generated {len(suggestions)} function-specific suggestions and {len(overall_suggestions)} overall suggestions")
        
        return {
            "function_suggestions": suggestions,
            "overall_suggestions": overall_suggestions,
            "suggestion_count": len(suggestions),
            "total_suggestion_items": sum(len(s["suggestions"]) for s in suggestions) + len(overall_suggestions),
            "suggestion_status": "success"
        }
        
    except Exception as e:
        error_msg = f"Error generating improvement suggestions: {str(e)}"
        logger.error(error_msg)
        return {
            "suggestions": [],
            "suggestion_error": error_msg,
            "suggestion_status": "failed"
        }


def generate_final_report(state: Dict[str, Any], context: Optional[Any] = None, **kwargs) -> Dict[str, Any]:
    """
    Generate a comprehensive final report of all analysis results.
    
    Args:
        state: Current workflow state containing all analysis data
        context: Execution context (optional)
        **kwargs: Additional parameters
        
    Returns:
        Dictionary with comprehensive analysis report
        
    Validates: Requirements 5.5
    """
    logger.info("Generating comprehensive code review report")
    
    try:
        # Gather all analysis data
        analyzed_functions = state.get("analyzed_functions", [])
        total_issues = state.get("total_issues", 0)
        avg_quality = state.get("average_quality_score", 0.0)
        threshold_met = state.get("threshold_met", False)
        quality_threshold = state.get("quality_threshold", 7.0)
        function_suggestions = state.get("function_suggestions", [])
        overall_suggestions = state.get("overall_suggestions", [])
        
        # Calculate additional metrics
        total_functions = len(analyzed_functions)
        functions_above_threshold = sum(1 for func in analyzed_functions if func["quality_score"] >= quality_threshold)
        functions_below_threshold = total_functions - functions_above_threshold
        
        complexity_distribution = {}
        for func in analyzed_functions:
            complexity = func["complexity"]
            if complexity <= 5:
                complexity_distribution["low"] = complexity_distribution.get("low", 0) + 1
            elif complexity <= 10:
                complexity_distribution["medium"] = complexity_distribution.get("medium", 0) + 1
            else:
                complexity_distribution["high"] = complexity_distribution.get("high", 0) + 1
        
        # Generate summary
        if threshold_met:
            overall_status = "PASSED"
            summary = f"Code review completed successfully. All {total_functions} functions meet quality standards."
        else:
            overall_status = "NEEDS_IMPROVEMENT"
            summary = f"Code review identified areas for improvement. {functions_below_threshold} of {total_functions} functions need attention."
        
        # Create comprehensive report
        report = {
            "report_timestamp": context.state.metadata.get("current_timestamp") if context else "unknown",
            "overall_status": overall_status,
            "summary": summary,
            
            # Quality metrics
            "quality_metrics": {
                "total_functions": total_functions,
                "average_quality_score": round(avg_quality, 2),
                "quality_threshold": quality_threshold,
                "threshold_met": threshold_met,
                "functions_above_threshold": functions_above_threshold,
                "functions_below_threshold": functions_below_threshold,
                "total_issues": total_issues
            },
            
            # Complexity analysis
            "complexity_analysis": {
                "distribution": complexity_distribution,
                "average_complexity": round(sum(func["complexity"] for func in analyzed_functions) / total_functions, 2) if total_functions > 0 else 0
            },
            
            # Function details
            "function_details": [
                {
                    "name": func["name"],
                    "quality_score": func["quality_score"],
                    "complexity": func["complexity"],
                    "issues": func["issues"],
                    "line_count": func["line_count"],
                    "status": "PASS" if func["quality_score"] >= quality_threshold and len(func["issues"]) == 0 else "NEEDS_IMPROVEMENT"
                }
                for func in analyzed_functions
            ],
            
            # Improvement recommendations
            "recommendations": {
                "function_specific": function_suggestions,
                "overall": overall_suggestions,
                "priority_actions": _generate_priority_actions(analyzed_functions, function_suggestions)
            },
            
            # Processing summary
            "processing_summary": {
                "extraction_status": state.get("extraction_status", "unknown"),
                "analysis_status": state.get("analysis_status", "unknown"),
                "evaluation_status": state.get("evaluation_status", "unknown"),
                "suggestion_status": state.get("suggestion_status", "unknown"),
                "code_lines_analyzed": state.get("code_lines", 0)
            }
        }
        
        logger.info(f"Generated comprehensive report: {overall_status}, {total_functions} functions, {total_issues} issues")
        
        return {
            "final_report": report,
            "report_status": "success",
            "report_summary": summary,
            "overall_status": overall_status
        }
        
    except Exception as e:
        error_msg = f"Error generating final report: {str(e)}"
        logger.error(error_msg)
        return {
            "final_report": {},
            "report_error": error_msg,
            "report_status": "failed"
        }


def _calculate_cyclomatic_complexity(node: ast.AST) -> int:
    """Calculate cyclomatic complexity of an AST node."""
    complexity = 1  # Base complexity
    
    for child in ast.walk(node):
        # Decision points that increase complexity
        if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
            complexity += 1
        elif isinstance(child, ast.Try):
            complexity += len(child.handlers)  # Each except handler adds complexity
        elif isinstance(child, ast.BoolOp):
            complexity += len(child.values) - 1  # And/Or operations
        elif isinstance(child, ast.ListComp):
            complexity += 1  # List comprehensions
        elif isinstance(child, ast.DictComp):
            complexity += 1  # Dict comprehensions
        elif isinstance(child, ast.SetComp):
            complexity += 1  # Set comprehensions
        elif isinstance(child, ast.GeneratorExp):
            complexity += 1  # Generator expressions
    
    return complexity


def _identify_code_issues(node: ast.AST, source_code: str) -> List[str]:
    """Identify potential code quality issues."""
    issues = []
    lines = source_code.split('\n')
    
    # Check function length
    if len(lines) > 50:
        issues.append("Function is too long (>50 lines)")
    elif len(lines) > 30:
        issues.append("Function is moderately long (>30 lines)")
    
    # Check for docstring
    if isinstance(node, ast.FunctionDef):
        if not ast.get_docstring(node):
            issues.append("Function has no docstring")
        
        # Check parameter count
        if len(node.args.args) > 5:
            issues.append("Function has too many parameters (>5)")
        
        # Check for deeply nested code
        max_nesting = _calculate_max_nesting_level(node)
        if max_nesting > 4:
            issues.append(f"Deeply nested code (max nesting: {max_nesting})")
    
    # Check for long lines
    long_lines = [i + 1 for i, line in enumerate(lines) if len(line) > 100]
    if long_lines:
        issues.append(f"Long lines detected (>100 chars): lines {long_lines}")
    
    # Check for naming conventions
    if isinstance(node, ast.FunctionDef):
        if not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
            issues.append("Function name doesn't follow snake_case convention")
    
    return issues


def _calculate_max_nesting_level(node: ast.AST) -> int:
    """Calculate maximum nesting level in an AST node."""
    def get_nesting_level(node, current_level=0):
        max_level = current_level
        
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try)):
                child_max = get_nesting_level(child, current_level + 1)
                max_level = max(max_level, child_max)
            else:
                child_max = get_nesting_level(child, current_level)
                max_level = max(max_level, child_max)
        
        return max_level
    
    return get_nesting_level(node)


def _calculate_quality_score(complexity: int, issue_count: int, line_count: int) -> float:
    """Calculate a quality score based on various metrics."""
    # Base score
    score = 10.0
    
    # Penalize high complexity
    if complexity > 10:
        score -= 3.0
    elif complexity > 5:
        score -= 1.5
    
    # Penalize issues
    score -= issue_count * 0.5
    
    # Penalize very long functions
    if line_count > 50:
        score -= 2.0
    elif line_count > 30:
        score -= 1.0
    
    # Ensure score is not negative
    return max(0.0, score)


def _generate_priority_actions(analyzed_functions: List[Dict], function_suggestions: List[Dict]) -> List[str]:
    """Generate priority actions based on analysis results."""
    priority_actions = []
    
    # Find functions with the lowest quality scores
    if analyzed_functions:
        lowest_quality_func = min(analyzed_functions, key=lambda f: f["quality_score"])
        if lowest_quality_func["quality_score"] < 5.0:
            priority_actions.append(f"URGENT: Refactor function '{lowest_quality_func['name']}' (quality score: {lowest_quality_func['quality_score']:.1f})")
    
    # Find functions with highest complexity
    high_complexity_funcs = [f for f in analyzed_functions if f["complexity"] > 10]
    if high_complexity_funcs:
        priority_actions.append(f"HIGH: Reduce complexity in {len(high_complexity_funcs)} function(s) with complexity > 10")
    
    # Count functions without docstrings
    no_docstring_count = sum(1 for f in analyzed_functions if "no docstring" in str(f["issues"]).lower())
    if no_docstring_count > 0:
        priority_actions.append(f"MEDIUM: Add docstrings to {no_docstring_count} function(s)")
    
    return priority_actions