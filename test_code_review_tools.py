#!/usr/bin/env python3
"""Test script for code review tools."""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.tools.code_review_tools import (
    extract_functions,
    analyze_function_complexity,
    evaluate_quality_threshold,
    generate_improvement_suggestions,
    generate_final_report
)

# Test code with quality issues
TEST_CODE = '''
def bad_function(a, b, c, d, e, f, g):
    if a > 0:
        if b > 0:
            if c > 0:
                if d > 0:
                    if e > 0:
                        return f + g
                    else:
                        return f - g
                else:
                    return f * g
            else:
                return f / g
        else:
            return f
    else:
        return g

def good_function(radius: float) -> float:
    """Calculate circle area."""
    import math
    return math.pi * radius ** 2
'''

def test_code_review_workflow():
    """Test the complete code review workflow."""
    print("🧪 Testing Code Review Workflow")
    print("=" * 50)
    
    # Step 1: Extract functions
    print("\n1. Testing function extraction...")
    state = {"code_input": TEST_CODE}
    result = extract_functions(state)
    
    print(f"   Extraction status: {result.get('extraction_status')}")
    print(f"   Functions found: {result.get('function_count')}")
    
    if result.get('extraction_status') == 'success':
        state.update(result)
        
        # Step 2: Analyze complexity
        print("\n2. Testing complexity analysis...")
        result = analyze_function_complexity(state)
        
        print(f"   Analysis status: {result.get('analysis_status')}")
        print(f"   Total issues: {result.get('total_issues')}")
        print(f"   Average quality: {result.get('average_quality_score', 0):.2f}")
        
        if result.get('analysis_status') == 'success':
            state.update(result)
            
            # Step 3: Evaluate threshold
            print("\n3. Testing threshold evaluation...")
            result = evaluate_quality_threshold(state, quality_threshold=7.0)
            
            print(f"   Evaluation status: {result.get('evaluation_status')}")
            print(f"   Threshold met: {result.get('threshold_met')}")
            print(f"   Functions below threshold: {result.get('functions_below_threshold')}")
            
            if result.get('evaluation_status') == 'success':
                state.update(result)
                
                # Step 4: Generate suggestions (if needed)
                if not result.get('threshold_met'):
                    print("\n4. Testing suggestion generation...")
                    result = generate_improvement_suggestions(state)
                    
                    print(f"   Suggestion status: {result.get('suggestion_status')}")
                    print(f"   Suggestions count: {result.get('suggestion_count')}")
                    
                    if result.get('suggestion_status') == 'success':
                        state.update(result)
                
                # Step 5: Generate final report
                print("\n5. Testing final report generation...")
                result = generate_final_report(state)
                
                print(f"   Report status: {result.get('report_status')}")
                print(f"   Overall status: {result.get('overall_status')}")
                
                if result.get('report_status') == 'success':
                    final_report = result.get('final_report', {})
                    
                    print("\n📊 Final Report Summary:")
                    quality_metrics = final_report.get('quality_metrics', {})
                    print(f"   Total functions: {quality_metrics.get('total_functions')}")
                    print(f"   Average quality: {quality_metrics.get('average_quality_score'):.2f}")
                    print(f"   Total issues: {quality_metrics.get('total_issues')}")
                    
                    recommendations = final_report.get('recommendations', {})
                    function_suggestions = recommendations.get('function_specific', [])
                    print(f"   Function suggestions: {len(function_suggestions)}")
                    
                    for suggestion in function_suggestions:
                        func_name = suggestion.get('function_name')
                        quality_score = suggestion.get('current_quality_score')
                        suggestions = suggestion.get('suggestions', [])
                        print(f"     {func_name} (Quality: {quality_score:.1f}): {len(suggestions)} suggestions")
    
    print("\n✅ Code review workflow test completed!")

if __name__ == "__main__":
    test_code_review_workflow()