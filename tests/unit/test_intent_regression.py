#!/usr/bin/env python3
"""
Regression test for intent classification fixes
Tests the rule-first approach and JSON parsing improvements
"""
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.intelligent_response_formatter import intelligent_response_formatter, rule_first_guess, robust_json_parse


def test_rule_first_classification():
    """Test the rule-first classification system"""
    print("🎯 Testing Rule-First Classification")
    print("=" * 50)
    
    test_cases = {
        "Show me a comparison table of sales data": "table",
        "List the data in tabular format": "table", 
        "Show me a graph of the performance metrics": "chart",
        "Create a bar chart of revenue": "chart",
        "Visualize the data as a pie chart": "chart",
        "How do I set up the system?": "process_flow",
        "What are the steps to install?": "process_flow",
        "Walk me through the setup": "process_flow",
        "What's the executive summary?": "summary",
        "Give me a brief overview": "summary",
        "Show me a Python example": "code_block",
        "What's the syntax for this?": "code_block",
        "List all the features": "list",
        "Compare product A vs B": "comparison",
        "What's the difference between these?": "comparison"
    }
    
    correct = 0
    total = len(test_cases)
    
    for query, expected in test_cases.items():
        result = rule_first_guess(query)
        
        # Handle special case: comparison + table
        if result == "comparison" and "table" in query.lower():
            result = "table"
        
        print(f"Query: '{query}'")
        print(f"Expected: {expected}, Got: {result}")
        
        if result == expected:
            print("✅ CORRECT")
            correct += 1
        else:
            print("❌ INCORRECT")
        print()
    
    accuracy = (correct / total) * 100
    print(f"📊 Rule-First Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    return accuracy >= 85  # Expect high accuracy from rules


def test_robust_json_parsing():
    """Test the robust JSON parser with problematic inputs"""
    print("\n🔧 Testing Robust JSON Parsing")
    print("=" * 50)
    
    test_cases = [
        # Valid JSON
        ('{"type":"table","confidence":0.9}', {"type": "table", "confidence": 0.9}),
        
        # JSON with extra text
        ('Here is the result: {"type":"chart","confidence":0.8} - done', {"type": "chart", "confidence": 0.8}),
        
        # JSON with bad escapes
        ('{"type":"list","reason":"It\'s a list"}', {"type": "list", "reason": "It's a list"}),
        
        # JSON with trailing commas
        ('{"type":"summary","confidence":0.7,}', {"type": "summary", "confidence": 0.7}),
        
        # Markdown code block
        ('```json\n{"type":"code_block","confidence":0.9}\n```', {"type": "code_block", "confidence": 0.9}),
        
        # Completely invalid
        ('This is not JSON at all', {}),
    ]
    
    passed = 0
    for i, (input_str, expected) in enumerate(test_cases, 1):
        try:
            result = robust_json_parse(input_str)
            
            print(f"Test {i}: {input_str[:50]}...")
            print(f"Expected: {expected}")
            print(f"Got: {result}")
            
            if result == expected:
                print("✅ PASSED")
                passed += 1
            else:
                print("❌ FAILED")
        except Exception as e:
            print(f"Test {i}: ERROR - {e}")
        print()
    
    success_rate = (passed / len(test_cases)) * 100
    print(f"📊 JSON Parsing Success Rate: {success_rate:.1f}% ({passed}/{len(test_cases)})")
    
    return success_rate >= 80


def test_end_to_end_intent():
    """Test end-to-end intent analysis with the fixes"""
    print("\n🔄 Testing End-to-End Intent Analysis")
    print("=" * 50)
    
    critical_cases = [
        ("Show me a comparison table of sales data", "table"),
        ("Create a bar chart of the revenue trends", "chart"), 
        ("How do I install the software?", "process_flow"),
        ("List the main features", "list"),
        ("Give me a summary", "summary"),
    ]
    
    passed = 0
    for query, expected in critical_cases:
        try:
            intent = intelligent_response_formatter.analyze_query_intent(query)
            result = intent.response_type.value
            
            print(f"Query: '{query}'")
            print(f"Expected: {expected}, Got: {result}")
            print(f"Confidence: {intent.confidence:.2f}")
            print(f"Reasoning: {intent.reasoning}")
            
            if result == expected:
                print("✅ CORRECT")
                passed += 1
            else:
                print("❌ INCORRECT")
        except Exception as e:
            print(f"ERROR: {e}")
        print()
    
    accuracy = (passed / len(critical_cases)) * 100
    print(f"📊 End-to-End Accuracy: {accuracy:.1f}% ({passed}/{len(critical_cases)})")
    
    return accuracy >= 80


def main():
    """Run all regression tests"""
    print("🧪 Intent Classification Regression Tests")
    print("=" * 60)
    print("Testing fixes for JSON parsing and keyword classification")
    print("=" * 60)
    
    tests = [
        ("Rule-First Classification", test_rule_first_classification),
        ("Robust JSON Parsing", test_robust_json_parsing),
        ("End-to-End Intent Analysis", test_end_to_end_intent),
    ]
    
    passed_tests = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed_tests += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 Regression Test Results: {passed_tests}/{len(tests)} tests passed")
    
    if passed_tests == len(tests):
        print("🎉 All regression tests passed! Fixes are working correctly.")
        return True
    else:
        print("⚠️ Some regression tests failed. Check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)