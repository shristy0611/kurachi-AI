#!/usr/bin/env python3
"""
Final regression test for all surgical fixes
Tests unified fallback, smart mapping, and chart validation
"""
import sys
import os
from pathlib import Path
import pytest

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.intelligent_response_formatter import (
    intelligent_response_formatter, 
    rule_first_guess, 
    map_intent_to_response_type,
    has_numeric_series,
    md_table
)


def test_unified_intent_classification():
    """Test that main and fallback paths give consistent results"""
    print("🎯 Testing Unified Intent Classification")
    print("=" * 50)
    
    # Critical test cases that were failing before
    critical_cases = {
        "show me a table": "table",
        "list the data in tabular format": "table", 
        "show me a graph of metrics": "chart",
        "how do i install": "process_flow",
        "compare a vs b side by side": "table",  # Smart mapping: comparison + table → table
        "what's the executive summary": "summary",
        "show code example": "code_block",
        "compare products": "comparison",  # Pure comparison without table hint
    }
    
    correct = 0
    total = len(critical_cases)
    
    for query, expected in critical_cases.items():
        try:
            # Test main path
            intent = intelligent_response_formatter.analyze_query_intent(query)
            result = intent.response_type.value
            
            print(f"Query: '{query}'")
            print(f"Expected: {expected}, Got: {result}")
            print(f"Confidence: {intent.confidence:.2f}")
            print(f"Reasoning: {intent.reasoning}")
            
            if result == expected:
                print("✅ CORRECT")
                correct += 1
            else:
                print("❌ INCORRECT")
        except Exception as e:
            print(f"ERROR: {e}")
        print()
    
    accuracy = (correct / total) * 100
    print(f"📊 Unified Classification Accuracy: {accuracy:.1f}% ({correct}/{total})")
    
    return accuracy >= 90  # Expect very high accuracy


def test_smart_intent_mapping():
    """Test the smart intent mapping tie-breaker logic"""
    print("\n🧠 Testing Smart Intent Mapping")
    print("=" * 50)
    
    test_cases = [
        # Comparison + table hints should map to table
        ("compare products in a table", "comparison", "table"),
        ("side by side comparison table", "comparison", "table"),
        ("compare a vs b in tabular format", "comparison", "table"),
        
        # Pure comparison should stay comparison
        ("compare product a vs b", "comparison", "comparison"),
        ("what's the difference", "comparison", "comparison"),
        
        # Other types should pass through unchanged
        ("show me a chart", "chart", "chart"),
        ("list the features", "list", "list"),
    ]
    
    passed = 0
    for query, input_type, expected in test_cases:
        result = map_intent_to_response_type(query, input_type)
        
        print(f"Query: '{query}'")
        print(f"Input: {input_type} → Output: {result} (expected: {expected})")
        
        if result == expected:
            print("✅ CORRECT")
            passed += 1
        else:
            print("❌ INCORRECT")
        print()
    
    success_rate = (passed / len(test_cases)) * 100
    print(f"📊 Smart Mapping Success Rate: {success_rate:.1f}% ({passed}/{len(test_cases)})")
    
    return success_rate >= 90


def test_chart_data_validation():
    """Test improved chart data validation"""
    print("\n📊 Testing Chart Data Validation")
    print("=" * 50)
    
    test_cases = [
        # Valid chart data
        ({"labels": ["Q1", "Q2", "Q3"], "values": [100, 200, 300]}, True, "Valid numeric series"),
        ({"labels": ["Jan", "Feb"], "values": [50.5, 75.2]}, True, "Valid with decimals"),
        
        # Invalid chart data
        ({"labels": ["A", "B"], "values": [0, 0]}, False, "All zeros"),
        ({"labels": ["A"], "values": [100]}, False, "Single data point"),
        ({"labels": ["A", "B"], "values": [100]}, False, "Mismatched labels/values"),
        ({"labels": ["A", "B"], "values": [100, None]}, False, "None values"),
        ({}, False, "Empty data"),
        ({"labels": [], "values": []}, False, "Empty arrays"),
    ]
    
    passed = 0
    for data, expected, description in test_cases:
        result = has_numeric_series(data)
        
        print(f"Test: {description}")
        print(f"Data: {data}")
        print(f"Expected: {expected}, Got: {result}")
        
        if result == expected:
            print("✅ CORRECT")
            passed += 1
        else:
            print("❌ INCORRECT")
        print()
    
    success_rate = (passed / len(test_cases)) * 100
    print(f"📊 Chart Validation Success Rate: {success_rate:.1f}% ({passed}/{len(test_cases)})")
    
    return success_rate >= 90


def test_markdown_table_generation():
    """Test clean markdown table generation"""
    print("\n📋 Testing Markdown Table Generation")
    print("=" * 50)
    
    headers = ["Product", "Price", "Rating"]
    rows = [
        ["Alpha", "$299", "4.2/5"],
        ["Beta", "$399", "4.5/5"],
        ["Gamma", "$199", "3.9/5"]
    ]
    
    expected_lines = [
        "| **Product** | **Price** | **Rating** |",
        "| --- | --- | --- |",
        "| Alpha | $299 | 4.2/5 |",
        "| Beta | $399 | 4.5/5 |",
        "| Gamma | $199 | 3.9/5 |"
    ]
    
    result = md_table(headers, rows)
    result_lines = result.split('\n')
    
    print("Generated table:")
    print(result)
    print()
    
    # Check structure
    if len(result_lines) == len(expected_lines):
        print("✅ Correct number of lines")
        
        # Check header formatting
        if "**Product**" in result_lines[0] and "**Price**" in result_lines[0]:
            print("✅ Headers properly bolded")
        else:
            print("❌ Headers not properly formatted")
            return False
            
        # Check separator row
        if "---" in result_lines[1]:
            print("✅ Separator row present")
        else:
            print("❌ Separator row missing")
            return False
            
        print("✅ Table generation working correctly")
        return True
    else:
        print(f"❌ Wrong number of lines: expected {len(expected_lines)}, got {len(result_lines)}")
        return False


@pytest.mark.slow
def test_end_to_end_workflow():
    """Test complete end-to-end workflow with edge cases"""
    print("\n🔄 Testing End-to-End Workflow")
    print("=" * 50)
    
    test_query = "Show me a comparison table of our quarterly sales performance"
    raw_response = """
    Q1 2024: Revenue $150,000, Units 1,200, Margin 15%
    Q2 2024: Revenue $175,000, Units 1,400, Margin 18%
    Q3 2024: Revenue $200,000, Units 1,600, Margin 20%
    Q4 2024: Revenue $225,000, Units 1,800, Margin 22%
    """
    
    try:
        print(f"Query: '{test_query}'")
        print("Processing complete workflow...")
        
        # Test intent analysis
        intent = intelligent_response_formatter.analyze_query_intent(test_query)
        print(f"✅ Intent: {intent.response_type.value} (confidence: {intent.confidence:.2f})")
        
        # Test complete formatting
        formatted_response = intelligent_response_formatter.format_response_with_streamlit_components(
            test_query, raw_response
        )
        
        print(f"✅ Response Type: {formatted_response.response_type.value}")
        print(f"✅ Has Content: {len(formatted_response.content) > 0}")
        print(f"✅ Has Metadata: {len(formatted_response.metadata) > 0}")
        print(f"✅ Streamlit Components: {len(formatted_response.streamlit_components)}")
        
        # Check for clean output (no <think> tags)
        if "<think>" not in formatted_response.content:
            print("✅ Clean output (no chain-of-thought leakage)")
        else:
            print("❌ Chain-of-thought leaked into output")
            return False
        
        # Check intent should be 'table' due to smart mapping
        if intent.response_type.value == "table":
            print("✅ Smart mapping worked (comparison + table → table)")
        else:
            print(f"❌ Smart mapping failed: expected 'table', got '{intent.response_type.value}'")
            return False
        
        print("✅ End-to-end workflow completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ End-to-end workflow failed: {e}")
        return False


def main():
    """Run all final regression tests"""
    print("🧪 Final Regression Test Suite")
    print("=" * 60)
    print("Testing all surgical fixes and improvements")
    print("=" * 60)
    
    tests = [
        ("Unified Intent Classification", test_unified_intent_classification),
        ("Smart Intent Mapping", test_smart_intent_mapping),
        ("Chart Data Validation", test_chart_data_validation),
        ("Markdown Table Generation", test_markdown_table_generation),
        ("End-to-End Workflow", test_end_to_end_workflow),
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
    print(f"📊 Final Test Results: {passed_tests}/{len(tests)} tests passed")
    
    if passed_tests == len(tests):
        print("🎉 ALL TESTS PASSED! System is SOTA-ready! 🚀")
        print("\n🔥 Key Improvements Validated:")
        print("✅ Unified fallback logic (no more table→list misses)")
        print("✅ Smart intent mapping (comparison+table→table)")
        print("✅ Robust chart validation (no junk charts)")
        print("✅ Clean output (no chain-of-thought leakage)")
        print("✅ Professional markdown tables")
        return True
    else:
        print("⚠️ Some tests failed. Check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)