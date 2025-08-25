#!/usr/bin/env python3
"""
Basic test for intelligent response formatting
Tests core functionality without extensive model calls
"""
import sys
import os
from pathlib import Path
import pytest

# Mark entire module as slow due to service initialization overhead
pytestmark = pytest.mark.slow

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from services.intelligent_response_formatter import intelligent_response_formatter, ResponseType


def test_basic_functionality():
    """Test basic functionality with simple queries"""
    print("🧪 Testing Basic Intelligent Response Formatting")
    print("=" * 60)
    
    # Test simple table request
    query = "Show me a table of sales data"
    raw_response = """
    Q1: $100,000
    Q2: $120,000
    Q3: $140,000
    Q4: $160,000
    """
    
    try:
        print(f"Query: '{query}'")
        print("Processing...")
        
        # Test the complete workflow
        formatted_response = intelligent_response_formatter.format_response_with_streamlit_components(
            query, raw_response
        )
        
        print(f"✅ Response Type: {formatted_response.response_type.value}")
        print(f"✅ Has Content: {len(formatted_response.content) > 0}")
        print(f"✅ Has Metadata: {len(formatted_response.metadata) > 0}")
        print(f"✅ Streamlit Components: {len(formatted_response.streamlit_components)}")
        
        # Show first 200 chars of formatted content
        content_preview = formatted_response.content[:200] + "..." if len(formatted_response.content) > 200 else formatted_response.content
        print(f"\nFormatted Content Preview:\n{content_preview}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


def test_fallback_system():
    """Test the fallback keyword-based system"""
    print("\n🔄 Testing Fallback System")
    print("=" * 40)
    
    test_cases = [
        ("compare product A vs B", ResponseType.COMPARISON),
        ("list the features", ResponseType.LIST),
        ("show me a table", ResponseType.TABLE),
        ("create a chart", ResponseType.CHART),
        ("how to install", ResponseType.PROCESS_FLOW),
        ("give me a summary", ResponseType.SUMMARY),
        ("show code example", ResponseType.CODE_BLOCK),
    ]
    
    correct = 0
    for query, expected in test_cases:
        try:
            # Use the unified fallback method directly
            intent = intelligent_response_formatter._unified_fallback_analysis(query)
            
            print(f"Query: '{query}' → {intent.response_type.value} (expected: {expected.value})")
            
            if intent.response_type == expected:
                print("✅ CORRECT")
                correct += 1
            else:
                print("❌ INCORRECT")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    accuracy = (correct / len(test_cases)) * 100
    print(f"\n📊 Fallback Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)})")
    
    return accuracy >= 70


def main():
    """Run basic tests"""
    print("🚀 Basic Intelligent Response Formatting Test")
    print("=" * 60)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Fallback System", test_fallback_system),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        try:
            print(f"\n{'='*20} {test_name} {'='*20}")
            if test_func():
                print(f"✅ {test_name} PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print(f"\n{'='*60}")
    print(f"📊 Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All basic tests passed!")
        return True
    else:
        print("⚠️ Some tests failed.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)